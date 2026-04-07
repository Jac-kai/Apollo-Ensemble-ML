# -------------------- Import Modules --------------------
import logging
from pprint import pprint

from Apollo.Apollo_ML_Engine import ApolloEngine
from Apollo.Apollo_Model_Menu_Helper import select_from_options
from Apollo.Menu_Config import PERMUTATION_IMPORTANCE_CONFIG, SCORING_CONFIG
from Apollo.Menu_Helper_Decorator import (
    input_int,
    input_list,
    input_yesno,
    menu_wrapper,
)

logger = logging.getLogger("Apollo")


# -------------------- Helper: select target column for multi-output plot --------------------
def _select_multioutput_target_col(apollo: ApolloEngine) -> tuple[str | None, bool]:
    """
    Select a target column for multi-output evaluation workflows.

    This helper inspects the current model stored in ``ApolloEngine`` and
    determines whether the active test target behaves like a multi-output
    target table with multiple named columns.

    Behavior
    --------
    - If no current model is available, the helper prints a warning and returns
      ``(None, False)``.
    - If ``Y_test`` is unavailable or does not expose ``columns``, the helper
      treats the workflow as non-multi-output and returns ``(None, False)``.
    - If the target table has one or zero columns, the helper also returns
      ``(None, False)`` because no explicit target-column selection is needed.
    - If multiple target columns are available, the helper displays them,
      prompts the user to select one, and returns the selected column name
      together with ``True``.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    tuple[str | None, bool]
        A tuple ``(target_col, selection_required)`` where:

        - ``target_col`` is the selected target column name when the user makes
          a valid selection, otherwise ``None``.
        - ``selection_required`` is ``True`` when a multi-output target
          selection workflow was actually shown to the user, otherwise ``False``.

    Notes
    -----
    If the user cancels or enters an invalid selection during a required
    multi-output target selection step, this helper returns ``(None, True)``.
    """
    logger.info("Selecting target column for multi-output workflow")

    if apollo.current_model is None:
        logger.warning("Target-column selection failed: no current model")
        print("⚠️ No current model available ‼️")
        return None, False

    y_test = getattr(apollo.current_model, "Y_test", None)

    if y_test is None or not hasattr(y_test, "columns"):
        logger.info(
            "Multi-output target selection skipped: Y_test is unavailable or has no columns"
        )
        return None, False

    col_map = {i: col for i, col in enumerate(y_test.columns, 1)}

    if len(col_map) <= 1:
        logger.info(
            "Multi-output target selection skipped: only one or zero target columns available"
        )
        return None, False

    logger.info("Displaying multi-output target columns: %s", list(y_test.columns))
    print("---------- 🔥 Target Column List 🔥 ----------")
    for idx, col in col_map.items():
        print(f"🍒 {idx}. {col}")

    selected_num = input_int("🎯 Select target column number for multi-output")
    if selected_num is None:
        logger.info("Target-column selection cancelled")
        return None, True

    if selected_num not in col_map:
        logger.warning(
            "Target-column selection failed: invalid selection %s", selected_num
        )
        print("⚠️ Invalid target column selection ‼️")
        return None, True

    target_col = col_map[selected_num]
    logger.info("Selected target column: %s", target_col)
    return target_col, True


# -------------------- Helper: tree plot index conversion --------------------
def _parse_int_list_input(raw_values: list[str] | str | None) -> list[int] | None:
    """
    Convert comma-separated list input into a validated integer list.

    This helper is designed for use together with ``input_list()`` in menu
    workflows that accept one or more integer indices entered in a single
    comma-separated input line.

    The function converts the raw string items returned by ``input_list()`` into
    integers, validates that all values are non-negative, removes duplicates
    while preserving the original input order, and returns the cleaned integer
    list.

    Parameters
    ----------
    raw_values : list[str] or str or None
        Raw return value from ``input_list()``.

        Typical cases include:
        - ``list[str]`` when the user enters comma-separated items such as
          ``"0,1,3"``
        - ``"__BACK__"`` when the user chooses back
        - ``None`` when the user skips input or input parsing fails upstream

    Returns
    -------
    list[int] or None
        Cleaned integer list when parsing succeeds, otherwise ``None``.

        ``None`` is returned when:
        - the input is ``None``
        - the input is ``"__BACK__"``
        - any item cannot be converted to ``int``
        - any parsed value is negative

    Notes
    -----
    - Duplicate values are removed while preserving their first-seen order.
    - Negative integers are treated as invalid because this helper is intended
      for menu index selection workflows.
    - The caller should handle the semantic meaning of the parsed indices, such
      as valid index range checks against estimator counts or feature counts.
    - Although first introduced for tree-plot index selection, this helper can
      also be reused for any other menu workflow that accepts comma-separated
      integer indices.
    """
    if raw_values is None or raw_values == "__BACK__":
        return None

    try:
        values = [int(v) for v in raw_values]
    except (TypeError, ValueError):
        return None

    if any(v < 0 for v in values):
        return None

    deduped = []
    seen = set()
    for v in values:
        if v not in seen:
            deduped.append(v)
            seen.add(v)

    return deduped


# -------------------- Helper: tree plot index amount --------------------
def _get_tree_plot_index_range(apollo: ApolloEngine) -> tuple[int, int] | None:
    """
    Get the selectable tree-index range for the current Apollo model.

    This helper inspects the active model stored in ``ApolloEngine`` and tries
    to determine how many fitted tree estimators are currently available for
    tree-plot visualization.

    The helper currently supports fitted mission-layer models that expose one of
    the following retrieval methods:

    - ``get_fitted_bagging_estimator()``
    - ``get_fitted_gradient_boosting_estimator()``

    If a supported fitted estimator is found and it exposes ``estimators_``,
    the helper returns the valid selectable tree-index range as
    ``(0, total_estimators - 1)``.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    tuple[int, int] or None
        A tuple ``(min_index, max_index)`` when the current model exposes fitted
        tree estimators, otherwise ``None``.

        ``None`` is returned when:
        - no current model is available
        - the current model does not support a recognized fitted-estimator
          retrieval method
        - the fitted estimator is unavailable
        - the fitted estimator does not expose ``estimators_``
        - the estimator collection is empty
        - an internal lookup step raises an exception

    Notes
    -----
    - For bagging workflows, the tree count is resolved from
      ``len(fitted.estimators_)``.
    - For gradient boosting workflows, the selectable tree count is resolved
      from ``fitted.estimators_.shape[0]``.
    - This helper is intended for user-facing menu hints only. Actual range
      validation should still be enforced by the downstream plotting engine.
    """
    if apollo.current_model is None:
        return None

    model = apollo.current_model

    try:
        if hasattr(model, "get_fitted_bagging_estimator"):
            fitted = model.get_fitted_bagging_estimator()
            if fitted is not None and hasattr(fitted, "estimators_"):
                total = len(fitted.estimators_)
                if total > 0:
                    return 0, total - 1

        if hasattr(model, "get_fitted_gradient_boosting_estimator"):
            fitted = model.get_fitted_gradient_boosting_estimator()
            if fitted is not None and hasattr(fitted, "estimators_"):
                total = fitted.estimators_.shape[0]
                if total > 0:
                    return 0, total - 1

    except Exception:
        return None

    return None


# -------------------- Show evaluation result menu --------------------
@menu_wrapper("Show Evaluation Result")
def show_evaluation_result_menu(apollo: ApolloEngine):
    """
    Display the stored evaluation result of the current Apollo model.

    This menu checks whether the current Apollo runtime contains a model result
    dictionary and whether that result includes an ``evaluation`` field. When
    available, the evaluation content is printed in a readable form using
    ``pprint``.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    If no current model result exists, or if the stored model result does not
    contain an ``evaluation`` field, the menu prints a warning and exits.
    """
    logger.info("Entered menu: Show Evaluation Result")

    if apollo.current_model_result is None:
        logger.warning("No current model result available")
        print("⚠️ No evaluation result available yet ‼️")
        return

    evaluation = apollo.current_model_result.get("evaluation")
    if evaluation is None:
        logger.warning("No evaluation field found in current_model_result")
        print("⚠️ No evaluation result available yet ‼️")
        return

    print("\n---------- 🔥 Evaluation Result 🔥 ----------")
    pprint(evaluation)
    logger.info("Evaluation result displayed successfully")
    print("-" * 100)


# -------------------- Prediction preview menu --------------------
@menu_wrapper("Prediction Preview")
def prediction_preview_menu(apollo: ApolloEngine):
    """
    Display a preview of model predictions from the current Apollo model.

    This menu validates that a current model exists and supports the
    ``predict_engine()`` workflow. It then runs the prediction workflow and
    shows a short prediction preview.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    - If the model exposes ``prediction_preview``, that preview is displayed.
    - Otherwise, the menu attempts to display the first few prediction values.
    - If prediction fails or preview generation is unsupported, the menu prints
      a warning and exits.
    """
    logger.info("Entered menu: Prediction Preview")

    if apollo.current_model is None:
        logger.warning("Prediction Preview failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    if not hasattr(apollo.current_model, "predict_engine"):
        logger.warning("Prediction Preview failed: current model has no predict_engine")
        print("⚠️ Prediction preview is not supported by the current model ‼️")
        return

    try:
        preds = apollo.current_model.predict_engine()
    except Exception as e:
        logger.warning("Prediction Preview failed: %s", e)
        print(f"⚠️ Prediction preview failed: {e}")
        return

    preview = getattr(apollo.current_model, "prediction_preview", None)
    if preview is None:
        try:
            preview = preds[:5]
        except Exception:
            preview = preds

    print("\n---------- 🔥 Prediction Preview 🔥 ----------")
    pprint(preview)
    logger.info(
        "Prediction preview displayed successfully | preview_size=%s", len(preview)
    )
    print("-" * 100)


# -------------------- Predict probability menu --------------------
@menu_wrapper("Predict Probability")
def predict_probability_menu(apollo: ApolloEngine):
    """
    Display a preview of class-probability predictions from the current model.

    This menu validates that a current model exists and supports the
    ``predict_proba_engine()`` workflow. It then runs probability prediction
    and displays a short preview of the resulting probability output.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    If predict-probability is unsupported or the probability workflow raises an
    error, the menu prints a warning and exits.
    """
    logger.info("Entered menu: Predict Probability")

    if apollo.current_model is None:
        logger.warning("Predict Probability failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    if not hasattr(apollo.current_model, "predict_proba_engine"):
        logger.warning(
            "Predict Probability failed: current model has no predict_proba_engine"
        )
        print("⚠️ Predict probability is not supported by the current model ‼️")
        return

    try:
        proba = apollo.current_model.predict_proba_engine()
    except Exception as e:
        logger.warning("Predict Probability failed: %s", e)
        print(f"⚠️ Predict probability failed: {e}")
        return

    try:
        preview = proba[:5]
    except Exception:
        preview = proba

    print("\n---------- 🔥 Predict Probability Preview 🔥 ----------")
    pprint(preview)
    logger.info("Predict probability preview displayed successfully")
    print("-" * 100)


# -------------------- Permutation importance menu --------------------
@menu_wrapper("Permutation Importance")
def permutation_importance_menu(apollo: ApolloEngine):
    """
    Run and display permutation importance for the current Apollo model.

    This menu validates that a current model exists and supports the
    ``permutation_importance_engine(...)`` workflow. It then interactively
    collects permutation-importance options from the user, including:

    1. the number of permutation repeats,
    2. the maximum number of displayed features,
    3. the scoring metric used to measure performance drop,
    4. whether the generated figure should be saved.

    The scoring menu is selected dynamically according to the current model task:

    - ``"classification"`` -> classifier scoring menu
    - ``"regression"`` -> regressor scoring menu

    After collecting the required options, the menu runs the underlying
    permutation-importance workflow and displays the returned result.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    - This menu requires that ``apollo.current_model`` already exists.
    - The current model must expose a ``permutation_importance_engine(...)``
    method.
    - The scoring-menu type is resolved from ``apollo.current_model.task``.
    - If the task type is unsupported, the menu prints a warning and exits.
    - If the user cancels any required option-selection step, the menu exits
    without running permutation importance.
    - If the returned result object supports ``head(...)``, the menu displays
    the top portion of the result for readability.
    - If permutation importance fails inside the model layer, the raised
    exception is caught, a warning is printed, and the menu returns normally.

    Examples
    --------
    Example user inputs during the menu flow::

        repeats      -> 2   # option mapped to 10
        max display  -> 2   # option mapped to 20
        scoring      -> 3   # e.g. "f1_weighted" or task-specific option
        save plot    -> n

    This runs permutation importance using the selected repeat count,
    display limit, scoring metric, and save option.
    """
    logger.info("Entered menu: Permutation Importance")

    if apollo.current_model is None:
        logger.warning("Permutation Importance failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    if not hasattr(apollo.current_model, "permutation_importance_engine"):
        logger.warning("Permutation Importance failed: method unavailable")
        print("⚠️ Permutation importance is not supported by the current model ‼️")
        return

    # ---------- Resolve task type for scoring menu ----------
    task_type = getattr(apollo.current_model, "task", None)
    if task_type == "classification":
        scoring_task_type = "classifier"
    elif task_type == "regression":
        scoring_task_type = "regressor"
    else:
        logger.warning("Permutation Importance failed: unknown task type %s", task_type)
        print("⚠️ Unknown model task type for permutation importance scoring ‼️")
        return

    # ---------- Select n_repeats ----------
    repeats_config = PERMUTATION_IMPORTANCE_CONFIG["n_repeats"]
    n_repeats = select_from_options(
        label=repeats_config["label"],
        options=repeats_config["options"],
        default=repeats_config["default"],
    )
    if n_repeats is None:
        logger.info("Permutation Importance cancelled at n_repeats selection")
        return

    # ---------- Select max_display ----------
    display_config = PERMUTATION_IMPORTANCE_CONFIG["max_display"]
    max_display = select_from_options(
        label=display_config["label"],
        options=display_config["options"],
        default=display_config["default"],
    )
    if max_display is None:
        logger.info("Permutation Importance cancelled at max_display selection")
        return

    # ---------- Select scoring ----------
    scoring_config = SCORING_CONFIG[scoring_task_type]
    scoring = select_from_options(
        label=scoring_config["label"],
        options=scoring_config["options"],
        default=scoring_config["default"],
    )
    if scoring is None:
        logger.info("Permutation Importance cancelled at scoring selection")
        return

    # ---------- Save figure ----------
    save_fig = input_yesno("💾 Save plot", default=False)
    if save_fig is None:
        logger.info("Permutation Importance cancelled at save option")
        return

    try:
        result = apollo.current_model.permutation_importance_engine(
            n_repeats=n_repeats,
            scoring=scoring,
            max_display=max_display,
            save_fig=save_fig,
        )
    except Exception as e:
        logger.warning("Permutation Importance failed: %s", e)
        print(f"⚠️ Permutation importance failed: {e}")
        return

    print("\n---------- 🔥 Permutation Importance Result 🔥 ----------")
    pprint(result.head(max_display) if hasattr(result, "head") else result)
    logger.info(
        "Permutation importance displayed successfully | n_repeats=%s | max_display=%s | scoring=%s | save_fig=%s",
        n_repeats,
        max_display,
        scoring,
        save_fig,
    )
    print("-" * 100)


# -------------------- Feature importance menu --------------------
@menu_wrapper("Feature Importance")
def feature_importance_menu(apollo: ApolloEngine):
    """
    Run and display feature importance for the current model.

    This menu validates that a current model exists and supports the
    ``feature_importance_engine(...)`` workflow. It asks the user whether the
    generated figure should be saved, then runs the feature-importance workflow
    and displays the result.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    If the underlying result object supports ``head(...)``, the menu displays
    the top portion of the result for readability.
    """
    logger.info("Entered menu: Feature Importance")

    if apollo.current_model is None:
        logger.warning("Feature Importance failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    if not hasattr(apollo.current_model, "feature_importance_engine"):
        logger.warning("Feature Importance failed: method unavailable")
        print("⚠️ Feature importance is not supported by the current model ‼️")
        return

    save_fig = input_yesno("💾 Save plot", default=False)
    if save_fig is None:
        logger.info("Feature Importance cancelled at save option")
        return

    try:
        result = apollo.current_model.feature_importance_engine(save_fig=save_fig)
    except Exception as e:
        logger.warning("Feature Importance failed: %s", e)
        print(f"⚠️ Feature importance failed: {e}")
        return

    print("\n---------- 🔥 Feature Importance Result 🔥 ----------")
    pprint(result.head(10) if hasattr(result, "head") else result)
    logger.info("Feature importance displayed successfully | save_fig=%s", save_fig)
    print("-" * 100)


# -------------------- Tree plot menu --------------------
@menu_wrapper("Tree Plot")
def tree_plot_menu(apollo: ApolloEngine):
    """
    Generate one or more tree plots for the current Apollo model.

    This menu validates that a current model exists and supports the
    ``tree_plot_engine(...)`` workflow. The user is guided through the tree-plot
    configuration process, including:

    1. selecting the maximum displayed tree depth,
    2. optionally reviewing the available tree-index range,
    3. entering one or more tree indices as comma-separated values,
    4. choosing whether the generated figure(s) should be saved.

    The selected tree indices are collected through ``input_list()`` and then
    converted into validated integer indices using
    ``_parse_int_list_input(...)`` before being passed to the model-layer tree
    plotting engine.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance containing the current trained model.

    Returns
    -------
    None

    Notes
    -----
    - This menu requires that ``apollo.current_model`` already exists.
    - The current model must expose a ``tree_plot_engine(...)`` method.
    - The available depth menu maps numeric choices to concrete ``max_depth``
      values:
      - ``1 -> 3``
      - ``2 -> 5``
      - ``3 -> 10``
      - ``4 -> None``
    - If available, the selectable tree-index range is displayed before the
      user enters tree indices.
    - Tree indices are entered as comma-separated values such as ``0,1,3``.
    - If the user cancels at any step, the menu exits without generating plots.
    - If tree plotting fails inside the model layer, the raised exception is
      caught, a warning is printed, and the menu returns normally.

    Examples
    --------
    Example user inputs during the menu flow::

        max depth  -> 1
        tree index -> 0,2,4
        save plot  -> n

    This requests tree plots for indices ``[0, 2, 4]`` with ``max_depth=3``
    and without saving output files.
    """
    logger.info("Entered menu: Tree Plot")

    if apollo.current_model is None:
        logger.warning("Tree Plot failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    if not hasattr(apollo.current_model, "tree_plot_engine"):
        logger.warning("Tree Plot failed: method unavailable")
        print("⚠️ Tree plot is not supported by the current model ‼️")
        return

    max_depth_menu = {
        1: 3,
        2: 5,
        3: 10,
        4: None,
    }

    print("\n----- 🌳 Tree Plot Max Depth -----")
    for i, value in max_depth_menu.items():
        print(f"🍀 {i}. {value}")
    print("-" * 50)

    selected_num = input_int("🕯️ Select max depth", default=1)
    if selected_num is None:
        logger.info("Tree Plot cancelled at max-depth selection")
        return

    if selected_num not in max_depth_menu:
        logger.warning("Tree Plot failed: invalid max-depth selection %s", selected_num)
        print("⚠️ Invalid selection ‼️")
        return

    index_range = _get_tree_plot_index_range(apollo)
    if index_range is not None:
        min_idx, max_idx = index_range
        print(f"🌲 Available tree indices: {min_idx} ~ {max_idx}")

    raw_tree_indices = input_list("🌲 Select tree index(es)")
    if raw_tree_indices == "__BACK__":
        logger.info("Tree Plot cancelled at tree-index selection")
        return

    tree_indices = _parse_int_list_input(raw_tree_indices)
    if not tree_indices:
        logger.warning(
            "Tree Plot failed: invalid tree-index selection %s", raw_tree_indices
        )
        print("⚠️ Invalid tree index selection ‼️")
        return

    save_fig = input_yesno("💾 Save plot", default=False)
    if save_fig is None:
        logger.info("Tree Plot cancelled at save option")
        return

    try:
        apollo.current_model.tree_plot_engine(
            tree_indices=tree_indices,
            max_depth=max_depth_menu[selected_num],
            save_fig=save_fig,
        )
        logger.info(
            "Tree Plot generated successfully | tree_indices=%s | max_depth=%s | save_fig=%s",
            tree_indices,
            max_depth_menu[selected_num],
            save_fig,
        )
    except Exception as e:
        logger.warning("Tree Plot failed: %s", e)
        print(f"⚠️ Tree plot failed: {e}")


# -------------------- Confusion matrix menu --------------------
@menu_wrapper("Confusion Matrix")
def confusion_matrix_menu(apollo: ApolloEngine):
    """
    Generate a confusion matrix for the current Apollo model.

    This menu validates that a current model exists and supports the
    ``confusion_matrix_engine(...)`` workflow. For multi-output targets, the
    user may first be asked to choose which target column to visualize. The
    user is then asked whether the figure should be saved before the confusion
    matrix is generated.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    If a multi-output target-column selection step is required and the user
    cancels or provides an invalid selection, the menu exits without generating
    a confusion matrix.
    """
    logger.info("Entered menu: Confusion Matrix")

    if apollo.current_model is None:
        logger.warning("Confusion Matrix failed: no current model")
        print("⚠️ No current model available ‼️")
        return

    if not hasattr(apollo.current_model, "confusion_matrix_engine"):
        logger.warning("Confusion Matrix failed: method unavailable")
        print("⚠️ Confusion matrix is not supported by the current model ‼️")
        return

    target_col, selection_required = _select_multioutput_target_col(apollo)
    if selection_required and target_col is None:
        logger.info("Confusion Matrix cancelled at target-column selection")
        return

    save_fig = input_yesno("💾 Save plot", default=False)
    if save_fig is None:
        logger.info("Confusion Matrix cancelled at save option")
        return

    try:
        apollo.current_model.confusion_matrix_engine(
            target_col=target_col,
            save_fig=save_fig,
        )
        logger.info(
            "Confusion Matrix generated successfully | target_col=%s | save_fig=%s",
            target_col,
            save_fig,
        )
    except Exception as e:
        logger.warning("Confusion Matrix failed: %s", e)
        print(f"⚠️ Confusion matrix failed: {e}")


# -------------------- Apollo evaluation menu --------------------
@menu_wrapper("Apollo Evaluation Menu")
def evaluation_menu(apollo: ApolloEngine):
    """
    Display the Apollo evaluation-services menu.

    This dispatcher provides access to evaluation-related workflows for the
    active Apollo model, including:

    - evaluation-result display
    - prediction preview
    - predict-probability preview
    - permutation importance
    - feature importance
    - tree plotting
    - confusion-matrix visualization

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    The same Apollo engine instance is reused across all evaluation actions so
    the current model state and evaluation-related runtime information remain
    available throughout the active session.

    Evaluation-function availability depends on the fitted model type.

    General guidance:
    - ``Show Evaluation Result`` and ``Prediction Preview`` are usually
    available after successful model training.
    - ``Permutation Importance`` is generally available for fitted models that
    support the underlying workflow.
    - ``Predict Probability`` is only available for classifier models whose
    fitted pipeline supports probability prediction.
    - ``Feature Importance`` is only available for models that expose a
    feature-importance workflow or equivalent importance attributes.
    - ``Tree Plot`` is only available for supported tree-based models or
    supported fitted tree-estimator collections.
    - ``Confusion Matrix`` is intended for classification workflows.

    Because Apollo supports multiple ensemble families, not every evaluation
    menu item is guaranteed to be available for every current model.

    If a selected evaluation workflow is unsupported by the current model,
    the corresponding submenu prints a warning and exits safely instead of
    crashing the session.
    """
    logger.info("Entered menu: Apollo Evaluation Menu")

    menu = [
        (1, "📜 Show Evaluation Result", show_evaluation_result_menu),
        (2, "🧪 Predict Probability", predict_probability_menu),
        (3, "📊 Permutation Importance", permutation_importance_menu),
        (4, "🌟 Feature Importance", feature_importance_menu),
        (5, "🌳 Tree Plot", tree_plot_menu),
        (6, "🧮 Confusion Matrix", confusion_matrix_menu),
        (7, "🔮 Prediction Preview", prediction_preview_menu),
        (0, "↩️ Back", None),
    ]
    menu_width = 50

    while True:
        print("🏮  Apollo Evaluation Menu 🏮 ".center(menu_width, "━"))
        for opt, label, _ in menu:
            print(f"{opt}. {label}")
        print("━" * menu_width)

        print("🔔 Notes:")
        print(
            " - Show Evaluation Result / Prediction Preview: usually available after training."
        )
        print(
            " - Predict Probability: only available for models that support probability prediction."
        )
        print(" - Permutation Importance: usually available for fitted models.")
        print(
            " - Feature Importance: only available for models exposing feature importance."
        )
        print(" - Tree Plot: mainly intended for tree-based models.")
        print(
            "   For Bagging, tree plot is most suitable when the base estimator is a single DecisionTree."
        )
        print(
            "   RandomForest-based estimators are better inspected with feature importance or permutation importance."
        )
        print(" - Confusion Matrix: classification workflow only.")
        print("━" * menu_width)

        choice = input_int("🕯️ Select Evaluation Services", default=-1)
        if choice is None:
            logger.info("Exited Apollo Evaluation Menu by cancel/back")
            return

        matched = False
        for opt, label, func in menu:
            if choice == opt:
                logger.info(
                    "Apollo Evaluation Menu selection: %s - %s",
                    choice,
                    label,
                )
                matched = True
                if func is None:
                    logger.info("Exited Apollo Evaluation Menu")
                    return
                func(apollo)
                break

        if not matched:
            logger.warning("Invalid selection in Apollo Evaluation Menu: %s", choice)
            print("⚠️ Invalid selection ‼️")


# -----------------------------------------
