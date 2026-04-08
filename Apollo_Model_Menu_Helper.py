# -------------------- Import Modules --------------------
from __future__ import annotations

from typing import Any, Sequence

from sklearn.decomposition import PCA

from Apollo.Apollo_ML_Engine import APOLLO_MODEL_REGISTRY, ApolloEngine
from Apollo.Estimators_ParamsGrid.Estimators import get_estimator_builder_map
from Apollo.Estimators_ParamsGrid.Params_Grid import get_default_param_grid_for_model
from Apollo.Menu_Config import (
    COMMON_PARAM_CONFIG,
    ENSEMBLE_PARAM_CONFIG,
    PCA_PARAM_CONFIG,
    SCORING_CONFIG,
)
from Apollo.Menu_Helper_Decorator import input_int


# -------------------- Helper: select model name --------------------
def select_model_name(apollo: ApolloEngine, task_type: str) -> str | None:
    """
    Interactively select a registered Apollo ensemble model name for a given task type.

    This helper retrieves the available Apollo model names from the engine using
    the requested task-type filter, displays them as a numbered menu, accepts
    one numeric user selection, and returns the matched model name.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance used to retrieve available model names.
    task_type : str
        Task-type filter used to request compatible model names from the engine.

        Typical values include:

        - ``"classifier"``
        - ``"regressor"``

    Returns
    -------
    str or None
        Selected Apollo model name when a valid menu option is chosen.

        Returns ``None`` if:

        - no compatible models are available,
        - the user cancels the selection,
        - the selected menu number is out of range.

    Notes
    -----
    This helper only resolves a public Apollo model name such as
    ``"VotingClassifier"`` or ``"BaggingRegressor"``.
    It does not instantiate the model object.

    Model availability is delegated to
    ``ApolloEngine.get_available_models(task_type=task_type)``.

    Workflow
    --------
    1. Request available model names from the engine.
    2. Build a numeric menu mapping.
    3. Display the model menu.
    4. Read one numeric selection from the user.
    5. Return the matched model name.

    Examples
    --------
    A classifier workflow may present options such as::

        1. VotingClassifier
        2. BaggingClassifier
        3. AdaBoostClassifier

    A valid selection returns the corresponding model name string.
    """
    # ---------- Get avaliable models ----------
    model_names = apollo.get_available_models(task_type=task_type)

    if not model_names:
        print(f"⚠️ No available {task_type} models found ‼️")
        return None

    # ---------- List avaliable models ----------
    model_map = {i: name for i, name in enumerate(model_names, 1)}

    print(f"\n----- 🔥 Available {task_type.title()} Ensemble Models 🔥 -----")
    for i, name in model_map.items():
        print(f"🧠 {i}. {name}")
    print("-" * 50)

    # ---------- Select avaliable models ----------
    selected_num = input_int("🕯️ Select model", default=-1)
    if selected_num is None:
        return None

    if selected_num not in model_map:
        print("⚠️ Model selection is out of range ‼️")
        return None

    return model_map[selected_num]  # Return the selected model


# -------------------- Helper: select from options --------------------
def select_from_options(
    label: str,
    options: dict[int, object],
    default: int | None = None,
):
    """
    Display a numbered option menu and return both the selected key and value.

    This helper is the shared menu-selection utility used throughout Apollo's
    menu workflows. It prints a menu title, displays all numeric options, reads
    one numeric input from the user, validates the input, and returns both the
    selected numeric key and its mapped option value.

    Parameters
    ----------
    label : str
        Menu title or prompt label shown above the option list.
    options : dict[int, object]
        Mapping from numeric menu keys to option values.

        Example::

            {
                1: True,
                2: False,
            }

        or::

            {
                1: "soft",
                2: "hard",
            }

    default : int or None, optional
        Default numeric selection forwarded to ``input_int(...)``.

        This is the default menu key, not the default option value.

    Returns
    -------
    tuple[int | None, object | None]
        A tuple ``(selected_num, selected_value)``.

        Returns:

        - ``(selected_num, options[selected_num])`` when the user makes a valid
          selection,
        - ``(None, None)`` if the user cancels the selection,
        - ``(None, None)`` if the selected numeric key is not found in
          ``options``.

    Notes
    -----
    This helper is intentionally generic and reusable. It is used by many
    parameter-collection workflows, including:

    - common training parameter selection,
    - ensemble simple-parameter selection,
    - PCA parameter selection,
    - voting-weight selection,
    - default-grid selection.

    Workflow
    --------
    1. Print the menu label.
    2. Print all numeric option pairs.
    3. Read one numeric selection.
    4. Validate that the selected key exists.
    5. Return ``(selected_num, options[selected_num])``.

    Examples
    --------
    If::

        options = {1: "soft", 2: "hard"}

    and the user selects ``2``, the returned value is::

        (2, "hard")

    If the user cancels the input, the returned value is::

        (None, None)
    """
    # ---------- List the parameters ----------
    print(f"\n----- {label} -----")
    for num, value in options.items():
        print(f"📌 {num}. {value}")
    print("-" * 50)

    selected_num = input_int("🕯️ Select option", default=default)
    if selected_num is None:
        return None, None

    if selected_num not in options:
        print("⚠️ Selection is out of range ‼️")
        return None, None

    return selected_num, options[selected_num]

# -------------------- Helper: get task type --------------------
def get_model_task_type(apollo: ApolloEngine, model_name: str) -> str | None:
    """
    Return the registered task type for a given Apollo model name.

    This helper looks up the requested model in ``APOLLO_MODEL_REGISTRY`` and
    returns its registered task type.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

        This parameter is accepted for workflow consistency, even though the
        actual lookup is performed through the module-level Apollo model
        registry rather than through the engine object itself.
    model_name : str
        Registered Apollo model name, such as ``"VotingClassifier"`` or
        ``"StackingRegressor"``.

    Returns
    -------
    str or None
        Registered task type associated with ``model_name``.

        Typical return values include:

        - ``"classifier"``
        - ``"regressor"``

        Returns ``None`` if the model name is not present in the registry.

    Notes
    -----
    This helper is mainly used to connect an Apollo public model name to the
    correct downstream menu logic, such as scoring selection or estimator
    compatibility checks.

    Examples
    --------
    ``"VotingClassifier"`` -> ``"classifier"``

    ``"BaggingRegressor"`` -> ``"regressor"``
    """
    info = APOLLO_MODEL_REGISTRY.get(model_name)
    return info.get("task_type") if info else None


# -------------------- Helper: get training target labels --------------------
def _get_training_target_labels(apollo: ApolloEngine) -> list | None:
    """
    Return sorted unique target labels from the current Apollo feature workflow.

    This helper inspects the target data currently stored in
    ``apollo.feature_core.y`` and attempts to extract the unique observed target
    labels for use in training-menu logic.

    It is primarily used to determine whether the current classification target
    behaves like:

    - numeric binary labels such as ``[0, 1]``,
    - non-numeric binary labels such as ``["e", "p"]``,
    - multiclass labels such as ``["A", "B", "C"]``.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance containing the current feature workflow.

    Returns
    -------
    list or None
        Sorted list of unique target labels when label extraction succeeds.

        Returns ``None`` if:

        - ``apollo`` is unavailable,
        - ``feature_core`` is unavailable,
        - target data ``y`` is unavailable,
        - the target structure is unsupported,
        - label extraction fails.

    Supported Input Shapes
    ----------------------
    - One-dimensional target-like objects such as pandas ``Series`` or other
      iterable single-target containers.
    - Two-dimensional target tables with exactly one column, such as a
      single-column pandas ``DataFrame``.

    If a two-dimensional target object contains more than one column, this
    helper treats it as unsupported for simple label-resolution purposes and
    returns ``None``.

    Notes
    -----
    This helper does not perform label encoding or label conversion.
    It only extracts and sorts the currently observed target labels.

    Missing values are excluded when the inspected target object supports
    ``dropna()``.

    This helper is mainly used to support dynamic scoring-menu filtering for
    classifier training workflows.
    """
    if apollo is None or getattr(apollo, "feature_core", None) is None:
        return None

    y = getattr(apollo.feature_core, "y", None)
    if y is None:
        return None

    try:
        if hasattr(y, "shape") and len(getattr(y, "shape", [])) == 2:
            if y.shape[1] != 1:
                return None
            y_series = y.iloc[:, 0]
        else:
            y_series = y

        unique_labels = (
            list(y_series.dropna().unique())
            if hasattr(y_series, "dropna")
            else list(set(y_series))
        )
        return sorted(unique_labels)
    except Exception:
        return None
    

# -------------------- Helper: training scoring options --------------------
def _get_training_scoring_config(
    task_type: str,
    apollo: ApolloEngine | None = None,
) -> dict:
    """
    Return the training scoring-menu configuration for the current workflow.

    This helper builds the scoring menu used by Apollo training workflows
    according to the requested task type and, for classifier workflows, the
    structure of the currently selected target labels.

    Behavior
    --------
    - If ``task_type == "regressor"``, the helper returns the configured
      regression scoring menu from ``SCORING_CONFIG["regressor"]``.
    - If ``task_type`` is not ``"classifier"`` or ``"regressor"``, the helper
      returns an empty scoring configuration.
    - If ``task_type == "classifier"``, the helper inspects the current target
      labels and chooses either the full classifier scoring menu or a safer
      reduced classifier scoring menu.

    Classification Logic
    --------------------
    For classifier workflows, target labels are resolved through
    ``_get_training_target_labels(apollo)``.

    The helper then applies the following rules:

    - If label resolution fails, use the safer reduced classifier menu.
    - If the labels are exactly numeric binary labels ``{0, 1}``, return the
      full classifier scoring menu from ``SCORING_CONFIG["classifier"]``.
    - If the labels are binary but not numeric ``0/1`` labels, return the
      safer reduced classifier menu.
    - If the labels are multiclass, return the safer reduced classifier menu.

    The safer reduced classifier menu includes:

    - ``accuracy``
    - ``f1_weighted``
    - ``precision_weighted``
    - ``recall_weighted``

    This avoids presenting binary-style scoring options that may fail during
    CV-based model selection when the current target labels do not match the
    default positive-label assumptions used by some scorers.

    Parameters
    ----------
    task_type : str
        Training task type.

        Typical values include:

        - ``"classifier"``
        - ``"regressor"``
    apollo : ApolloEngine or None, optional
        Active Apollo engine instance used to inspect current target labels for
        classifier workflows.

    Returns
    -------
    dict
        Scoring-menu configuration dictionary with the standard structure:

        - ``"label"``: menu title string
        - ``"options"``: numeric-option mapping
        - ``"default"``: default menu key

        The returned dictionary may contain an empty ``"options"`` mapping when
        no valid scoring menu can be constructed.

    Notes
    -----
    This helper does not train a model and does not run cross-validation.
    Its responsibility is limited to preparing the scoring options shown in the
    training menu before model fitting begins.

    This helper is used only for training scoring selection and is separate
    from the permutation-importance scoring helper used in the evaluation
    workflow.
    """
    if task_type == "regressor":
        return SCORING_CONFIG["regressor"]

    if task_type != "classifier":
        return {
            "label": "🎯 Scoring",
            "options": {},
            "default": None,
        }

    safe_classifier_options = {
        1: "accuracy",
        2: "f1_weighted",
        3: "precision_weighted",
        4: "recall_weighted",
    }

    labels = _get_training_target_labels(apollo)

    if not labels:
        return {
            "label": "🎯 Scoring",
            "options": safe_classifier_options,
            "default": 2,
        }

    if len(labels) == 2 and set(labels) == {0, 1}:
        return SCORING_CONFIG["classifier"]

    if len(labels) == 2:
        return {
            "label": "🎯 Scoring",
            "options": safe_classifier_options,
            "default": 2,
        }

    return {
        "label": "🎯 Scoring",
        "options": safe_classifier_options,
        "default": 2,
    }


# -------------------- Helper: collect common training params --------------------
def collect_common_training_params(
    task_type: str,
    apollo: ApolloEngine | None = None,
) -> dict | None:
    """
    Collect common training parameters for an Apollo model workflow.

    This helper interactively gathers shared training parameters used across
    Apollo model-training workflows, including:

    - train/test split settings,
    - random-state settings,
    - cross-validation usage,
    - preprocessing usage,
    - scoring method selection.

    For classifier workflows, the scoring menu is selected dynamically
    according to the current target-label structure so that binary-style
    scoring options that may be incompatible with non-numeric class labels are
    not shown unnecessarily.

    Parameters
    ----------
    task_type : str
        Task type used to determine which scoring menu should be shown.

        Typical values include:

        - ``"classifier"``
        - ``"regressor"``
    apollo : ApolloEngine or None, optional
        Active Apollo engine instance used to inspect the current feature/target
        workflow when dynamic classifier scoring selection is needed.

    Returns
    -------
    dict or None
        Dictionary containing the collected common training parameters when all
        required menu selections are completed successfully.

        The returned dictionary contains:

        - ``test_size``
        - ``split_random_state``
        - ``model_random_state``
        - ``use_cv``
        - ``use_preprocess``
        - ``cv_folds``
        - ``scoring``

        Returns ``None`` if the user cancels during any required selection
        step or if no valid scoring options are available.

    Notes
    -----
    This helper is shared by both classifier and regressor training menus.

    Scoring Selection
    -----------------
    - Regressor workflows use the configured regression scoring menu directly.
    - Classifier workflows use a dynamic scoring menu based on the current
      target-label structure.
    - When the current classifier target labels are not numeric binary
      ``0/1`` labels, the helper may show a note and restrict the available
      scoring options to safer weighted metrics plus accuracy.

    When cross-validation is disabled, ``cv_folds`` is stored as ``None``.

    Workflow
    --------
    1. Collect shared training parameters from ``COMMON_PARAM_CONFIG``:
       - test size
       - split random state
       - model random state
       - cross-validation usage
       - outer preprocessing usage
    2. If cross-validation is enabled, collect the CV fold count.
    3. Build the task-appropriate scoring menu.
    4. Collect the scoring method from that menu.
    5. Return the final common-parameter dictionary.

    Examples
    --------
    A classifier workflow may return something like::

        {
            "test_size": 0.2,
            "split_random_state": 42,
            "model_random_state": 42,
            "use_cv": True,
            "use_preprocess": True,
            "cv_folds": 5,
            "scoring": "f1_weighted",
        }

    A regressor workflow may return something like::

        {
            "test_size": 0.25,
            "split_random_state": 7,
            "model_random_state": 7,
            "use_cv": False,
            "use_preprocess": True,
            "cv_folds": None,
            "scoring": "r2",
        }
    """
    params = {}

    # ---------- Get common training parameters ----------
    for param_name in (
        "test_size",
        "split_random_state",
        "model_random_state",
        "use_cv",
        "use_preprocess",
    ):
        config = COMMON_PARAM_CONFIG[param_name]
        selected_num, selected_value = select_from_options(
            label=config["label"],
            options=config["options"],
            default=config["default"],
        )
        if selected_num is None:
            return None
        params[param_name] = selected_value

    # ---------- Get CV fold parameter ----------
    if params["use_cv"]:
        config = COMMON_PARAM_CONFIG["cv_folds"]
        selected_num, selected_value = select_from_options(
            label=config["label"],
            options=config["options"],
            default=config["default"],
        )
        if selected_num is None:
            return None
        params["cv_folds"] = selected_value
    else:
        params["cv_folds"] = None

    # ---------- Get scoring method ----------
    scoring_config = _get_training_scoring_config(
        task_type=task_type,
        apollo=apollo,
    )

    if not scoring_config["options"]:
        print("⚠️ No valid scoring options available ‼️")
        return None

    if task_type == "classifier":
        labels = _get_training_target_labels(apollo)
        if labels is not None and not (len(labels) == 2 and set(labels) == {0, 1}):
            print(
                "🔔 Note: binary scoring like f1 may require numeric binary labels (0/1). "
                "Safer weighted scoring options are shown for the current target labels."
            )

    selected_num, scoring_value = select_from_options(
        label=scoring_config["label"],
        options=scoring_config["options"],
        default=scoring_config["default"],
    )
    if selected_num is None:
        return None

    params["scoring"] = scoring_value
    return params  # Return collected common training parameters


# -------------------- Helper: collect extra steps --------------------
def collect_extra_steps(feature_count: int | None = None):
    """
    Collect optional extra outer-pipeline steps for Apollo training workflows.

    This helper currently supports optional PCA configuration. It first asks
    whether PCA should be included in the outer training pipeline and, if so,
    collects a valid ``n_components`` choice and returns the corresponding
    pipeline-step definition.

    Parameters
    ----------
    feature_count : int or None, optional
        Number of currently available feature columns.

        When provided, PCA ``n_components`` options are filtered so that
        options greater than the available feature count are excluded.

        When ``None``, no feature-count-based filtering is applied and all
        configured PCA component options remain available.

    Returns
    -------
    list[tuple[str, object]] or None
        Returns one of the following:

        - ``[("pca", PCA(...))]`` when PCA is enabled and configured,
        - ``[]`` when PCA is not used,
        - ``None`` if the user cancels during selection.

    Notes
    -----
    The returned value is structured as a pipeline-step list so it can be
    merged directly into downstream outer-pipeline construction logic.

    This helper currently supports only PCA, but its return format is designed
    to remain compatible with future extra-step additions.

    Feature-Count Filtering
    -----------------------
    When ``feature_count`` is known, the helper filters PCA ``n_components``
    options using the rule:

    - keep values where ``v is None`` or ``v <= feature_count``

    This prevents clearly invalid PCA component selections from being shown.

    Workflow
    --------
    1. Ask whether PCA should be used.
    2. If PCA is disabled, return an empty list.
    3. Retrieve configured PCA ``n_components`` options.
    4. Optionally filter them using ``feature_count``.
    5. Ask the user to choose ``n_components``.
    6. Return the corresponding PCA pipeline step.

    Examples
    --------
    If the user enables PCA and selects ``n_components=3``, the returned value
    is::

        [("pca", PCA(n_components=3))]
    """
    # ---------- Get PCA parameter ----------
    selected_num, use_pca = select_from_options(
        label=PCA_PARAM_CONFIG["use_pca"]["label"],
        options=PCA_PARAM_CONFIG["use_pca"]["options"],
        default=PCA_PARAM_CONFIG["use_pca"]["default"],
    )

    if selected_num is None:
        return None

    if not use_pca:
        return []

    # ---------- Get PCA n component parameter ----------
    pca_config = PCA_PARAM_CONFIG["pca_n_components"]
    all_options = pca_config["options"]

    # Keep values where v is None or v <= feature_count
    if feature_count is not None:
        filtered_options = {
            k: v for k, v in all_options.items()
            if v is None or v <= feature_count
        }
    else:
        filtered_options = all_options.copy()

    if not filtered_options:
        print("⚠️ No valid PCA component options available ‼️")
        return None

    # ---------- Resolve default key ----------
    default_key = pca_config["default"]
    if default_key not in filtered_options:
        default_key = next(iter(filtered_options))

    # ---------- Select PCA n_components ----------
    selected_num, pca_n_components = select_from_options(
        label=pca_config["label"],
        options=filtered_options,
        default=default_key,
    )

    if selected_num is None:
        return None

    return [("pca", PCA(n_components=pca_n_components))]


# -------------------- Helper: collect ensemble simple params --------------------
def collect_ensemble_simple_params(model_name: str) -> dict | None:
    """
    Collect simple ensemble parameters for a specific Apollo model.

    This helper reads the parameter configuration for the requested ensemble
    model from ``ENSEMBLE_PARAM_CONFIG`` and interactively collects scalar or
    boolean-style parameter values.

    Parameters
    ----------
    model_name : str
        Registry name of the ensemble model.

    Returns
    -------
    dict or None
        Dictionary of collected parameter names and selected values.
        Returns ``None`` if the user cancels during parameter selection.

    Notes
    -----
    This helper is intended for simple menu-driven parameters only. More
    complex estimator-selection workflows are handled by other helpers.
    """
    param_list = ENSEMBLE_PARAM_CONFIG.get(
        model_name, []
    )  # Setup esemble parameters for each esemble model
    kwargs = {}

    # ---------- Collect esemble parameters ----------
    for param_config in param_list:
        selected_num, selected_value = select_from_options(
            label=param_config["label"],
            options=param_config["options"],
            default=param_config.get("default"),
        )
        if selected_num is None:
            return None

        kwargs[param_config["name"]] = selected_value

    return kwargs


# -------------------- Helper: select single estimator --------------------
def select_single_estimator(task_type: str):
    """
    Interactively select a single base estimator for the given task type.

    This helper retrieves the estimator-builder map for the specified task type,
    displays available estimators as a numbered menu, and returns both the
    selected estimator name and a newly built estimator instance.

    Parameters
    ----------
    task_type : str
        Task type used to retrieve available estimator builders, such as
        ``"classifier"`` or ``"regressor"``.

    Returns
    -------
    tuple[str, Any] or None
        A tuple ``(estimator_name, estimator_instance)`` when the selection is
        valid. Returns ``None`` if the user cancels or selects an invalid
        estimator.

    Notes
    -----
    Estimator instances are created immediately by calling the selected builder
    function.
    """
    builder_map = get_estimator_builder_map(task_type=task_type)

    # ---------- List estimator ----------
    print(f"\n----- 🔥 Available {task_type.title()} Estimators 🔥 -----")
    for num, (name, _) in builder_map.items():
        print(f"🧠 {num}. {name}")
    print("-" * 50)

    # ---------- Select estimator ----------
    selected_num = input_int("🕯️ Select estimator", default=-1)
    if selected_num is None:
        return None

    if selected_num not in builder_map:
        print("⚠️ Estimator selection is out of range ‼️")
        return None

    est_name, builder = builder_map[selected_num]
    estimator = builder()

    return est_name, estimator


# -------------------- Helper: collect estimator list --------------------
def collect_estimators(
    task_type: str,
    min_estimators: int = 2,
    max_estimators: int = 5,
) -> list[tuple[str, Any]] | None:
    """
    Collect a list of base estimators for multi-estimator ensemble workflows.

    This helper first asks the user how many base estimators should be included,
    then repeatedly calls ``select_single_estimator(...)`` until the requested
    number of estimator instances has been collected.

    Each selected estimator is returned as a sklearn-style
    ``(alias, estimator_instance)`` pair suitable for downstream ensemble APIs
    such as ``VotingClassifier``, ``VotingRegressor``,
    ``StackingClassifier``, and ``StackingRegressor``.

    If the same estimator type is selected more than once, this helper
    automatically creates unique aliases such as ``"svc"``, ``"svc2"``,
    and ``"svc3"`` so the final estimator names remain unique.

    Parameters
    ----------
    task_type : str
        Task type used to retrieve compatible estimator builders.

        Supported values typically include:

        - ``"classifier"``
        - ``"regressor"``

    min_estimators : int, default=2
        Minimum number of estimators shown in the estimator-count menu.

    max_estimators : int, default=5
        Maximum number of estimators shown in the estimator-count menu.

    Returns
    -------
    list[tuple[str, Any]] or None
        List of ``(alias, estimator_instance)`` tuples when selection succeeds.

        Example return value::

            [
                ("rf", RandomForestClassifier(...)),
                ("svc", SVC(...)),
                ("svc2", SVC(...)),
            ]

        Returns ``None`` if the user cancels during estimator-count selection
        or during any estimator-selection step.

    Notes
    -----
    This helper is intended for ensemble families that require multiple base
    estimators, such as voting and stacking workflows.

    This helper is not used for ensemble families that require only one base
    estimator, such as bagging or AdaBoost.

    Workflow
    --------
    1. Build a numeric menu from ``min_estimators`` to ``max_estimators``.
    2. Ask the user how many base estimators to include.
    3. Repeatedly call ``select_single_estimator(task_type=task_type)``.
    4. Track how many times each estimator name has been selected.
    5. Generate unique aliases for repeated estimator types.
    6. Return the final estimator list.
    """
    # ---------- Count estimator ----------
    count_options = {i: i for i in range(min_estimators, max_estimators + 1)}

    selected_num, estimator_count = select_from_options(
        label="🧮 Number of Base Estimators",
        options=count_options,
        default=min_estimators,
    )
    if selected_num is None:
        return None

    # ---------- Initialize collectors ----------
    estimators = []
    used_names = {}

    for idx in range(estimator_count):
        selected = select_single_estimator(task_type=task_type)
        if selected is None:
            return None

        base_name, estimator = selected

        used_names[base_name] = used_names.get(base_name, 0) + 1
        alias = (
            base_name
            if used_names[base_name] == 1
            else f"{base_name}{used_names[base_name]}"
        )

        estimators.append((alias, estimator))

    return estimators


# -------------------- Helper: select final estimator (stacking esmeble model) --------------------
def select_final_estimator(task_type: str):
    """
    Select the final estimator used in stacking ensemble workflows.

    This helper is a lightweight wrapper around
    ``select_single_estimator(task_type=task_type)`` and is used to collect the
    final estimator for stacking models.

    Parameters
    ----------
    task_type : str
        Task type used to retrieve a compatible final-estimator selection menu.

        Typical values include:

        - ``"classifier"``
        - ``"regressor"``

    Returns
    -------
    tuple[str, Any] or None
        A tuple ``(estimator_name, estimator_instance)`` when a valid final
        estimator is selected.

        Returns ``None`` if the user cancels or selects an invalid option.

    Notes
    -----
    This helper is intended specifically for stacking workflows, where the
    final estimator is conceptually different from the list of base estimators.

    Although it delegates directly to ``select_single_estimator(...)``, keeping
    it as a dedicated helper improves readability and makes the stacking
    workflow easier to understand.

    Examples
    --------
    A stacking-classifier workflow may return something like::

        ("logistic", LogisticRegression(...))

    A stacking-regressor workflow may return something like::

        ("ridge", Ridge(...))
    """
    selected = select_single_estimator(task_type=task_type)
    if selected is None:
        return None

    return selected


# -------------------- Helper: ask default param grid --------------------
def maybe_get_default_param_grid(
    model_name: str,
    use_cv: bool,
    final_estimator_name: str | None = None,
) -> dict | None:
    """
    Optionally return a predefined default parameter grid for CV-based workflows.

    This helper is responsible for deciding whether a default GridSearchCV
    parameter grid should be attached to the current training workflow.

    The helper only offers a default grid when cross-validation is enabled.
    If the user agrees, the grid is retrieved from
    ``get_default_param_grid_for_model(...)``.

    Parameters
    ----------
    model_name : str
        Registered Apollo ensemble model name used to resolve the appropriate
        default parameter grid.
    use_cv : bool
        Whether cross-validation is enabled for the current training workflow.

        If ``False``, this helper returns ``None`` immediately.
    final_estimator_name : str or None, optional
        Optional final-estimator name used by model families whose default grid
        depends on the chosen final estimator, such as stacking workflows.

    Returns
    -------
    dict or None
        Default parameter-grid dictionary when:

        - cross-validation is enabled,
        - the user chooses to use the predefined default grid,
        - a compatible default grid exists.

        Returns ``None`` if:

        - cross-validation is disabled,
        - the user declines the default grid,
        - the user cancels the menu step,
        - no compatible default grid is available.

    Notes
    -----
    This helper does not build custom parameter grids.
    Its responsibility is limited to controlling whether an existing predefined
    default grid should be used.

    Behavioral Summary
    ------------------
    - ``use_cv=False`` -> return ``None``
    - ``use_cv=True`` and user selects default grid -> return predefined grid
    - ``use_cv=True`` and user declines default grid -> return ``None``

    When the returned value is ``None``, downstream training may still proceed
    using only the fixed training parameters collected elsewhere.

    Workflow
    --------
    1. If CV is disabled, return ``None``.
    2. Ask whether the predefined default grid should be used.
    3. If declined or cancelled, return ``None``.
    4. Otherwise retrieve and return the matched default parameter grid.
    """
    if not use_cv:
        return None

    # ---------- Using default CV prids for training ----------
    selected_num, use_default_grid = select_from_options(
        label="🧪 Use Default Param Grid",
        options={1: True, 2: False},
        default=1,
    )
    if selected_num is None:
        return None

    if not use_default_grid:
        return None

    return get_default_param_grid_for_model(
        model_name=model_name,
        final_estimator_name=final_estimator_name,
    )


# -------------------- Helper: select voting weights --------------------
def _select_voting_weights(estimators: Sequence[tuple[str, Any]]) -> list[int] | None:
    """
    Interactively select voting weights for a voting ensemble workflow.

    This helper builds a predefined weight-selection menu based on the number
    of selected base estimators and returns the chosen weight list.

    The selected weights are aligned with the order of the supplied
    ``estimators`` sequence. For example, if the estimator list is::

        [("rf", ...), ("svc", ...), ("knn", ...)]

    then the returned weight list ``[2, 1, 1]`` means:

    - weight 2 for ``"rf"``
    - weight 1 for ``"svc"``
    - weight 1 for ``"knn"``

    Parameters
    ----------
    estimators : Sequence[tuple[str, Any]]
        Sequence of selected estimator tuples used to determine the required
        voting-weight length.

    Returns
    -------
    list[int] or None
        Selected voting-weight list when a valid menu option is chosen.

        Returns a uniform weight list ``[1] * est_count`` automatically when
        the estimator count falls outside the explicitly supported menu sizes.

        Returns ``None`` if the user cancels during weight selection.

    Notes
    -----
    Predefined voting-weight menus are currently provided for:

    - 2 estimators
    - 3 estimators
    - 4 estimators
    - 5 estimators

    The default menu option is the uniform-weight configuration, such as:

    - ``[1, 1]``
    - ``[1, 1, 1]``
    - ``[1, 1, 1, 1]``
    - ``[1, 1, 1, 1, 1]``

    If a future workflow supplies an unsupported estimator count, this helper
    falls back to a uniform-weight list instead of raising an error.

    Workflow
    --------
    1. Count the number of selected estimators.
    2. Build a matching predefined weight-option menu.
    3. Show the weight menu through ``select_from_options(...)``.
    4. Return the selected weight list.

    Examples
    --------
    For 3 estimators, the menu may include options such as::

        1 -> [1, 1, 1]
        2 -> [2, 1, 1]
        3 -> [1, 2, 1]
        4 -> [1, 1, 2]

    For 5 estimators, the menu may include options such as::

        1 -> [1, 1, 1, 1, 1]
        2 -> [2, 1, 1, 1, 1]
        3 -> [1, 2, 1, 1, 1]
    """
    est_count = len(estimators)

    if est_count == 2:
        options = {
            1: [1, 1],
            2: [2, 1],
            3: [1, 2],
        }
    elif est_count == 3:
        options = {
            1: [1, 1, 1],
            2: [2, 1, 1],
            3: [1, 2, 1],
            4: [1, 1, 2],
            5: [2, 2, 1],
            6: [2, 1, 2],
            7: [1, 2, 2],
        }
    elif est_count == 4:
        options = {
            1: [1, 1, 1, 1],
            2: [2, 1, 1, 1],
            3: [1, 2, 1, 1],
            4: [1, 1, 2, 1],
            5: [1, 1, 1, 2],
            6: [2, 2, 1, 1],
            7: [2, 1, 2, 1],
            8: [1, 2, 1, 2],
        }
    elif est_count == 5:
        options = {
            1: [1, 1, 1, 1, 1],
            2: [2, 1, 1, 1, 1],
            3: [1, 2, 1, 1, 1],
            4: [1, 1, 2, 1, 1],
            5: [1, 1, 1, 2, 1],
            6: [1, 1, 1, 1, 2],
            7: [2, 2, 1, 1, 1],
            8: [2, 1, 2, 1, 1],
            9: [1, 2, 2, 1, 1],
        }
    else:
        return [1] * est_count

    selected_num, weights = select_from_options(
        label="⚖️ Voting Weights",
        options=options,
        default=1,
    )

    if selected_num is None:
        return None

    return weights


# -------------------- Helper: collect ensemble train kwargs --------------------
def collect_ensemble_train_kwargs(
    model_name: str,
    task_type: str,
    use_cv: bool,
    use_preprocess: bool,
    model_random_state: int,
    feature_count: int | None = None,
) -> dict | None:
    """
    Collect ensemble-specific training keyword arguments for Apollo workflows.

    This helper acts as the main ensemble-parameter coordinator before
    ``ApolloEngine.train_model(...)`` is called. It combines multiple smaller
    helper workflows into one model-family-specific training-argument package.

    Depending on the selected ensemble family, this helper may collect:

    - simple model parameters from ``ENSEMBLE_PARAM_CONFIG``
    - optional extra pipeline steps such as PCA
    - one base estimator
    - multiple base estimators
    - voting weights
    - a stacking final estimator
    - an optional default GridSearchCV parameter grid

    Parameters
    ----------
    model_name : str
        Registered Apollo ensemble model name.

        Supported values include:

        - ``"VotingClassifier"``
        - ``"VotingRegressor"``
        - ``"BaggingClassifier"``
        - ``"BaggingRegressor"``
        - ``"AdaBoostClassifier"``
        - ``"AdaBoostRegressor"``
        - ``"GradientBoostingClassifier"``
        - ``"GradientBoostingRegressor"``
        - ``"StackingClassifier"``
        - ``"StackingRegressor"``

    task_type : str
        Task type associated with the selected model, typically
        ``"classifier"`` or ``"regressor"``.

    use_cv : bool
        Whether cross-validation is enabled for the current workflow.

        When ``True``, this helper may additionally ask whether the predefined
        default parameter grid should be used.

    use_preprocess : bool
        Whether outer preprocessing should be enabled in the downstream model
        training workflow.

    model_random_state : int
        Random-state value forwarded to downstream model-training methods when
        the target model supports it.

    feature_count : int or None, optional
        Current feature-column count used to filter PCA ``n_components``
        options during extra-step selection.

    Returns
    -------
    dict or None
        Dictionary of training keyword arguments tailored to the selected
        ensemble family.

        Depending on the workflow, the returned dictionary may contain keys
        such as:

        - ``estimators``
        - ``weights``
        - ``estimator``
        - ``final_estimator``
        - ``param_grid``
        - ``random_state``
        - ``use_preprocess``
        - ``extra_steps``

        Returns ``None`` if the user cancels at any required selection step or
        if the model name is unsupported.

    Notes
    -----
    This helper is the main dispatcher for ensemble-family-specific parameter
    collection.

    Internally it always performs these two common steps first:

    1. collect simple model parameters through
       ``collect_ensemble_simple_params(model_name)``
    2. collect optional extra pipeline steps through
       ``collect_extra_steps(feature_count=feature_count)``

    After that, it branches by ensemble family.

    Family-Specific Behavior
    ------------------------
    Voting
        - collect multiple base estimators
        - optionally collect custom voting weights
        - optionally attach a default voting parameter grid

    Bagging / AdaBoost
        - collect one base estimator
        - optionally attach a default parameter grid

    Gradient Boosting
        - does not collect base estimators
        - optionally attaches a default parameter grid
        - uses only simple kwargs, preprocessing settings, and extra steps

    Stacking
        - collect multiple base estimators
        - collect one final estimator
        - optionally attach a compatible default parameter grid

    Default Grid Compatibility
    --------------------------
    ``StackingClassifier``
        The predefined default grid is designed for a logistic final estimator.
        If another final estimator is selected, ``param_grid`` is set to
        ``None``.

    ``StackingRegressor``
        The predefined default grid is designed for ``linear`` or ``ridge``
        final estimators. If another final estimator is selected,
        ``param_grid`` is set to ``None``.

    Voting Weight Behavior
    ----------------------
    For voting workflows, the simple-parameter collector may return the flag
    ``use_default_weights``. This flag is removed from the final kwargs by
    ``pop(...)`` because it is only a menu-control flag, not a direct training
    parameter.

    If ``use_default_weights`` is ``True``, a uniform weight list is created:

    - ``[1] * len(estimators)``

    Otherwise, the user is asked to select a custom weight list.

    Workflow
    --------
    1. Collect simple ensemble parameters.
    2. Collect optional extra pipeline steps.
    3. Branch by model family.
    4. Collect any required estimators or final estimators.
    5. Optionally attach a default parameter grid.
    6. Return the assembled training keyword-argument dictionary.

    Examples
    --------
    A voting workflow may return something like::

        {
            "estimators": [("rf", ...), ("svc", ...), ("knn", ...)],
            "weights": [1, 2, 1],
            "param_grid": {"classifier__voting": ["hard", "soft"], ...},
            "random_state": 42,
            "use_preprocess": True,
            "extra_steps": [("pca", PCA(n_components=3))],
            "voting": "soft",
            "flatten_transform": True,
        }

    A bagging workflow may return something like::

        {
            "estimator": DecisionTreeClassifier(...),
            "param_grid": {"classifier__n_estimators": [10, 30, 50, 100], ...},
            "random_state": 42,
            "use_preprocess": True,
            "extra_steps": [],
            "n_estimators": 100,
            "max_samples": 1.0,
            "bootstrap": True,
        }
    """
    simple_kwargs = collect_ensemble_simple_params(model_name)
    if simple_kwargs is None:
        return None

    extra_steps = collect_extra_steps(feature_count=feature_count)
    if extra_steps is None:
        return None

    # ---------- Voting ----------
    if model_name in {"VotingClassifier", "VotingRegressor"}:
        estimators = collect_estimators(
            task_type=task_type,
            min_estimators=2,
            max_estimators=5,
        )
        if estimators is None:
            return None

        use_default_weights = simple_kwargs.pop("use_default_weights", True)
        if use_default_weights:
            weights = [1] * len(estimators)
        else:
            weights = _select_voting_weights(estimators)
            if weights is None:
                return None

        param_grid = maybe_get_default_param_grid(
            model_name=model_name,
            use_cv=use_cv,
        )

        return {
            "estimators": estimators,
            "weights": weights,
            "param_grid": param_grid,
            "random_state": model_random_state,
            "use_preprocess": use_preprocess,
            "extra_steps": extra_steps,
            **simple_kwargs,
        }

    # ---------- Bagging / AdaBoost ----------
    if model_name in {
        "BaggingClassifier",
        "BaggingRegressor",
        "AdaBoostClassifier",
        "AdaBoostRegressor",
    }:

        selected = select_single_estimator(task_type=task_type)
        if selected is None:
            return None

        _, estimator = selected

        param_grid = maybe_get_default_param_grid(
            model_name=model_name,
            use_cv=use_cv,
        )

        return {
            "estimator": estimator,
            "param_grid": param_grid,
            "random_state": model_random_state,
            "use_preprocess": use_preprocess,
            "extra_steps": extra_steps,
            **simple_kwargs,
        }

    # ---------- Gradient Boosting ----------
    if model_name in {
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
    }:
        param_grid = maybe_get_default_param_grid(
            model_name=model_name,
            use_cv=use_cv,
        )

        return {
            "param_grid": param_grid,
            "random_state": model_random_state,
            "use_preprocess": use_preprocess,
            "extra_steps": extra_steps,
            **simple_kwargs,
        }

    # ---------- Stacking ----------
    if model_name in {"StackingClassifier", "StackingRegressor"}:
        estimators = collect_estimators(
            task_type=task_type,
            min_estimators=2,
            max_estimators=3,
        )
        if estimators is None:
            return None

        final_selected = select_final_estimator(task_type=task_type)
        if final_selected is None:
            return None

        final_estimator_name, final_estimator = final_selected

        # ---------- Default param grid compatibility ----------
        if model_name == "StackingClassifier":
            if final_estimator_name != "logistic":
                print(
                    "🔔 Default stacking classifier grid is designed for logistic final_estimator."
                )
                param_grid = None
            else:
                param_grid = maybe_get_default_param_grid(
                    model_name=model_name,
                    use_cv=use_cv,
                    final_estimator_name=final_estimator_name,
                )

        elif model_name == "StackingRegressor":
            if final_estimator_name not in {"linear", "ridge"}:
                print(
                    "🔔 Default stacking regressor grid is designed for linear or ridge final_estimator."
                )
                param_grid = None
            else:
                param_grid = maybe_get_default_param_grid(
                    model_name=model_name,
                    use_cv=use_cv,
                    final_estimator_name=final_estimator_name,
                )

        return {
            "estimators": estimators,
            "final_estimator": final_estimator,
            "param_grid": param_grid,
            "random_state": model_random_state,
            "use_preprocess": use_preprocess,
            "extra_steps": extra_steps,
            **simple_kwargs,
        }

    print(f"⚠️ Unsupported ensemble model: {model_name} ‼️")
    return None


# =================================================
