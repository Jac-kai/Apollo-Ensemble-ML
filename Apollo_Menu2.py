# -------------------- Import Modules --------------------
import logging
import os

import pandas as pd

from Apollo.Apollo_ML_Engine import ApolloEngine
from Apollo.Apollo_Model_Menu_Helper import (
    collect_common_training_params,
    collect_ensemble_train_kwargs,
    select_from_options,
    select_model_name,
)
from Apollo.Menu_Config import COMMON_PARAM_CONFIG
from Apollo.Menu_Helper_Decorator import input_int, input_yesno, menu_wrapper

logger = logging.getLogger("Apollo")


# -------------------- Helper: categorical feature check --------------------
def _has_categorical_features(data) -> bool:
    """
    Check whether the given feature dataset contains categorical-like columns.

    This helper inspects the provided feature dataset and determines whether it
    includes at least one column with a categorical-like dtype, such as
    ``object``, ``category``, or ``bool``.

    Parameters
    ----------
    data : pandas.DataFrame or None
        Feature dataset to inspect.

    Returns
    -------
    bool
        ``True`` if at least one categorical-like column is found; otherwise
        ``False``.

    Notes
    -----
    If ``data`` is ``None``, this helper returns ``False``.
    """
    logger.info("Checking whether feature data contains categorical features")

    if data is None:
        logger.warning("_has_categorical_features received None data")
        return False

    categorical_cols = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    has_cat = len(categorical_cols) > 0
    logger.info(
        "Categorical feature check completed | has_categorical=%s | categorical_columns=%s",
        has_cat,
        list(categorical_cols),
    )
    return has_cat


# -------------------- Helper: get feature count --------------------
def _get_feature_count(apollo: ApolloEngine) -> int | None:
    """
    Return the number of currently selected feature columns.

    This helper tries to resolve the current feature count from the active
    Apollo feature workflow. It first checks ``feature_core.feature_columns``.
    If that is unavailable, it falls back to the column count of
    ``feature_core.X`` when possible.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    int or None
        Number of selected feature columns if available; otherwise ``None``.

    Notes
    -----
    This helper is mainly used to support downstream parameter filtering
    workflows such as PCA-related option checks.
    """
    logger.info("Getting current feature count from ApolloEngine")

    if apollo.feature_core is None:
        logger.warning("_get_feature_count failed: feature_core is None")
        return None

    # ---------- Count feature columns ----------
    feature_columns = getattr(apollo.feature_core, "feature_columns", None)
    if feature_columns is not None:
        count = len(feature_columns)
        logger.info("Feature count resolved from feature_columns: %s", count)
        return count

    # ---------- Count X variable ----------
    feature_data = getattr(apollo.feature_core, "X", None)
    if feature_data is not None and hasattr(feature_data, "columns"):
        count = len(feature_data.columns)
        logger.info("Feature count resolved from feature data columns: %s", count)
        return count

    logger.warning(
        "_get_feature_count failed: no feature_columns or X.columns available"
    )
    return None


# -------------------- Train classifier menu --------------------
@menu_wrapper("Train Classifier Ensemble")
def train_classifier_menu(apollo: ApolloEngine):
    """
    Train a classifier ensemble model through the Apollo workflow.

    This menu validates that feature preparation has been completed, then guides
    the user through classifier-ensemble training:

    1. select a classifier ensemble model name,
    2. collect common training parameters,
    3. resolve the current feature count,
    4. collect ensemble-family-specific training kwargs,
    5. optionally choose a categorical encoder when categorical features exist,
    6. dispatch training through ``ApolloEngine.train_model(...)``.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    - This menu requires ``feature_core`` plus prepared ``X`` and ``y`` data.
    - If categorical feature columns are present, the user is prompted to
      choose an encoder strategy.
    - If no categorical features are detected, the default categorical encoder
      value is kept unchanged.
    """
    logger.info("Entered menu: Train Classifier Ensemble")

    if apollo.feature_core is None:
        logger.warning("Train Classifier failed: feature_core is None")
        print("⚠️ Please load data first ‼️")
        return

    if (
        getattr(apollo.feature_core, "X", None) is None
        or getattr(apollo.feature_core, "y", None) is None
    ):
        logger.warning("Train Classifier failed: X or y not prepared")
        print("⚠️ Please complete feature/target selection first ‼️")
        return

    # ---------- Select classifier ensemble ----------
    model_name = select_model_name(apollo, task_type="classifier")
    if model_name is None:
        logger.info("Train Classifier cancelled at model selection")
        return

    logger.info("Classifier ensemble selected: %s", model_name)

    # ---------- Collect common params ----------
    common_params = collect_common_training_params(task_type="classifier")
    if common_params is None:
        logger.info("Train Classifier cancelled at common parameter selection")
        return

    # ---------- Get feature count for PCA filtering ----------
    feature_count = _get_feature_count(apollo)
    logger.info("Current feature_count for classifier workflow: %s", feature_count)

    # ---------- Collect ensemble-specific kwargs ----------
    train_kwargs = collect_ensemble_train_kwargs(
        model_name=model_name,
        task_type="classifier",
        use_cv=common_params["use_cv"],
        use_preprocess=common_params["use_preprocess"],
        model_random_state=common_params["model_random_state"],
        feature_count=feature_count,
    )
    if train_kwargs is None:
        logger.info("Train Classifier cancelled at ensemble parameter selection")
        return

    # ---------- Categorical encoder ----------
    cat_encoder = "ohe"  # Default encoding method
    feature_data = getattr(apollo.feature_core, "X", None)

    if _has_categorical_features(feature_data):
        logger.info("Categorical features detected; encoder selection required")
        config = COMMON_PARAM_CONFIG["cat_encoder"]  # Show the encoding options
        selected_encoder = select_from_options(
            label=config["label"],
            options=config["options"],
            default=config["default"],
        )

        if selected_encoder is None:
            logger.info("Train Classifier cancelled at categorical encoder selection")
            return

        cat_encoder = selected_encoder
        logger.info("Categorical encoder selected: %s", cat_encoder)
    else:
        logger.info(
            "No categorical features detected; using default cat_encoder=%s",
            cat_encoder,
        )

    logger.info(
        "Start classifier ensemble training | model=%s | common_params=%s | train_kwargs=%s | cat_encoder=%s",
        model_name,
        common_params,
        train_kwargs,
        cat_encoder,
    )

    # ---------- Dispatch training ----------
    result = apollo.train_model(
        model_name=model_name,
        test_size=common_params["test_size"],
        split_random_state=common_params["split_random_state"],
        use_cv=common_params["use_cv"],
        cv_folds=common_params["cv_folds"],
        scoring=common_params["scoring"],
        cat_encoder=cat_encoder,
        **train_kwargs,
    )

    if result is None:
        logger.warning("Classifier ensemble training failed: %s", model_name)
        print("⚠️ Classifier ensemble training failed ‼️")
        return

    logger.info("Classifier ensemble training completed successfully: %s", model_name)
    print(f"🍁 Classifier ensemble training completed: {model_name}")


# -------------------- Train regressor menu --------------------
@menu_wrapper("Train Regressor Ensemble")
def train_regressor_menu(apollo: ApolloEngine):
    """
    Train a regressor ensemble model through the Apollo workflow.

    This menu validates that feature preparation has been completed, then guides
    the user through regressor-ensemble training:

    1. select a regressor ensemble model name,
    2. collect common training parameters,
    3. resolve the current feature count,
    4. collect ensemble-family-specific training kwargs,
    5. optionally choose a categorical encoder when categorical features exist,
    6. dispatch training through ``ApolloEngine.train_model(...)``.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    - This menu requires ``feature_core`` plus prepared ``X`` and ``y`` data.
    - If categorical feature columns are present, the user is prompted to
      choose an encoder strategy.
    - If no categorical features are detected, the default categorical encoder
      value is kept unchanged.
    """
    logger.info("Entered menu: Train Regressor Ensemble")

    if apollo.feature_core is None:
        logger.warning("Train Regressor failed: feature_core is None")
        print("⚠️ Please load data first ‼️")
        return

    if (
        getattr(apollo.feature_core, "X", None) is None
        or getattr(apollo.feature_core, "y", None) is None
    ):
        logger.warning("Train Regressor failed: X or y not prepared")
        print("⚠️ Please complete feature/target selection first ‼️")
        return

    # ---------- Select regressor ensemble ----------
    model_name = select_model_name(apollo, task_type="regressor")
    if model_name is None:
        logger.info("Train Regressor cancelled at model selection")
        return

    logger.info("Regressor ensemble selected: %s", model_name)

    # ---------- Collect common params ----------
    common_params = collect_common_training_params(task_type="regressor")
    if common_params is None:
        logger.info("Train Regressor cancelled at common parameter selection")
        return

    # ---------- Get feature count for PCA filtering ----------
    feature_count = _get_feature_count(apollo)
    logger.info("Current feature_count for regressor workflow: %s", feature_count)

    # ---------- Collect ensemble-specific kwargs ----------
    train_kwargs = collect_ensemble_train_kwargs(
        model_name=model_name,
        task_type="regressor",
        use_cv=common_params["use_cv"],
        use_preprocess=common_params["use_preprocess"],
        model_random_state=common_params["model_random_state"],
        feature_count=feature_count,
    )
    if train_kwargs is None:
        logger.info("Train Regressor cancelled at ensemble parameter selection")
        return

    # ---------- Categorical encoder ----------
    cat_encoder = "ohe"  # Default encoding method
    feature_data = getattr(apollo.feature_core, "X", None)

    if _has_categorical_features(feature_data):
        logger.info("Categorical features detected; encoder selection required")
        config = COMMON_PARAM_CONFIG["cat_encoder"]  # Show the encoding options
        selected_encoder = select_from_options(
            label=config["label"],
            options=config["options"],
            default=config["default"],
        )

        if selected_encoder is None:
            logger.info("Train Regressor cancelled at categorical encoder selection")
            return

        cat_encoder = selected_encoder
        logger.info("Categorical encoder selected: %s", cat_encoder)
    else:
        logger.info(
            "No categorical features detected; using default cat_encoder=%s",
            cat_encoder,
        )
    logger.info(
        "Start regressor ensemble training | model=%s | common_params=%s | train_kwargs=%s | cat_encoder=%s",
        model_name,
        common_params,
        train_kwargs,
        cat_encoder,
    )

    # ---------- Dispatch training ----------
    result = apollo.train_model(
        model_name=model_name,
        test_size=common_params["test_size"],
        split_random_state=common_params["split_random_state"],
        use_cv=common_params["use_cv"],
        cv_folds=common_params["cv_folds"],
        scoring=common_params["scoring"],
        cat_encoder=cat_encoder,
        **train_kwargs,
    )

    if result is None:
        logger.warning("Regressor ensemble training failed: %s", model_name)
        print("⚠️ Regressor ensemble training failed ‼️")
        return

    logger.info("Regressor ensemble training completed successfully: %s", model_name)
    print(f"🍁 Regressor ensemble training completed: {model_name}")


# -------------------- Current model summary menu --------------------
@menu_wrapper("Current Model Summary")
def current_model_summary_menu(apollo: ApolloEngine):
    """
    Display the summary of the current active Apollo model.

    This menu delegates summary display to
    ``ApolloEngine.show_current_model_summary()`` and reports the outcome.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    If no current model summary is available, the menu exits after the engine
    reports the failure condition.
    """
    logger.info("Entered menu: Current Model Summary")

    result = apollo.show_current_model_summary()
    if result is None:
        logger.warning("Current model summary display failed")
        return

    logger.info("Current model summary displayed")


# -------------------- Save current model menu --------------------
@menu_wrapper("Save Current Model")
def save_current_model_menu(apollo: ApolloEngine):
    """
    Save the current trained model through ApolloEngine.

    This menu validates that a current model exists, then delegates model
    persistence to ``ApolloEngine.save_current_model()``.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    The save destination is determined internally by the Apollo engine and the
    current model type.
    """
    logger.info("Entered menu: Save Current Model")

    if apollo.current_model is None:
        logger.warning("Save Current Model failed: no current model")
        print("⚠️ No trained model to save ‼️")
        return

    logger.info("Start saving current model using model default MODEL_DIR")
    result = apollo.save_current_model()

    if result is None:
        logger.warning("Failed to save current model")
        print("⚠️ Failed to save current model ‼️")
        return

    logger.info("Current model saved successfully: %s", result)
    print(f"🔥 Model saved successfully: {result}")


# -------------------- Load trained model menu --------------------
@menu_wrapper("Load Trained Model")
def load_trained_model_menu(apollo: ApolloEngine):
    """
    Load a previously saved trained model into ApolloEngine.

    This menu shows the registered Apollo model types, prompts the user to
    choose a model family, lists saved model files for that family, and then
    loads the selected saved model into the current Apollo runtime.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    - If no registered models are available, the menu exits immediately.
    - If no saved files exist for the selected model type, the menu exits after
      showing a warning.
    - Invalid model or file selections are rejected without changing the
      current Apollo model state.
    """
    logger.info("Entered menu: Load Trained Model")

    # ---------- List available models ----------
    all_models = apollo.get_available_models()

    if not all_models:
        logger.warning("Load Trained Model failed: no model registry found")
        print("⚠️ No model registry found ‼️")
        return

    # ---------- Show available model types ----------
    model_map = {i: name for i, name in enumerate(all_models, 1)}

    print("\n----- 🔥 Registered Apollo Models 🔥 -----")
    for i, name in model_map.items():
        print(f"🗄️ {i}. {name}")
    print("-" * 50)

    # ---------- Select available model type ----------
    selected_num = input_int("🕯️ Select trained model type", default=-1)
    if selected_num is None:
        logger.info("Load Trained Model cancelled at model type selection")
        return

    if selected_num not in model_map:
        logger.warning(
            "Load Trained Model failed: model selection out of range | selected=%s",
            selected_num,
        )
        print("⚠️ Model selection is out of range ‼️")
        return

    model_name = model_map[selected_num]  # Get model name
    logger.info("Load model type selected: %s", model_name)

    # ---------- List trained model files ----------
    saved_model_files = apollo._get_saved_model_files(model_name)
    if not saved_model_files:
        logger.warning("No saved model files found for model: %s", model_name)
        print("⚠️ No saved model files found for this model type ‼️")
        return

    file_map = {
        i: path for i, path in enumerate(saved_model_files, 1)
    }  # Turn into dictionary format

    print("\n----- 🔥 Saved Model Files 🔥 -----")
    for i, path in file_map.items():
        print(f"📦 {i}. {os.path.basename(path)}")
    print("-" * 50)

    # ---------- Select trained model file ----------
    selected_file_num = input_int("🕯️ Select saved model file", default=-1)
    if selected_file_num is None:
        logger.info("Load Trained Model cancelled at saved-file selection")
        return

    if selected_file_num not in file_map:
        logger.warning(
            "Load Trained Model failed: saved-file selection out of range | selected=%s",
            selected_file_num,
        )
        print("⚠️ Saved model selection is out of range ‼️")
        return

    filepath = file_map[selected_file_num]
    logger.info("Selected saved model file: %s", filepath)

    logger.info(
        "Start loading trained model | model=%s | path=%s", model_name, filepath
    )

    # ---------- Load trained model ----------
    result = apollo.load_trained_model(
        model_name=model_name,
        filepath=filepath,
    )

    if result is None:
        logger.warning(
            "Failed to load trained model | model=%s | path=%s", model_name, filepath
        )
        print("⚠️ Failed to load trained model ‼️")
        return

    logger.info("Trained model loaded successfully: %s", model_name)
    print(f"🔥 Trained model loaded successfully: {model_name}")


# -------------------- Predict with current model --------------------
@menu_wrapper("Predict with Current Model")
def predict_with_current_model_menu(apollo: ApolloEngine):
    """
    Predict target values for the currently loaded dataset using the active model.

    This menu validates that a current model, feature workflow, and source
    dataset are available, optionally shows the required feature columns for the
    active model, asks the user for confirmation, and then dispatches the
    prediction workflow through ``ApolloEngine.predict_with_current_model(...)``.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    - Prediction uses the current source dataset stored in the Apollo engine.
    - The source dataset must be a ``pandas.DataFrame``.
    - A short preview of prediction results is displayed after successful
      prediction.
    """
    logger.info("Entered menu: Predict with Current Model")

    if apollo.current_model is None:
        logger.warning("Predict failed: no current model")
        print("⚠️ No current model available for prediction ‼️")
        return

    if apollo.feature_core is None:
        logger.warning("Predict failed: FeatureCore not built")
        print("⚠️ FeatureCore has not been built ‼️")
        return

    new_data = apollo.source_data
    if new_data is None:
        logger.warning("Predict failed: no source data available")
        print("⚠️ No source data available for prediction ‼️")
        return

    if not isinstance(new_data, pd.DataFrame):
        logger.warning("Predict failed: source data is not a DataFrame")
        print("⚠️ Prediction data must be a pandas DataFrame ‼️")
        return

    # ---------- List trained feature names ----------
    feature_names = getattr(apollo.current_model, "feature_names", None)
    if feature_names:
        logger.info(
            "Displaying required feature columns for prediction: %s", feature_names
        )
        print("----- 🔥 Required Feature Columns 🔥 -----")
        for i, col in enumerate(feature_names, 1):
            print(f"🍒 {i}. {col}")
        print("-" * 50)

    confirm = input_yesno("🕯️ Continue prediction with current dataset")
    if confirm is None:
        logger.info("Predict cancelled at confirmation step by back/cancel")
        return

    if confirm is False:
        logger.info("Predict declined by user at confirmation step")
        return

    logger.info(
        "Start prediction with current model | rows=%s | cols=%s",
        len(new_data),
        len(new_data.columns),
    )
    predictions = apollo.predict_with_current_model(new_data)

    if predictions is None:
        logger.warning("Prediction failed")
        print("⚠️ Prediction failed ‼️")
        return

    logger.info("Prediction completed successfully | output_size=%s", len(predictions))

    print("----- 🔥 Prediction Result Preview 🔥 -----")
    preview_count = min(10, len(predictions))
    for i in range(preview_count):
        print(f"{i+1}. {predictions[i]}")
    print("-" * 50)
    print(
        f"🔥 Prediction completed successfully. Total predictions: {len(predictions)}"
    )


# -------------------- Apollo Menu2 dispatcher --------------------
@menu_wrapper("Apollo Model Management Menu")
def model_management_menu(apollo: ApolloEngine):
    """
    Display the top-level model management menu for Apollo workflows.

    This dispatcher provides access to the main model-layer actions in Apollo,
    including classifier training, regressor training, current-model summary,
    model saving, model loading, and prediction with the active model.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance.

    Returns
    -------
    None

    Notes
    -----
    The same Apollo engine instance is reused across all menu actions so that
    current model state, training results, and loaded resources persist within
    the active session.
    """
    logger.info("Entered menu: Apollo Model Management Menu")

    menu = [
        (1, "👁️ Training Classification Ensemble", train_classifier_menu),
        (2, "🪢 Training Regression Ensemble", train_regressor_menu),
        (3, "🔥 Current Model Summary", current_model_summary_menu),
        (4, "💾 Save Current Trained Model", save_current_model_menu),
        (5, "📥 Load Trained Model", load_trained_model_menu),
        (6, "🔮 Predict with Current Model", predict_with_current_model_menu),
        (0, "↩️ Back", None),
    ]
    menu_width = 54

    while True:
        print("🏮  Apollo Model Management Menu 🏮 ".center(menu_width, "━"))
        for opt, label, _ in menu:
            print(f"{opt}. {label}")
        print("━" * menu_width)

        choice = input_int("🕯️ Select Model Services", default=-1)
        if choice is None:
            logger.info("Exited Apollo Model Management Menu by cancel/back")
            return

        matched = False
        for opt, label, func in menu:
            if choice == opt:
                logger.info(
                    "Apollo Model Management Menu selection: %s - %s",
                    choice,
                    label,
                )
                matched = True
                if func is None:
                    logger.info("Exited Apollo Model Management Menu")
                    return
                func(apollo)
                break

        if not matched:
            logger.warning(
                "Invalid selection in Apollo Model Management Menu: %s", choice
            )
            print("⚠️ Invalid selection ‼️")


# =================================================
