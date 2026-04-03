# -------------------- Import Modules --------------------
import glob
import logging
import os
from pprint import pprint

from Apollo.Backbone.FeatureCore import FeatureCore
from Apollo.Ensemble_Model.AdaBoosting_Model import (
    AdaBoostClassifier_Model,
    AdaBoostRegressor_Model,
)
from Apollo.Ensemble_Model.Bagging_Model import (
    BaggingClassifier_Model,
    BaggingRegressor_Model,
)
from Apollo.Ensemble_Model.GradientBoosting_Model import (
    GradientBoostingClassifier_Model,
    GradientBoostingRegressor_Model,
)
from Apollo.Ensemble_Model.Stacking_Model import (
    StackingClassifier_Model,
    StackingRegressor_Model,
)
from Apollo.Ensemble_Model.Voting_Model import (
    VotingClassifier_Model,
    VotingRegressor_Model,
)
from Cornus.Data_Hunter.HuntingDataCore import HuntingDataCore
from Cornus.MetaUnits.VisionCore import VisionCore

logger = logging.getLogger("Apollo")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
# -------------------- Model registry --------------------
APOLLO_MODEL_REGISTRY = {
    "VotingClassifier": {
        "class": VotingClassifier_Model,
        "task_type": "classifier",
        "family": "voting",
        "default_scoring": "f1_weighted",
    },
    "VotingRegressor": {
        "class": VotingRegressor_Model,
        "task_type": "regressor",
        "family": "voting",
        "default_scoring": "r2",
    },
    "BaggingClassifier": {
        "class": BaggingClassifier_Model,
        "task_type": "classifier",
        "family": "bagging",
        "default_scoring": "f1_weighted",
    },
    "BaggingRegressor": {
        "class": BaggingRegressor_Model,
        "task_type": "regressor",
        "family": "bagging",
        "default_scoring": "r2",
    },
    "AdaBoostClassifier": {
        "class": AdaBoostClassifier_Model,
        "task_type": "classifier",
        "family": "adaboost",
        "default_scoring": "f1_weighted",
    },
    "AdaBoostRegressor": {
        "class": AdaBoostRegressor_Model,
        "task_type": "regressor",
        "family": "adaboost",
        "default_scoring": "r2",
    },
    "GradientBoostingClassifier": {
        "class": GradientBoostingClassifier_Model,
        "task_type": "classifier",
        "family": "gradient_boosting",
        "default_scoring": "f1_weighted",
    },
    "GradientBoostingRegressor": {
        "class": GradientBoostingRegressor_Model,
        "task_type": "regressor",
        "family": "gradient_boosting",
        "default_scoring": "r2",
    },
    "StackingClassifier": {
        "class": StackingClassifier_Model,
        "task_type": "classifier",
        "family": "stacking",
        "default_scoring": "f1_weighted",
    },
    "StackingRegressor": {
        "class": StackingRegressor_Model,
        "task_type": "regressor",
        "family": "stacking",
        "default_scoring": "r2",
    },
}


# -------------------- Apollo Engine --------------------
class ApolloEngine:
    """
    Central workflow engine for the Apollo ensemble-learning system.

    This engine coordinates dataset loading, feature/target selection, core
    object construction, model creation, model training, model persistence,
    and prediction workflows for Apollo ensemble models.

    The engine acts as a high-level controller that connects:
    - ``HuntingDataCore`` for dataset discovery and loading,
    - ``VisionCore`` for data inspection support,
    - ``FeatureCore`` for feature/target configuration and X/Y building,
    - registered Apollo ensemble model classes for training and inference.

    Attributes
    ----------
    hunter_core : HuntingDataCore
        Core object responsible for locating and loading source datasets.
    vision_core : VisionCore or None
        Data-inspection helper core built after a dataset is loaded.
    feature_core : FeatureCore or None
        Feature/target configuration core built from the current source data.
    current_model : object or None
        Current instantiated or loaded Apollo model object.
    current_model_name : str or None
        Registry name of the current model.
    current_model_result : Any or None
        Latest training result returned by the current model workflow.

    Notes
    -----
    The engine keeps runtime state across operations so that loaded data,
    configured features, built models, and training results can be reused
    throughout the current session.
    """

    # -------------------- Initialization --------------------
    def __init__(self):
        """
        Initialize the Apollo engine runtime state.

        This constructor creates the base data-loading core and initializes
        workflow state for auxiliary cores, active model objects, and current
        training results.

        Returns
        -------
        None

        Notes
        -----
        After initialization, no dataset, feature configuration, or model is
        available yet. These must be built through later workflow steps.
        """
        # ---------- Import cores ----------
        self.hunter_core = HuntingDataCore()
        self.vision_core = None
        self.feature_core = None

        # ---------- Record current model informations ----------
        self.current_model = None
        self.current_model_name = None
        self.current_model_result = None
        logger.info("ApolloEngine initialized")

    # -------------------- Source data property --------------------
    @property
    def source_data(self):
        """
        Return the currently loaded source dataset.

        This property exposes the target dataset stored in ``hunter_core`` so the
        Apollo engine can build dependent cores from the active data source.

        Returns
        -------
        Any
            The currently loaded dataset stored in ``self.hunter_core.target_data``.
            The exact type depends on the loader implementation.

        Notes
        -----
        If no dataset has been loaded yet, this property typically returns ``None``.
        """
        return self.hunter_core.target_data

    # -------------------- Build cores --------------------
    def build_cores(self):
        """
        Build Apollo helper cores from the current source dataset.

        This method validates that source data is available, then rebuilds the
        runtime helper cores used for data inspection and feature/target handling.

        Specifically, it creates:
        - ``VisionCore`` from the current ``HuntingDataCore`` instance,
        - ``FeatureCore`` from the current loaded source dataset.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no source dataset is currently available.

        Notes
        -----
        This method should be called after a dataset is loaded or whenever the
        source dataset changes and dependent cores need to be refreshed.
        """
        logger.info("Building Apollo cores")

        if self.source_data is None:
            logger.warning("build_cores failed: source_data is None")
            raise ValueError("⚠️ No source data available. Please load data first ‼️")

        self.vision_core = VisionCore(self.hunter_core)
        self.feature_core = FeatureCore(self.source_data)

        logger.info("Apollo cores built successfully")

    # -------------------- Refresh cores --------------------
    def _refresh_cores(self):
        """
        Refresh Apollo helper cores using the current source dataset.

        This helper method rebuilds runtime cores by calling ``build_cores()``.
        It is intended for internal workflow use when the engine needs to
        reconstruct dependent helper objects after source data changes.

        Returns
        -------
        None
        """
        logger.info("Refreshing Apollo cores")
        self.build_cores()  # Rebuild import cores

    # -------------------- Load dataset --------------------
    def ml_dataset_search(
        self,
        selected_folder_num: int,
        selected_file_num: int,
        opener_param_dict: dict | None = None,
    ):
        """
        Search for and load a machine-learning dataset into the Apollo workflow.

        This method delegates folder discovery, file selection, and file opening to
        ``HuntingDataCore``. If the dataset is loaded successfully, Apollo helper
        cores are built immediately so the data can be used in later feature and
        model workflows.

        Parameters
        ----------
        selected_folder_num : int
            Index of the selected folder from the available working-place folders.
        selected_file_num : int
            Index of the selected file within the chosen folder.
        opener_param_dict : dict or None, optional
            Optional keyword arguments passed to the dataset opener. If ``None``,
            an empty dictionary is used.

        Returns
        -------
        Any
            The loaded dataset returned by the underlying opener. The exact type
            depends on the loader implementation. Returns ``None`` if loading fails.

        Notes
        -----
        When loading succeeds, this method automatically rebuilds ``VisionCore``
        and ``FeatureCore`` so the Apollo engine is ready for downstream steps.
        """
        logger.info(
            "Starting dataset search | folder_num=%s | file_num=%s",
            selected_folder_num,
            selected_file_num,
        )

        # ---------- Select folder and select file ----------
        self.hunter_core.working_place_searcher()
        self.hunter_core.files_searcher_from_folders(
            selected_folder_num=selected_folder_num,
            selected_file_num=selected_file_num,
        )

        # ---------- Open selected file ----------
        opener_param_dict = opener_param_dict or {}
        loaded_data = self.hunter_core.opener(
            **opener_param_dict
        )  # Target data uploaded

        if loaded_data is not None:
            self.build_cores()
            logger.info("Dataset loaded successfully")
        else:
            logger.warning(
                "Dataset loading failed | folder_num=%s | file_num=%s",
                selected_folder_num,
                selected_file_num,
            )

        return loaded_data

    # -------------------- Set target column --------------------
    def set_target_column(self, target_column: str | list[str]):
        """
        Set the target column or columns in the current feature workflow.

        This method forwards the target-column selection to ``FeatureCore`` so the
        current dataset can be configured for supervised learning workflows.

        Parameters
        ----------
        target_column : str or list[str]
            Target column name for single-output tasks, or a list of target column
            names for multi-output tasks.

        Returns
        -------
        Any or None
            Result returned by ``FeatureCore.set_target_column(...)``.
            Returns ``None`` if ``feature_core`` has not been built yet.

        Notes
        -----
        A dataset must be loaded and helper cores must already exist before this
        method can be used successfully.
        """
        logger.info("Setting target column(s): %s", target_column)

        if self.feature_core is None:
            logger.warning("set_target_column failed: feature_core is None")
            print("⚠️ FeatureCore has not been built ‼️")
            return None

        # ---------- Get taeget columns from FeatureCore ----------
        result = self.feature_core.set_target_column(target_column)
        logger.info("Target column(s) set successfully: %s", target_column)
        return result

    # -------------------- Set feature columns --------------------
    def set_feature_columns(self, feature_columns: list[str]):
        """
        Set the feature columns in the current feature workflow.

        This method forwards selected feature-column names to ``FeatureCore`` so
        the current dataset can be prepared for model-building workflows.

        Parameters
        ----------
        feature_columns : list[str]
            List of column names to use as model input features.

        Returns
        -------
        Any or None
            Result returned by ``FeatureCore.set_feature_columns(...)``.
            Returns ``None`` if ``feature_core`` has not been built yet.

        Notes
        -----
        This method only configures the selected feature columns. Actual X/Y
        dataset building is performed later through ``build_xy_data()``.
        """
        logger.info("Setting feature columns: %s", feature_columns)

        if self.feature_core is None:
            logger.warning("set_feature_columns failed: feature_core is None")
            print("⚠️ FeatureCore has not been built ‼️")
            return None

        result = self.feature_core.set_feature_columns(feature_columns)
        logger.info("Feature columns set successfully: %s", feature_columns)
        return result

    # -------------------- Build X / Y --------------------
    def build_xy_data(self):
        """
        Build X and Y datasets from the current feature configuration.

        This method validates that ``FeatureCore`` exists and supports the
        ``build_xy_data()`` workflow, then delegates the actual dataset split
        into feature matrix and target data.

        Returns
        -------
        tuple[Any, Any] or None
            A two-item tuple ``(cleaned_X_data, cleaned_Y_data)`` returned by
            ``FeatureCore.build_xy_data()`` when successful.
            Returns ``None`` if the feature workflow is unavailable.

        Notes
        -----
        This method is typically used before model instantiation so the selected
        feature and target configuration can be passed into the model layer.
        """
        logger.info("Building X/Y data from FeatureCore")

        if self.feature_core is None:
            logger.warning("build_xy_data failed: feature_core is None")
            print("⚠️ FeatureCore has not been built ‼️")
            return None

        if not hasattr(self.feature_core, "build_xy_data"):
            logger.warning(
                "build_xy_data failed: FeatureCore has no build_xy_data method"
            )
            print("⚠️ FeatureCore does not support build_xy_data() ‼️")
            return None

        # ---------- Get X/Y variables ----------
        xy_data = (
            self.feature_core.build_xy_data()
        )  # Input selections to build_xy_data from FeatureCore
        logger.info("X/Y data built successfully")
        return xy_data

    # -------------------- Get available models --------------------
    def get_available_models(self, task_type: str | None = None) -> list[str]:
        """
        Return available Apollo model names from the registry.

        Parameters
        ----------
        task_type : str or None, default=None
            Optional task-type filter. Typical values are ``"classifier"`` and
            ``"regressor"``. If ``None``, all registered Apollo model names are
            returned.

        Returns
        -------
        list[str]
            Available model names, optionally filtered by task type.

        Notes
        -----
        Model availability is determined from ``APOLLO_MODEL_REGISTRY``.
        """
        logger.info("Getting available models | task_type=%s", task_type)

        # ---------- List avaliable models ----------
        if task_type is None:
            return list(
                APOLLO_MODEL_REGISTRY.keys()
            )  # Show all model name without task type input

        return [
            model_name
            for model_name, meta in APOLLO_MODEL_REGISTRY.items()  # Get model by task type
            if meta["task_type"] == task_type
        ]

    # -------------------- Build model instance --------------------
    def build_model(self, model_name: str):
        """
        Build a model instance from the Apollo registry.

        This method validates the requested model name, builds the current X/Y data
        from ``FeatureCore``, instantiates the corresponding model class, and stores
        the created object as the current active Apollo model.

        Parameters
        ----------
        model_name : str
            Registry name of the model to build.

        Returns
        -------
        object or None
            Instantiated Apollo model object when successful.
            Returns ``None`` if the model name is unsupported or X/Y data could not
            be built.

        Notes
        -----
        Building a model resets ``current_model_result`` to ``None`` because a new
        model instance has not yet been trained.
        """
        logger.info("Building model instance | model_name=%s", model_name)

        # ---------- Get model's metadata ----------
        model_meta = APOLLO_MODEL_REGISTRY.get(model_name)
        if model_meta is None:
            logger.warning("build_model failed: unsupported model_name=%s", model_name)
            print(f"⚠️ Unsupported model type: {model_name} ‼️")
            return None

        # ---------- Get X/Y vaiables ----------
        xy_data = self.build_xy_data()
        if xy_data is None:
            logger.warning(
                "build_model failed: X/Y data is None | model_name=%s", model_name
            )
            return None

        # ---------- Model initialization ----------
        cleaned_X_data, cleaned_Y_data = xy_data  # Return self.X. self.Y
        model_cls = model_meta["class"]  # Setup model object

        # ---------- Input X/Y to model onject ----------
        model = model_cls(
            cleaned_X_data=cleaned_X_data,
            cleaned_Y_data=cleaned_Y_data,
        )

        # ---------- Record current model info. ----------
        self.current_model = model
        self.current_model_name = model_name
        self.current_model_result = None

        logger.info("Built model instance: %s", model_name)
        return model

    # -------------------- Train model --------------------
    def train_model(
        self,
        model_name: str,
        test_size: float = 0.2,
        split_random_state: int = 42,
        **train_kwargs,
    ):
        """
        Build and train an Apollo model using the current feature configuration.

        This method first builds the requested model instance, then runs the model's
        train/test split workflow, and finally delegates the actual fitting process
        to the model-layer ``train(...)`` method.

        Parameters
        ----------
        model_name : str
            Registry name of the model to train.
        test_size : float, default=0.2
            Proportion of the dataset used for the test split.
        split_random_state : int, default=42
            Random seed used for train/test splitting.
        **train_kwargs
            Additional keyword arguments forwarded to the model-layer
            ``train(...)`` method.

        Returns
        -------
        Any or None
            Training result returned by ``model.train(...)`` when successful.
            Returns ``None`` if model construction fails or the current model does
            not support the expected split workflow.

        Raises
        ------
        Exception
            Re-raises any exception raised by the underlying model-layer
            ``train(...)`` call.

        Notes
        -----
        On success, this method updates:
        - ``self.current_model``
        - ``self.current_model_name``
        - ``self.current_model_result``

        This method assumes the model class provides
        ``train_test_split_engine(...)`` before training.
        """
        logger.info(
            "Starting model training | model_name=%s | test_size=%s | split_random_state=%s | train_kwargs=%s",
            model_name,
            test_size,
            split_random_state,
            train_kwargs,
        )

        model = self.build_model(model_name)
        if model is None:
            logger.warning(
                "train_model failed: build_model returned None | model_name=%s",
                model_name,
            )
            return None

        # ---------- Run split before model-layer train ----------
        if not hasattr(model, "train_test_split_engine"):
            logger.warning(
                "train_model failed: model has no train_test_split_engine | model_name=%s",
                model_name,
            )
            print("⚠️ Current model does not support train/test split workflow ‼️")
            return None

        model.train_test_split_engine(
            test_size=test_size,
            random_state=split_random_state,
        )

        logger.info("Training model: %s", model_name)

        # ---------- Input model training parameters ----------
        try:
            result = model.train(**train_kwargs)
        except Exception:
            logger.exception(
                "Training failed with exception | model_name=%s", model_name
            )
            raise

        # ---------- Record current model info. ----------
        self.current_model = model
        self.current_model_name = model_name
        self.current_model_result = result

        logger.info("Training completed: %s", model_name)
        return result

    # -------------------- Show current model summary --------------------
    def show_current_model_summary(self):
        """
        Display the current training result summary.

        This method prints the content stored in ``self.current_model_result`` using
        ``pprint`` so the latest training result can be reviewed in a readable form.

        Returns
        -------
        Any or None
            The current training result if available.
            Returns ``None`` if no current model or no training result exists yet.

        Notes
        -----
        This method does not compute a new summary. It only displays the result
        already stored from the latest successful training workflow.
        """
        logger.info("Showing current model summary")

        if self.current_model is None:
            logger.warning("show_current_model_summary failed: current_model is None")
            print("⚠️ No current model available ‼️")
            return None

        if self.current_model_result is None:
            logger.warning(
                "show_current_model_summary failed: current_model_result is None"
            )
            print("⚠️ No training result available yet ‼️")
            return None

        # ---------- Show current model summary after training ----------
        pprint(self.current_model_result)
        logger.info("Current model summary displayed successfully")
        return self.current_model_result

    # -------------------- Save current model --------------------
    def save_current_model(self):
        """
        Save the current Apollo model to disk.

        This method validates that a current model exists and supports the
        ``save_model_joblib(...)`` workflow, resolves the target save folder from
        the current model name, and delegates model persistence to the model layer.

        Returns
        -------
        str or None
            Saved file path returned by ``save_model_joblib(...)`` when successful.
            Returns ``None`` if no model is available, saving is unsupported, or no
            valid save folder can be resolved.

        Notes
        -----
        Save-folder resolution is handled by ``_get_model_save_folder(...)`` based
        on the current model name.
        """
        logger.info("Saving current model | model_name=%s", self.current_model_name)

        if self.current_model is None:
            logger.warning("save_current_model failed: current_model is None")
            print("⚠️ No current model available to save ‼️")
            return None

        if not hasattr(self.current_model, "save_model_joblib"):
            logger.warning(
                "save_current_model failed: model does not support save_model_joblib | model_name=%s",
                self.current_model_name,
            )
            print("⚠️ Current model does not support model saving ‼️")
            return None

        # ---------- Save trained model to its folder ----------
        folder_path = self._get_model_save_folder(self.current_model_name)
        if folder_path is None:
            logger.warning(
                "save_current_model failed: save folder not found | model_name=%s",
                self.current_model_name,
            )
            return None

        # ---------- Saved path ----------
        saved_path = self.current_model.save_model_joblib(folder_path=folder_path)
        logger.info(
            "Model saved successfully | model_name=%s | path=%s",
            self.current_model_name,
            saved_path,
        )
        return saved_path

    # -------------------- Helper: get model save folder --------------------
    def _get_model_save_folder(self, model_name: str) -> str | None:
        """
        Resolve the default save folder for a given Apollo model name.

        Parameters
        ----------
        model_name : str
            Registry name of the Apollo model.

        Returns
        -------
        str or None
            Absolute folder path used for saving trained models when the model name
            is recognized. Returns ``None`` for unknown model names.

        Notes
        -----
        The mapping is hard-coded from model names to subfolder names under the
        Apollo engine report directory.
        """
        # ---------- Folders for trained model ----------
        folder_map = {
            "VotingClassifier": "VotingCla_Trained_Model",
            "VotingRegressor": "VotingReg_Trained_Model",
            "BaggingClassifier": "BaggingCla_Trained_Model",
            "BaggingRegressor": "BaggingReg_Trained_Model",
            "AdaBoostClassifier": "AdaBoostCla_Trained_Model",
            "AdaBoostRegressor": "AdaBoostReg_Trained_Model",
            "GradientBoostingClassifier": "GradientBoostingCla_Trained_Model",
            "GradientBoostingRegressor": "GradientBoostingReg_Trained_Model",
            "StackingClassifier": "StackingCla_Trained_Model",
            "StackingRegressor": "StackingReg_Trained_Model",
        }

        # ---------- Select folder to save trained model ----------
        folder_name = folder_map.get(model_name)
        if folder_name is None:
            logger.warning(
                "_get_model_save_folder failed: unknown model_name=%s", model_name
            )
            return None

        return os.path.join(
            project_root, "ES_ML_Report", folder_name
        )  # Return absolute path

    # -------------------- Helper: get saved model files --------------------
    def _get_saved_model_files(self, model_name: str) -> list[str]:
        """
        Return saved model files for a specific Apollo model type.

        This helper resolves the model's save folder, searches for ``.joblib``
        files inside that folder, and returns the file list sorted by modified time
        in descending order.

        Parameters
        ----------
        model_name : str
            Registry name of the Apollo model.

        Returns
        -------
        list[str]
            List of matching saved model file paths. Returns an empty list if the
            model folder cannot be resolved or does not exist.

        Notes
        -----
        The newest saved files appear first in the returned list.
        """
        logger.info("Getting saved model files | model_name=%s", model_name)

        # ---------- Get saved folder path ----------
        folder_path = self._get_model_save_folder(model_name)
        if folder_path is None:
            logger.warning(
                "_get_saved_model_files failed: folder_path is None | model_name=%s",
                model_name,
            )
            return []

        if not os.path.isdir(folder_path):
            logger.warning(
                "_get_saved_model_files failed: folder does not exist | model_name=%s | folder_path=%s",
                model_name,
                folder_path,
            )
            return []

        # ---------- Saved as joblib format ----------
        file_pattern = os.path.join(folder_path, "*.joblib")
        model_files = glob.glob(file_pattern)
        model_files.sort(key=os.path.getmtime, reverse=True)  # Sorting files

        logger.info(
            "Saved model files retrieved | model_name=%s | file_count=%s",
            model_name,
            len(model_files),
        )
        return model_files

    # -------------------- Load trained model --------------------
    def load_trained_model(self, model_name: str, filepath: str):
        """
        Load a trained Apollo model from disk.

        This method validates the requested model name, checks whether the model
        class supports ``load_model_joblib(...)``, loads the saved model object, and
        stores it as the current active Apollo model.

        Parameters
        ----------
        model_name : str
            Registry name of the model type to load.
        filepath : str
            Path to the saved model file.

        Returns
        -------
        object or None
            Loaded model object when successful.
            Returns ``None`` if the model type is unsupported, loading is
            unsupported by the class, or the file could not be loaded.

        Notes
        -----
        After loading, ``current_model_result`` is reset to ``None`` because a
        loaded model may not carry the same runtime training summary format used
        during the current session.
        """
        logger.info(
            "Loading trained model | model_name=%s | filepath=%s", model_name, filepath
        )

        # ---------- Select trained model ----------
        model_meta = APOLLO_MODEL_REGISTRY.get(model_name)
        if model_meta is None:
            logger.warning(
                "load_trained_model failed: unsupported model_name=%s", model_name
            )
            print(f"⚠️ Unsupported model type: {model_name} ‼️")
            return None

        model_cls = model_meta["class"]  # Setup model object

        if not hasattr(model_cls, "load_model_joblib"):
            logger.warning(
                "load_trained_model failed: class has no load_model_joblib | model_name=%s",
                model_name,
            )
            print("⚠️ This model class does not support loading ‼️")
            return None

        # ---------- Load trained model ----------
        try:
            loaded_model = model_cls.load_model_joblib(
                filepath=filepath
            )  # Recall model loading method (Ensemble_BaseConfig)
        except Exception:
            logger.exception(
                "load_trained_model exception | model_name=%s | filepath=%s",
                model_name,
                filepath,
            )
            print("⚠️ Failed to load model ‼️")
            return None

        if loaded_model is None:
            logger.warning(
                "load_trained_model failed: loaded_model is None | model_name=%s | filepath=%s",
                model_name,
                filepath,
            )
            print("⚠️ Failed to load model ‼️")
            return None

        # ---------- Record loaded model info. ----------
        self.current_model = loaded_model
        self.current_model_name = model_name
        self.current_model_result = None

        logger.info(
            "Model loaded successfully | model_name=%s | filepath=%s",
            model_name,
            filepath,
        )
        print(f"🔥 Model loaded successfully: {model_name}")
        return loaded_model

    # -------------------- Predict with current model --------------------
    def predict_with_current_model(self, new_data):
        """
        Generate predictions using the current loaded or trained Apollo model.

        This method validates that the current model, fitted pipeline, and expected
        feature-name list are available. It then aligns the input data to the
        required feature order and calls the model pipeline's ``predict(...)``
        method.

        Parameters
        ----------
        new_data : pandas.DataFrame
            New input data used for prediction. It must contain all feature columns
            required by the current model.

        Returns
        -------
        Any or None
            Prediction output returned by the model pipeline when successful.
            Returns ``None`` if no current model is available, the model pipeline is
            missing, feature names are unavailable, or required columns are missing
            from ``new_data``.

        Notes
        -----
        Input columns are reordered to match the feature order stored in the
        current model before prediction is executed.
        """
        logger.info("Predicting with current model")

        if self.current_model is None:
            logger.warning("predict_with_current_model failed: current_model is None")
            print("⚠️ No current model available ‼️")
            return None

        # ---------- Get model metadata stored in pipeline ----------
        model_pipeline = getattr(self.current_model, "model_pipeline", None)
        if model_pipeline is None:
            logger.warning("predict_with_current_model failed: model_pipeline is None")
            print("⚠️ Current model pipeline is not available ‼️")
            return None

        # ---------- Get feature names stored in pipeline ----------
        feature_names = getattr(self.current_model, "feature_names", None)
        if feature_names is None:
            logger.warning("predict_with_current_model failed: feature_names is None")
            print("⚠️ Feature names are missing from current model ‼️")
            return None

        missing_cols = [col for col in feature_names if col not in new_data.columns]
        if missing_cols:
            logger.warning(
                "predict_with_current_model failed: missing columns=%s",
                missing_cols,
            )
            print(f"⚠️ Missing required columns: {missing_cols} ‼️")
            return None

        # ---------- Align data ----------
        aligned_data = new_data.loc[:, feature_names]
        predictions = model_pipeline.predict(aligned_data)

        logger.info(
            "Prediction completed successfully | sample_count=%s", len(aligned_data)
        )
        return predictions


# =================================================
