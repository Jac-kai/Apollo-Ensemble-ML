# -------------------- Import Modules --------------------
from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
report_root = os.path.join(project_root, "ES_ML_Report")

os.makedirs(report_root, exist_ok=True)


# -------------------- Ensemble base model configure --------------------
class EnsembleBaseModelConfig(ABC):
    """
    EnsembleBaseModelConfig
    =======================

    Abstract base class for sklearn-style ensemble model managers.

    Intended Purpose
    ----------------
    This base class is designed for ensemble-learning projects such as:

    - VotingClassifier / VotingRegressor
    - BaggingClassifier / BaggingRegressor
    - AdaBoostClassifier / AdaBoostRegressor
    - StackingClassifier / StackingRegressor

    It centralizes shared workflow components so each ensemble model module
    does not need to repeatedly implement:

    - cleaned X / Y storage
    - train / test splitting
    - optional preprocessing construction
    - optional GridSearchCV
    - feature-name extraction
    - prediction / metadata buffers

    Core Design Philosophy
    ----------------------
    Ensemble models often have two different preprocessing architectures:

    1) Outer unified preprocessing
       Example:
           Pipeline([
               ("preprocess", preprocessor),
               ("classifier", VotingClassifier(...))
           ])

       In this case, preprocessing is applied once before the ensemble model.

    2) Inner estimator-owned preprocessing
       Example:
           VotingClassifier(
               estimators=[
                   ("knn", knn_pipeline),
                   ("svm", svm_pipeline),
                   ("rf", rf_pipeline),
               ]
           )

       In this case, each base estimator already owns its own pipeline, so the
       outer layer should NOT build another preprocessing step.

    This base class supports both approaches via `use_preprocess`.

    Responsibilities
    ----------------
    - Store cleaned feature / target data.
    - Normalize single-output vs multi-output targets.
    - Perform train/test splitting with optional stratification.
    - Build a numeric/categorical preprocessing ColumnTransformer when needed.
    - Fit a full Pipeline with or without GridSearchCV.
    - Support both single-output scoring strings and subclass-defined
      multi-output scorers.
    - Extract feature names after preprocessing when applicable.
    - Provide helper validation for ensemble estimator lists.

    Subclass Contract
    -----------------
    Subclasses must implement:

    - task:
        Return either "classification" or "regression".

    - multioutput_scorer:
        A scorer function used only when:
        * GridSearchCV is enabled
        * Y is multi-output

    Notes
    -----
    - This class is intentionally ensemble-oriented, but can still fit any
      sklearn-compatible estimator or pipeline.
    - `step_name` is not abstract here. A default value is derived from `task`.
      You may override it in subclasses if desired.
    """

    # -------------------- Initialization --------------------
    def __init__(
        self,
        cleaned_X_data: pd.DataFrame,
        cleaned_Y_data: Union[pd.Series, pd.DataFrame],
    ):
        """
        Initialize the ensemble base manager.

        Parameters
        ----------
        cleaned_X_data : pd.DataFrame
            Cleaned input feature matrix.

        cleaned_Y_data : Union[pd.Series, pd.DataFrame]
            Cleaned target data.

            Normalization policy:
            - Series -> kept as Series (single-output)
            - DataFrame with 1 column -> converted to Series (single-output)
            - DataFrame with >= 2 columns -> kept as DataFrame (multi-output)

        Raises
        ------
        TypeError
            If cleaned_X_data is not a pandas DataFrame.
        TypeError
            If cleaned_Y_data is not a pandas Series or DataFrame.
        ValueError
            If cleaned_X_data is empty.
        ValueError
            If cleaned_Y_data is empty.

        Side Effects
        ------------
        Initializes shared runtime state:

        - split buffers:
        ``X_train``, ``X_test``, ``Y_train``, ``Y_test``

        - prediction buffers:
        ``y_train_pred``, ``y_test_pred``, ``prediction_preview``

        - fitted pipeline and feature-name cache:
        ``model_pipeline``, ``feature_names``

        - preprocessing column cache:
        ``_numeric_cols``, ``_categorical_cols``

        - runtime metadata:
        ``input_model_type``, ``input_use_cv``, ``input_cv_folds``,
        ``input_use_preprocess``, ``input_scoring``

        - CV result buffers:
        ``cv_search_report``, ``cv_results_raw``
        """
        if not isinstance(cleaned_X_data, pd.DataFrame):
            raise TypeError("⚠️ cleaned_X_data must be a pandas DataFrame ‼️")

        if cleaned_X_data.empty:
            raise ValueError("⚠️ cleaned_X_data is empty ‼️")

        if not isinstance(cleaned_Y_data, (pd.Series, pd.DataFrame)):
            raise TypeError("⚠️ cleaned_Y_data must be a pandas Series or DataFrame ‼️")

        if len(cleaned_Y_data) == 0:
            raise ValueError("⚠️ cleaned_Y_data is empty ‼️")

        self.cleaned_X_data = cleaned_X_data.copy()

        # ---------- Normalize Y storage ----------
        if isinstance(cleaned_Y_data, pd.DataFrame):
            if cleaned_Y_data.shape[1] == 1:
                self.cleaned_Y_data = cleaned_Y_data.iloc[:, 0].copy()
            else:
                self.cleaned_Y_data = cleaned_Y_data.copy()
        else:
            self.cleaned_Y_data = cleaned_Y_data.copy()

        # ---------- Split train and test dataset ----------
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        # ---------- Predictions of  train and test dataset ----------
        self.y_train_pred = None
        self.y_test_pred = None
        self.prediction_preview = None

        # ---------- Target encoding states ----------
        self.target_encoder = None
        self.target_encoders = {}
        self.target_classes_ = None
        self.multi_target_classes_ = {}
        self.y_label_encoded = False
        self.original_y_dtype = getattr(self.cleaned_Y_data, "dtype", None)
        self.original_y_name = getattr(self.cleaned_Y_data, "name", None)
        self.original_y_columns = (
            list(self.cleaned_Y_data.columns)
            if isinstance(self.cleaned_Y_data, pd.DataFrame)
            else None
        )

        # ---------- Fit target label encoder if needed ----------
        self._fit_target_label_encoder_if_needed()

        # ---------- Trained objects and feature names ----------
        self.model_pipeline = None
        self.feature_names = None

        # ---------- Numeric- and categorical-type columns ----------
        self._numeric_cols = None
        self._categorical_cols = None

        # ---------- Record metadata ----------
        self.input_model_type = None
        self.input_use_cv = None
        self.input_cv_folds = None
        self.input_use_preprocess = None
        self.input_scoring = None

        # ---------- Record CV report ----------
        self.cv_search_report = None
        self.cv_results_raw = None

    # -------------------- Task setup (necessory for child class) --------------------
    @property
    @abstractmethod
    def task(self) -> str:
        """
        Task type.

        Returns
        -------
        str
            Must be either:
            - "classification"
            - "regression"
        """

    # -------------------- Multiple output scoring method (necessory for child class) --------------------
    @staticmethod
    @abstractmethod
    def multioutput_scorer(y_true, y_pred) -> float:
        """
        Multi-output scorer used in GridSearchCV.

        Parameters
        ----------
        y_true : Any
            True target values.
        y_pred : Any
            Predicted target values.

        Returns
        -------
        float
            Aggregated score across all outputs.
        """

    # -------------------- Step name setup --------------------
    @property
    def step_name(self) -> str:
        """
        Default final estimator step name for the outer Pipeline.

        Returns
        -------
        str
            - "classifier" if task == "classification"
            - "regressor" if task == "regression"

        Notes
        -----
        You may override this property in subclasses if a more specific step
        name is needed.
        """
        if self.task == "classification":
            return "classifier"
        if self.task == "regression":
            return "regressor"
        raise ValueError("⚠️ task must be 'classification' or 'regression' ‼️")

    # -------------------- Check Y is multiple output --------------------
    def _is_multi_output(self, y: Union[pd.Series, pd.DataFrame]) -> bool:
        """
        Check whether target data is multi-output.

        Parameters
        ----------
        y : Union[pd.Series, pd.DataFrame]
            Target data to inspect.

        Returns
        -------
        bool
            True if y is a DataFrame with two or more columns, otherwise False.
        """
        return isinstance(y, pd.DataFrame) and y.shape[1] > 1

    # -------------------- Helper: Check target variable dimension (Encoding/Single-output) --------------------
    def _should_encode_single_target(self, y: pd.Series) -> bool:
        """
        Return whether a single-output target series should be label-encoded.

        This helper is used only for classification workflows. It checks whether
        the provided single-output target series has a label-like dtype that should
        be converted into numeric class indices before model training.

        Parameters
        ----------
        y : pd.Series
            Single-output target series to inspect.

        Returns
        -------
        bool
            True if all of the following conditions are satisfied:

            - ``self.task == "classification"``
            - ``y`` has one of these dtypes:
            - object
            - category
            - bool

            Otherwise returns False.

        Notes
        -----
        This helper does not perform the encoding itself. It only determines
        whether encoding should be applied.

        Numeric target labels such as 0, 1, 2 are treated as already usable for
        sklearn classification workflows and therefore are not encoded again.

        This method is intended only for single-output targets stored as a
        pandas Series.
        """
        return self.task == "classification" and (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
            or pd.api.types.is_bool_dtype(y)
        )

    # -------------------- Helper: Check target variable dimension (Encoding/Multiple-output) --------------------
    def _should_encode_multi_target_column(self, y_col: pd.Series) -> bool:
        """
        Return whether one multi-output target column should be label-encoded.

        This helper is used in multi-output classification workflows where the
        target data is stored as a pandas DataFrame. Each target column is checked
        independently so that only label-like columns are encoded.

        Parameters
        ----------
        y_col : pd.Series
            One target column from a multi-output target DataFrame.

        Returns
        -------
        bool
            True if all of the following conditions are satisfied:

            - ``self.task == "classification"``
            - ``y_col`` has one of these dtypes:
            - object
            - category
            - bool

            Otherwise returns False.

        Notes
        -----
        This helper allows mixed multi-output targets, for example when some target
        columns are categorical labels and other columns are already numeric.

        This method only decides whether encoding is needed. It does not perform
        the encoding itself.
        """
        return self.task == "classification" and (
            pd.api.types.is_object_dtype(y_col)
            or pd.api.types.is_categorical_dtype(y_col)
            or pd.api.types.is_bool_dtype(y_col)
        )

    # -------------------- Helper: Encoding target variable --------------------
    def _fit_target_label_encoder_if_needed(self):
        """
        Fit and apply target label encoding for classification targets when needed.

        This helper initializes and applies target-side label encoding before
        downstream workflows such as train/test split, model fitting, and
        prediction.

        Supported target formats are:

        - single-output target stored as ``pd.Series``
        - multi-output target stored as ``pd.DataFrame``

        Encoding policy
        ---------------
        - Regression targets are never encoded.
        - Single-output classification targets are encoded only when the target
        dtype is object, category, or bool.
        - Multi-output classification targets are processed column by column, and
        only columns with object, category, or bool dtype are encoded.

        Side Effects
        ------------
        Depending on the target structure, this method may update:

        - ``self.cleaned_Y_data``
        - ``self.target_encoder``
        - ``self.target_encoders``
        - ``self.target_classes_``
        - ``self.multi_target_classes_``
        - ``self.y_label_encoded``

        Returns
        -------
        None

        Notes
        -----
        For single-output classification, one shared ``LabelEncoder`` is stored in
        ``self.target_encoder``.

        For multi-output classification, one ``LabelEncoder`` is stored per encoded
        column in ``self.target_encoders``.

        Columns in a multi-output target DataFrame that are already numeric are left
        unchanged.

        This method is intended to be called during object initialization so that
        all later workflows operate on the encoded target representation when
        needed.
        """
        if self.task != "classification":
            return

        # ---------- Single-output ----------
        if isinstance(self.cleaned_Y_data, pd.Series):
            if not self._should_encode_single_target(self.cleaned_Y_data):
                return

            encoder = LabelEncoder()
            encoded = encoder.fit_transform(self.cleaned_Y_data)

            self.target_encoder = encoder
            self.target_classes_ = list(encoder.classes_)
            self.y_label_encoded = True
            self.cleaned_Y_data = pd.Series(
                encoded,
                index=self.cleaned_Y_data.index,
                name=self.original_y_name,
            )
            return

        # ---------- Multi-output ----------
        if isinstance(self.cleaned_Y_data, pd.DataFrame):
            encoded_df = self.cleaned_Y_data.copy()

            for col in encoded_df.columns:
                y_col = encoded_df[col]

                if self._should_encode_multi_target_column(y_col):
                    encoder = LabelEncoder()
                    encoded_df[col] = encoder.fit_transform(y_col)
                    self.target_encoders[col] = encoder
                    self.multi_target_classes_[col] = list(encoder.classes_)

            if self.target_encoders:
                self.y_label_encoded = True
                self.cleaned_Y_data = encoded_df

    # -------------------- Decoding target variable Y --------------------
    def decode_target_labels(self, preds):
        """
        Decode encoded classification predictions back to original target labels.

        This helper reverses target-side label encoding after model prediction.
        It supports both single-output and multi-output classification workflows.

        Parameters
        ----------
        preds : Any
            Prediction output returned by the fitted model pipeline.

            Typical forms include:

            - NumPy array
            - pandas Series
            - pandas DataFrame
            - other sklearn-compatible prediction outputs

        Returns
        -------
        Any
            Decoded prediction output when target label encoding is active.

            - For single-output classification, the returned object is typically a
            one-dimensional array-like structure containing original class labels.
            - For multi-output classification, the returned object is typically a
            pandas DataFrame with decoded target columns.

            If target encoding was not used, the original ``preds`` object is
            returned unchanged.

        Notes
        -----
        This helper is only meaningful for classification tasks.

        In multi-output workflows, only the target columns that were actually
        encoded are inverse-transformed. Columns that were not encoded remain
        unchanged.

        When multi-output predictions are provided as a non-DataFrame object, this
        method reconstructs a pandas DataFrame using ``self.original_y_columns``
        before applying column-wise inverse transformation.
        """
        if not self.y_label_encoded or self.task != "classification":
            return preds

        # ---------- Single-output ----------
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(preds)

        # ---------- Multi-output ----------
        if self.target_encoders:
            if isinstance(preds, pd.DataFrame):
                decoded_df = preds.copy()
            else:
                decoded_df = pd.DataFrame(
                    preds,
                    columns=self.original_y_columns,
                )

            for col, encoder in self.target_encoders.items():
                decoded_df[col] = encoder.inverse_transform(decoded_df[col])

            return decoded_df

        return preds

    # -------------------- Helper: decoding for prediction (missioners) --------------------
    def _maybe_decode_predictions(self, preds):
        """
        Decode prediction outputs when target label encoding is active.

        This is a lightweight wrapper around ``decode_target_labels()`` intended for
        mission-layer prediction helpers such as ``predict_engine()``.

        Parameters
        ----------
        preds : Any
            Raw prediction output returned by ``self.model_pipeline.predict(...)``.

        Returns
        -------
        Any
            Decoded prediction output when target-side label encoding was applied.
            Otherwise returns the original prediction output unchanged.

        Notes
        -----
        This helper allows mission-layer classes to remain simple and avoid
        duplicating target decoding logic.

        Typical usage is:

        - generate predictions from the fitted pipeline
        - pass predictions into ``self._maybe_decode_predictions(preds)``
        - store preview / return final decoded predictions
        """
        return self.decode_target_labels(preds)

    # -------------------- Check tasks setting --------------------
    def _validate_task(self):
        """
        Validate task value.

        Raises
        ------
        ValueError
            If task is not 'classification' or 'regression'.
        """
        if self.task not in {"classification", "regression"}:
            raise ValueError("⚠️ task must be 'classification' or 'regression' ‼️")

    # -------------------- Check estimators setting --------------------
    def _validate_estimators(
        self,
        estimators: Sequence[Tuple[str, Any]],
        min_estimators: int = 1,
    ) -> List[Tuple[str, Any]]:
        """
        Validate a sklearn-style estimator list.

        Parameters
        ----------
        estimators : Sequence[Tuple[str, Any]]
            Estimator list such as:
            [
                ("knn", knn_model),
                ("svm", svm_model),
            ]

        min_estimators : int, default=1
            Minimum number of estimators required.

        Returns
        -------
        List[Tuple[str, Any]]
            Validated estimator list.

        Raises
        ------
        TypeError
            If estimators is not a list/tuple-like collection.
        ValueError
            If estimator count is below min_estimators.
        ValueError
            If an estimator item is not a (name, estimator) pair.
        ValueError
            If estimator names are duplicated or invalid.
        """
        if not isinstance(estimators, (list, tuple)):
            raise TypeError(
                "⚠️ Estimators must be a list or tuple of (name, estimator) pairs ‼️"
            )

        if len(estimators) < min_estimators:
            raise ValueError(
                f"⚠️ At least {min_estimators} estimator(s) are required ‼️"
            )

        checked = []
        names = []

        for item in estimators:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(
                    "⚠️ Each estimator must be a (name, estimator) pair ‼️"
                )

            name, estimator = item

            if not isinstance(name, str) or not name.strip():
                raise ValueError("⚠️ Estimator name must be a non-empty string ‼️")

            if estimator is None:
                raise ValueError(f"⚠️ Estimator '{name}' is None ‼️")

            checked.append((name.strip(), estimator))
            names.append(name.strip())

        if len(set(names)) != len(names):
            raise ValueError("⚠️ Estimator names must be unique ‼️")

        return checked

    # -------------------- Build scalers --------------------
    def _build_scaler(self, scaler_type: str = "standard"):
        """
        Build a scaler instance from a user-specified scaler type.

        Parameters
        ----------
        scaler_type : str, default="standard"
            Supported values:
            - "standard", "std"
            - "minmax", "min_max"
            - "robust", "rbst"
            - "none", "no", "off"

        Returns
        -------
        Any or None
            Scaler instance, or None when no scaling is requested.

        Raises
        ------
        ValueError
            If scaler_type is unsupported.
        """
        scaler_type = scaler_type.lower().strip()

        if scaler_type in ["none", "no", "off"]:  # None Scaler
            return None
        if scaler_type in ["standard", "std"]:  # StandardScaler
            return StandardScaler()
        if scaler_type in ["minmax", "min_max"]:  # MinMaxScaler
            return MinMaxScaler()
        if scaler_type in ["robust", "rbst"]:  # RobustScaler
            return RobustScaler()

        raise ValueError(
            "⚠️ scaler_type must be 'standard', 'minmax', 'robust', or 'none' ‼️"
        )

    # -------------------- Split train and test dataset --------------------
    def train_test_split_engine(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ):
        """
        Split cleaned X/Y into training and testing sets.

        Parameters
        ----------
        test_size : float, default=0.2
            Fraction used as test set.

        random_state : int, default=42
            Random seed.

        stratify : bool, default=True
            Whether to stratify labels.

            Stratification is applied only when:
            - task == "classification"
            - target is single-output

        Returns
        -------
        tuple
            (X_train, X_test, Y_train, Y_test)

        Notes
        -----
        - Multi-output targets do not use stratification.
        """
        self._validate_task()

        y = self.cleaned_Y_data
        use_stratify = None

        # ---------- Classification with single Y ----------
        # Classification with multiple Y don't not use stratify
        if (
            self.task == "classification"
            and stratify
            and (not self._is_multi_output(y))
        ):
            use_stratify = y

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.cleaned_X_data,
            self.cleaned_Y_data,
            test_size=test_size,
            random_state=random_state,
            stratify=use_stratify,
        )

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    # -------------------- Preprocesses --------------------
    def build_preprocessor(
        self,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        cat_encoder: str = "ohe",
    ) -> ColumnTransformer:
        """
        Build a preprocessing ColumnTransformer.

        Parameters
        ----------
        categorical_cols : Optional[List[str]], default=None
            Categorical columns. If None, inferred from:
            object / category / bool dtypes.

        numeric_cols : Optional[List[str]], default=None
            Numeric columns. If None, inferred as all columns not included in
            categorical_cols.

        cat_encoder : str, default="ohe"
            Categorical encoder type:
            - "ohe", "onehot", "one_hot"
            - "ordinal", "ord", "ord_label"

        Returns
        -------
        ColumnTransformer
            Configured preprocessing transformer.

        Raises
        ------
        ValueError
            If cat_encoder is invalid.

        Side Effects
        ------------
        Stores:
        - self._numeric_cols
        - self._categorical_cols
        """
        df = self.cleaned_X_data

        if categorical_cols is None:
            categorical_cols = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

        if numeric_cols is None:
            numeric_cols = [col for col in df.columns if col not in categorical_cols]

        self._numeric_cols = list(numeric_cols)
        self._categorical_cols = list(categorical_cols)

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        cat_encoder = cat_encoder.lower().strip()
        if cat_encoder in ["ohe", "onehot", "one_hot"]:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
        elif cat_encoder in ["ordinal", "ord", "ord_label"]:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ord",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )
        else:
            raise ValueError("⚠️ cat_encoder must be 'ohe' or 'ordinal' ‼️")

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    # -------------------- CV setting --------------------
    def _build_cv_and_scoring(
        self,
        cv_folds: int,
        scoring: str,
        random_state: int = 42,
    ):
        """
        Build CV splitter and scoring strategy.

        Parameters
        ----------
        cv_folds : int
            Number of CV folds.
        scoring : str
            sklearn scoring string for single-output tasks.
        random_state : int, default=42
            Random seed for KFold / StratifiedKFold.

        Returns
        -------
        tuple
            (splitter, scoring_for_cv)
        """
        if self.task == "classification" and not self._is_multi_output(self.Y_train):
            splitter = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state,
            )
            scoring_for_cv = scoring
        else:
            splitter = KFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state,
            )
            scoring_for_cv = (
                make_scorer(self.multioutput_scorer, greater_is_better=True)
                if self._is_multi_output(self.Y_train)
                else scoring
            )

        return splitter, scoring_for_cv

    # -------------------- Fit model and CV --------------------
    def fit_with_grid(
        self,
        base_model: Any,
        param_grid: Optional[Dict[str, Any]],
        use_cv: bool,
        cv_folds: int,
        scoring: str,
        random_state: int = 42,
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
        use_preprocess: bool = True,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        cat_encoder: str = "ohe",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """
        Fit an outer Pipeline with optional preprocessing and optional GridSearchCV.

        This method builds a full sklearn outer pipeline for the current ensemble
        workflow and fits it on the training split. The pipeline can include:

        1. an optional outer preprocessing step,
        2. optional extra intermediate steps such as PCA or scaling,
        3. the final estimator step defined by ``self.step_name``.

        When cross-validation is enabled, the method runs ``GridSearchCV`` on the
        outer pipeline, stores the best fitted estimator pipeline, and records both:

        - a compact CV summary report in ``self.cv_search_report``
        - the full raw ``GridSearchCV.cv_results_`` dictionary in
        ``self.cv_results_raw``

        When cross-validation is disabled, the method fits the outer pipeline
        directly and clears previously stored CV result buffers so the object state
        remains consistent with the current training run.

        Parameters
        ----------
        base_model : Any
            Final estimator placed at the end of the outer Pipeline.
            This may be:
            - a plain sklearn estimator
            - an ensemble estimator
            - a Pipeline-wrapped estimator
            - a stacking / voting estimator

        param_grid : Optional[Dict[str, Any]]
            GridSearchCV parameter grid.

            Examples
            --------
            Outer-pipeline ensemble params:
            - ``{"classifier__n_estimators": [50, 100]}``
            - ``{"regressor__final_estimator__alpha": [0.1, 1.0]}``

        use_cv : bool
            Whether to perform GridSearchCV.

        cv_folds : int
            Number of folds for cross-validation.

        scoring : str
            sklearn scoring string for single-output tasks.
            Multi-output tasks use subclass-defined ``multioutput_scorer``.

        random_state : int, default=42
            Seed used by CV splitters.

        extra_steps : Optional[List[Tuple[str, Any]]], default=None
            Optional steps inserted between preprocessing and the final estimator.

            Example:
            - ``[("scaler", StandardScaler())]``
            - ``[("pca", PCA(n_components=5))]``

        use_preprocess : bool, default=True
            Whether the outer Pipeline should include a preprocessing step.

            Use True when:
            - raw tabular X is fed into the ensemble
            - preprocessing should happen once globally outside the ensemble

            Use False when:
            - each inner estimator already owns its own preprocessing pipeline
            - you do not want duplicated preprocessing

        categorical_cols : Optional[List[str]], default=None
            Passed to ``build_preprocessor()`` if ``use_preprocess=True``.

        numeric_cols : Optional[List[str]], default=None
            Passed to ``build_preprocessor()`` if ``use_preprocess=True``.

        cat_encoder : str, default="ohe"
            Passed to ``build_preprocessor()`` if ``use_preprocess=True``.

        Returns
        -------
        tuple
            ``(best_params, best_score)``

            best_params : Optional[Dict[str, Any]]
                Best parameter set from ``GridSearchCV``.
                ``None`` when ``use_cv=False``.

            best_score : Optional[float]
                Best CV score from ``GridSearchCV``.
                ``None`` when ``use_cv=False``.

        Raises
        ------
        ValueError
            If ``train_test_split_engine()`` has not been called first.

        Side Effects
        ------------
        Sets or updates:

        - ``self.model_pipeline``
        Fitted outer pipeline. When CV is used, this is ``gs.best_estimator_``.
        Otherwise, it is the directly fitted pipeline.

        - ``self.feature_names``
        Extracted feature names after fitting.

        - ``self.input_use_cv``
        Whether CV was enabled for the current run.

        - ``self.input_cv_folds``
        CV fold count used for the current run.

        - ``self.input_use_preprocess``
        Whether outer preprocessing was enabled for the current run.

        - ``self.input_scoring``
        Scoring method recorded for the current run.

        - ``self.cv_search_report``
        Compact CV summary dictionary when ``use_cv=True``. This includes:
        ``use_cv``, ``cv_folds``, ``scoring``, ``best_params``,
        ``best_cv_score``, and ``top_cv_results``.

        - ``self.cv_results_raw``
        Full raw ``GridSearchCV.cv_results_`` dictionary when ``use_cv=True``.

        When ``use_cv=False``, both ``self.cv_search_report`` and
        ``self.cv_results_raw`` are reset to ``None`` so no stale CV results remain
        from previous training runs.

        Notes
        -----
        CV splitter policy is delegated to ``_build_cv_and_scoring(...)``.

        Typical behavior:
        - classification + single-output -> ``StratifiedKFold``
        - otherwise -> ``KFold``

        If CV is used, the method also calls ``save_cv_search_report()`` to export
        the compact top-ranked CV result summary to CSV.

        This method stores two levels of CV result tracking:

        1. ``self.cv_search_report``
        A compact, human-readable summary intended for quick inspection and CSV
        export.

        2. ``self.cv_results_raw``
        The full raw ``GridSearchCV.cv_results_`` dictionary intended for deeper
        analysis, debugging, or later report expansion.
        """
        if self.X_train is None or self.Y_train is None:
            raise ValueError("⚠️ Run train_test_split_engine() before training ‼️")

        self._validate_task()  # Check task

        # ---------- Record metadata ----------
        self.input_use_cv = use_cv
        self.input_cv_folds = cv_folds
        self.input_use_preprocess = use_preprocess
        self.input_scoring = scoring

        # ---------- Setup steps ----------
        steps: List[Tuple[str, Any]] = []

        # ---------- Setup preprocess ----------
        if use_preprocess:
            preprocess = self.build_preprocessor(
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                cat_encoder=cat_encoder,
            )
            steps.append(("preprocess", preprocess))

        # ---------- Extra steps ----------
        if extra_steps:
            steps.extend(extra_steps)

        # ---------- Setup pipeline and input steps ----------
        steps.append((self.step_name, base_model))
        pipe = Pipeline(steps=steps)

        best_params = None
        best_score = None

        # ---------- CV application ----------
        if use_cv:
            splitter, scoring_for_cv = self._build_cv_and_scoring(
                cv_folds=cv_folds,
                scoring=scoring,
                random_state=random_state,
            )

            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid or {},
                scoring=scoring_for_cv,
                cv=splitter,
                n_jobs=-1,
                refit=True,
            )
            gs.fit(self.X_train, self.Y_train)  # CV training

            self.model_pipeline = gs.best_estimator_
            best_params = gs.best_params_
            best_score = gs.best_score_
            self.cv_results_raw = gs.cv_results_

            # ---------- Build compact CV report ----------
            cv_results_df = pd.DataFrame(gs.cv_results_)
            top_cv_results = (
                cv_results_df[
                    ["rank_test_score", "mean_test_score", "std_test_score", "params"]
                ]
                .sort_values("rank_test_score")
                .head(5)
                .to_dict(orient="records")
            )

            self.cv_search_report = {
                "use_cv": use_cv,
                "cv_folds": cv_folds,
                "scoring": scoring,
                "best_params": best_params,
                "best_cv_score": best_score,
                "top_cv_results": top_cv_results,
            }

            # ---------- Save compact CV report ----------
            self.save_cv_search_report()
        else:
            # ---------- Original model training ----------
            pipe.fit(self.X_train, self.Y_train)
            self.model_pipeline = pipe

            # ---------- Reset CV report ----------
            self.cv_search_report = None
            self.cv_results_raw = None

        self._extract_feature_names()
        return best_params, best_score

    # -------------------- Save CV report --------------------
    def save_cv_search_report(
        self,
        folder_name: str = "CV_Search_Report",
        file_name: str | None = None,
    ) -> str | None:
        """
        Save the stored CV search summary to a CSV file.

        This method exports the top-ranked cross-validation search results stored in
        ``self.cv_search_report["top_cv_results"]`` into a report folder under
        ``report_root``.

        Parameters
        ----------
        folder_name : str, default="CV_Search_Report"
            Folder name created under ``report_root`` for saved CV report files.

        file_name : str or None, default=None
            Optional custom CSV filename.
            If ``None``, a timestamp-based filename is generated automatically.

        Returns
        -------
        str or None
            Full saved file path if the report is successfully exported.
            Returns ``None`` if no CV search report or no top CV results are available.

        Side Effects
        ------------
        - Creates a CSV file under ``report_root / folder_name``.
        - Prints the saved file path when export succeeds.
        - Prints a warning message and returns ``None`` when export is skipped.

        Notes
        -----
        The saved CSV contains the top-ranked CV result rows with these fields:

        - ``rank_test_score``
        - ``mean_test_score``
        - ``std_test_score``
        - ``params``

        The default filename format is:

        ``{model_name}_cv_report_{YYYYMMDD_HHMMSS}.csv``

        where ``model_name`` comes from ``self.input_model_type``. If that value is
        unavailable, the current class name is used as fallback.

        This method exports the compact summary report only. The full raw GridSearchCV
        result dictionary remains available separately in ``self.cv_results_raw`` when
        CV was used.
        """
        if not self.cv_search_report:
            print("⚠️ No CV search report available to save ‼️")
            return None

        top_cv_results = self.cv_search_report.get("top_cv_results")
        if not top_cv_results:
            print("⚠️ No top CV results available to save ‼️")
            return None

        save_folder = os.path.join(report_root, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        if file_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = self.input_model_type or self.__class__.__name__
            file_name = f"{model_name}_cv_report_{timestamp}.csv"

        if not file_name.endswith(".csv"):
            file_name += ".csv"

        save_path = os.path.join(save_folder, file_name)

        cv_df = pd.DataFrame(top_cv_results)
        cv_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"📦 CV report saved path: {save_path}")
        return save_path

    # -------------------- Extract feature names --------------------
    def _extract_feature_names(self):
        """
        Extract feature names after fitting.

        Behavior
        --------
        Case 1: Outer pipeline contains "preprocess"
            Try ColumnTransformer.get_feature_names_out() first.

        Case 2: No outer preprocess exists
            Use original cleaned_X_data column names.

        Fallback logic
        --------------
        If transformed feature names cannot be extracted cleanly:
        - use cached numeric columns
        - expand OHE names when possible
        - otherwise fall back to numeric + categorical column lists
        """
        if self.model_pipeline is None:
            self.feature_names = None
            return

        # ---------- No outer preprocess ----------
        pre = self.model_pipeline.named_steps.get("preprocess", None)
        if pre is None:
            self.feature_names = self.cleaned_X_data.columns.tolist()
            return

        # ---------- Extract feature names ----------
        try:
            if hasattr(pre, "get_feature_names_out"):
                names = pre.get_feature_names_out()
                if names is not None:
                    names = list(names)

                    # block generic names like x0, x1, ...
                    if all(re.match(r"^[a-zA-Z]\d+$", str(name)) for name in names):
                        raise ValueError("generic feature names")

                    self.feature_names = names
                    return
            raise ValueError("no usable feature names")
        except Exception:
            # ---------- Backup methos-1 ----------
            num_cols = getattr(self, "_numeric_cols", None)
            cat_cols = getattr(self, "_categorical_cols", None)

            if num_cols is None or cat_cols is None:
                self.feature_names = self.cleaned_X_data.columns.tolist()
                return

            names_out = list(num_cols)
            try:
                cat_pipe = pre.named_transformers_.get("cat", None)

                if (
                    cat_pipe is not None
                    and hasattr(cat_pipe, "named_steps")
                    and ("ohe" in cat_pipe.named_steps)
                ):
                    ohe = cat_pipe.named_steps["ohe"]
                    names_out.extend(ohe.get_feature_names_out(cat_cols).tolist())
                else:
                    names_out.extend(list(cat_cols))

                self.feature_names = names_out

            except Exception:
                # ---------- Backup methos-2 ----------
                self.feature_names = list(num_cols) + list(cat_cols)

    # -------------------- Save fitted model --------------------
    def save_model_joblib(
        self,
        folder_path: str | None = None,
        file_name: str | None = None,
    ):
        """
        Save the current fitted ensemble model object as a joblib file.

        Parameters
        ----------
        folder_path : str or None, default=None
            Folder path used to save the joblib file.

            If None, the current working directory is used.

        file_name : str or None, default=None
            Custom file name for the saved model.

            If None, an automatic file name is generated using the class name
            and current timestamp.

        Returns
        -------
        str or None
            Full saved file path if successful, otherwise None.

        Raises
        ------
        ValueError
            If no fitted pipeline is available.

        Notes
        -----
        This method saves the entire current object instead of saving only
        ``self.model_pipeline`` so that runtime state is preserved, including:

        - fitted pipeline
        - feature names
        - cleaned X / Y references
        - train/test split buffers
        - prediction buffers
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ No fitted model pipeline available to save ‼️")

        if folder_path is None:
            folder_path = os.getcwd()

        os.makedirs(folder_path, exist_ok=True)

        if file_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"{self.__class__.__name__}_{timestamp}.joblib"

        if not file_name.endswith(".joblib"):
            file_name += ".joblib"

        file_path = os.path.join(folder_path, file_name)
        joblib.dump(self, file_path)

        return file_path

    # -------------------- Load fitted model --------------------
    @classmethod
    def load_model_joblib(cls, filepath: str):
        """
        Load a previously saved fitted ensemble model object from a joblib file.

        Parameters
        ----------
        filepath : str
            Path to the saved joblib file.

        Returns
        -------
        Any
            Loaded model object.

        Raises
        ------
        FileNotFoundError
            If the given file path does not exist.

        TypeError
            If the loaded object is not an instance of the expected class.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"⚠️ File not found: {filepath} ‼️")

        loaded_obj = joblib.load(filepath)

        if not isinstance(loaded_obj, cls):
            raise TypeError(
                f"⚠️ Loaded object type mismatch: expected {cls.__name__}, "
                f"got {type(loaded_obj).__name__} ‼️"
            )

        return loaded_obj

    # -------------------- Permutation importance engine --------------------
    def permutation_importance_engine(
        self,
        n_repeats: int = 10,
        scoring: str | None = None,
        max_display: int = 20,
        save_fig: bool = False,
        folder_name: str = "Permutation_Importance",
        file_name: str | None = None,
    ):
        """
        Compute and optionally plot permutation importance on the test set.

        This method evaluates feature importance by repeatedly shuffling each
        original input feature column in ``self.X_test`` and measuring the
        resulting drop in model performance on ``self.Y_test``.

        Parameters
        ----------
        n_repeats : int, default=10
            Number of shuffle repetitions used for each feature.

        scoring : str or None, default=None
            Scoring method passed to ``sklearn.inspection.permutation_importance``.

            If ``None``, a task-based default is used:
            - ``"accuracy"`` for classification
            - ``"r2"`` for regression

        max_display : int, default=20
            Maximum number of features displayed in the output plot and preview.

        save_fig : bool, default=False
            Whether to save the generated permutation-importance plot.

        folder_name : str, default="Permutation_Importance"
            Folder name created under ``report_root`` when saving the figure.

        file_name : str or None, default=None
            Optional custom figure filename.
            If ``None``, a timestamp-based filename is generated automatically.

        Returns
        -------
        pd.DataFrame
            A DataFrame sorted by descending importance, containing:

            - ``feature``:
            Original input feature name from ``self.X_test.columns``

            - ``importance_mean``:
            Mean permutation importance across repeated shuffles

            - ``importance_std``:
            Standard deviation of permutation importance across repeated shuffles

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        ValueError
            If test data is unavailable.
        ValueError
            If the number of original input feature names does not match the
            number of permutation-importance results.

        Notes
        -----
        This method intentionally uses the original input feature names from
        ``self.X_test.columns`` rather than transformed feature names stored in
        ``self.feature_names``.

        Reason:
        - ``permutation_importance(...)`` is executed on ``X=self.X_test``
        - therefore, the returned importance values correspond to the original
        input columns
        - they do NOT correspond to post-preprocessing expanded columns such as
        one-hot encoded feature names

        This distinction is especially important when outer preprocessing uses
        encoders like ``OneHotEncoder``. In such cases, one original categorical
        feature may expand into multiple transformed columns, but permutation
        importance still returns one importance value per original input feature.

        Examples
        --------
        If the original input feature set is:

            ["odor", "habitat", "population"]

        then the returned importance table also contains exactly those three
        feature labels.

        If ``odor`` is one-hot encoded internally into multiple transformed
        columns, permutation importance still reports one importance value for
        the original ``odor`` column because the shuffle is applied at the raw
        input-data level before the pipeline preprocessing step.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Train the model before permutation importance ‼️")

        if self.X_test is None or self.Y_test is None:
            raise ValueError(
                "⚠️ Test data is unavailable for permutation importance ‼️"
            )

        if scoring is None:
            scoring = "accuracy" if self.task == "classification" else "r2"

        result = permutation_importance(
            estimator=self.model_pipeline,
            X=self.X_test,
            y=self.Y_test,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
        )

        feature_names = self.X_test.columns.tolist()

        if len(feature_names) != len(result.importances_mean):
            raise ValueError(
                "⚠️ Feature name count does not match permutation importance result length ‼️"
            )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        display_df = importance_df.head(max_display).iloc[::-1]

        plt.figure(figsize=(10, max(5, len(display_df) * 0.35)))
        plt.barh(display_df["feature"], display_df["importance_mean"])
        plt.xlabel("Permutation Importance")
        plt.ylabel("Feature")
        plt.title(f"{self.__class__.__name__} Permutation Importance")
        plt.tight_layout()

        if save_fig:
            save_folder = os.path.join(report_root, folder_name)
            os.makedirs(save_folder, exist_ok=True)

            if file_name is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_name = (
                    f"{self.__class__.__name__}_permutation_importance_{timestamp}.png"
                )

            save_path = os.path.join(save_folder, file_name)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"🔥 Permutation importance figure saved: {save_path}")

        plt.show()
        return importance_df


# =================================================
