# -------------------- Import Modules --------------------
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from Apollo.Backbone.Ensemble_BaseConfig import report_root
from Apollo.Ensemble_Missioner.Bagging_Missioner import Bagging_Missioner


# -------------------- Bagging classifier model --------------------
class BaggingClassifier_Model(Bagging_Missioner):
    """
    Train, evaluate, and visualize a bagging-based classification model.

    This class provides the model-layer workflow for classification tasks based on
    ``BaggingClassifier`` through the inherited bagging mission pipeline. It is
    responsible for declaring the task type, defining classification-specific
    scoring behavior, training the bagging ensemble, evaluating fitted models,
    and visualizing confusion matrices.

    The class supports both single-output and multi-output classification tasks.
    For multi-output classification, evaluation metrics are computed separately
    for each target column and then aggregated by mean.

    Architecture Role
    -----------------
    This class belongs to the model layer in the project architecture.

    A typical responsibility split is:

    - base configuration layer
        Handles shared preprocessing, train/test splitting, pipeline building,
        and generic fitting or GridSearchCV workflows.

    - mission layer
        Handles bagging-specific estimator construction, validation, prediction
        helpers, and fitted-estimator access.

    - model layer
        Handles task declaration, scoring behavior, evaluation logic, and the
        user-facing training interface.

    Expected State
    --------------
    Before calling ``train()``, the inherited train/test split workflow is
    expected to have already prepared:

    - ``self.X_train``
    - ``self.X_test``
    - ``self.Y_train``
    - ``self.Y_test``

    Attributes
    ----------
    task : str
        Read-only property returning ``"classification"``.

    Notes
    -----
    This class delegates bagging model construction and fitting to the inherited
    mission-layer implementation and focuses on classification-specific model
    behavior only.

    The fitted model pipeline is stored in ``self.model_pipeline``. Predictions
    generated during evaluation are stored in ``self.y_train_pred`` and
    ``self.y_test_pred``.
    """

    # -------------------- Task type --------------------
    @property
    def task(self) -> str:
        """
        Return the task type handled by this model class.

        Returns
        -------
        str
            Fixed task identifier ``"classification"``.

        Notes
        -----
        This property is used by inherited mission-layer logic to determine
        whether classification-specific estimators, scoring behavior, and
        evaluation rules should be applied.
        """
        return "classification"

    # -------------------- Multi-output scoring --------------------
    @staticmethod
    def multioutput_scorer(y_true, y_pred) -> float:
        """
        Compute a classification score for single-output or multi-output targets.

        This scorer is designed for use in GridSearchCV or related model-selection
        workflows. For single-output classification, it returns the weighted F1
        score directly. For multi-output classification, it computes the weighted
        F1 score separately for each target column and returns the arithmetic mean
        across all target columns.

        Parameters
        ----------
        y_true : pandas.Series, pandas.DataFrame, or numpy.ndarray
            Ground-truth target values.
            - If 1-dimensional, the task is treated as single-output classification.
            - If 2-dimensional, the task is treated as multi-output classification.

        y_pred : pandas.Series, pandas.DataFrame, or numpy.ndarray
            Predicted target values produced by the model.

        Returns
        -------
        float
            Weighted F1 score for single-output classification, or the mean of
            per-target weighted F1 scores for multi-output classification.

        Notes
        -----
        For single-output classification, this method computes:

        - ``f1_score(y_true, y_pred, average="weighted")``

        For multi-output classification, the method computes weighted F1
        separately for each target column and then returns the arithmetic mean
        across all targets.

        If ``y_true`` is not already a DataFrame in multi-output mode, it is
        converted to one. ``y_pred`` is also converted when necessary so that
        column alignment can follow ``y_true.columns``.
        """
        if isinstance(y_true, pd.Series) or (
            isinstance(y_true, np.ndarray) and y_true.ndim == 1
        ):
            return float(f1_score(y_true, y_pred, average="weighted"))

        if not isinstance(y_true, pd.DataFrame):
            y_true = pd.DataFrame(y_true)

        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y_true.columns)

        scores = [
            f1_score(y_true[col], y_pred[col], average="weighted")
            for col in y_true.columns
        ]
        return float(np.mean(scores)) if scores else 0.0

    # -------------------- Train bagging classifier --------------------
    def train(
        self,
        estimator: Any,
        n_estimators: int = 10,
        use_cv: bool = False,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
        max_samples: float | int = 1.0,
        max_features: float | int = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: Optional[int] = -1,
        param_grid: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
        use_preprocess: bool = True,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        cat_encoder: str = "ohe",
    ) -> Dict[str, Any]:
        """
        Train a bagging classifier and return a full training summary.

        This method fits a classification bagging ensemble using the inherited
        ``bagging_fit_engine()`` workflow, then evaluates the fitted model on
        both training and test sets through ``model_evaluation_engine()``.

        Parameters
        ----------
        estimator : Any
            Base estimator used inside the bagging ensemble.

        n_estimators : int, default=10
            Number of base estimators in the ensemble.

        use_cv : bool, default=False
            Whether to perform hyperparameter search with GridSearchCV on the
            outer pipeline.

        cv_folds : int, default=5
            Number of folds used for outer cross-validation when ``use_cv=True``.

        scoring : str, default="f1_weighted"
            Scikit-learn scoring name used for model selection in single-output
            classification tasks.

        max_samples : float or int, default=1.0
            Number or proportion of samples drawn from the training set to train
            each base estimator.

        max_features : float or int, default=1.0
            Number or proportion of features drawn from the input features to
            train each base estimator.

        bootstrap : bool, default=True
            Whether samples are drawn with replacement.

        bootstrap_features : bool, default=False
            Whether features are drawn with replacement.

        oob_score : bool, default=False
            Whether to compute out-of-bag estimates when bootstrap sampling is
            enabled.

        warm_start : bool, default=False
            Whether to reuse the solution of the previous fit call and add more
            estimators to the ensemble.

        n_jobs : int or None, default=-1
            Number of parallel jobs used where supported by the bagging
            estimator.

        param_grid : dict[str, Any] or None, default=None
            Parameter grid passed to GridSearchCV for outer pipeline search.

        random_state : int, default=42
            Random seed used for reproducibility where applicable.

        extra_steps : list[tuple[str, Any]] or None, default=None
            Optional pipeline steps inserted between preprocessing and the final
            bagging classifier in the outer pipeline.

        use_preprocess : bool, default=True
            Whether to build and use the outer preprocessing pipeline.

        categorical_cols : list[str] or None, default=None
            Categorical columns passed to the inherited preprocessing builder when
            outer preprocessing is enabled.

        numeric_cols : list[str] or None, default=None
            Numeric columns passed to the inherited preprocessing builder when
            outer preprocessing is enabled.

        cat_encoder : str, default="ohe"
            Encoding strategy passed to the inherited preprocessing builder when
            outer preprocessing is enabled.

        Returns
        -------
        dict[str, Any]
            Full training summary dictionary containing:

            - fit summary returned by ``bagging_fit_engine()``
            - ``feature_names_len`` : number of tracked transformed feature names
            - ``evaluation`` : evaluation dictionary returned by
              ``model_evaluation_engine()``

        Raises
        ------
        ValueError
            If ``X_train`` or ``Y_train`` is unavailable before training.

        Notes
        -----
        This method assumes that train/test splitting has already been completed.

        After fitting, the model is immediately evaluated on both train and test
        data. For multi-output classification, the returned evaluation includes
        both per-target metrics and mean aggregated metrics.
        """
        if self.X_train is None or self.Y_train is None:
            raise ValueError("⚠️ Run train_test_split_engine() before training ‼️")

        fit_summary = self.bagging_fit_engine(
            estimator=estimator,
            n_estimators=n_estimators,
            use_cv=use_cv,
            cv_folds=cv_folds,
            scoring=scoring,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            param_grid=param_grid,
            random_state=random_state,
            extra_steps=extra_steps,
            use_preprocess=use_preprocess,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            cat_encoder=cat_encoder,
        )

        eval_results = self.model_evaluation_engine()

        return {
            **fit_summary,
            "feature_names_len": (
                None if self.feature_names is None else len(self.feature_names)
            ),
            "evaluation": eval_results,
        }

    # -------------------- Classification evaluation --------------------
    def model_evaluation_engine(self) -> Dict[str, Any]:
        """
        Evaluate the fitted bagging classifier on training and test datasets.

        This method generates predictions from the fitted model pipeline for both
        ``X_train`` and ``X_test``, then computes classification metrics.

        Evaluation Behavior
        -------------------
        Single-output classification:
            - accuracy
            - weighted precision
            - weighted recall
            - weighted F1

        Multi-output classification:
            - per-target accuracy
            - per-target weighted precision
            - per-target weighted recall
            - per-target weighted F1
            - mean of each metric across all target columns

        Returns
        -------
        dict[str, Any]
            Evaluation-result dictionary.

            For single-output classification, the returned dictionary contains:
            - ``train_accuracy``
            - ``test_accuracy``
            - ``train_precision_weighted``
            - ``test_precision_weighted``
            - ``train_recall_weighted``
            - ``test_recall_weighted``
            - ``train_f1_weighted``
            - ``test_f1_weighted``

            For multi-output classification, the returned dictionary contains:
            - mean metrics across targets
            - per-target metric dictionaries for train and test data

        Raises
        ------
        ValueError
            If ``self.model_pipeline`` is not fitted or missing.

        ValueError
            If train/test feature or target data is unavailable.

        Notes
        -----
        Predictions are stored in:
        - ``self.y_train_pred``
        - ``self.y_test_pred``

        For multi-output classification, this method assumes that predictions are
        returned as a 2-dimensional array whose column order matches
        ``self.Y_train.columns`` and ``self.Y_test.columns``.

        Weighted precision and weighted recall are computed with
        ``zero_division=0`` to avoid warnings when a class has no predicted
        samples.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Train the model before evaluation ‼️")

        if (
            self.X_train is None
            or self.X_test is None
            or self.Y_train is None
            or self.Y_test is None
        ):
            raise ValueError("⚠️ Train/test data is unavailable for evaluation ‼️")

        self.y_train_pred = self.model_pipeline.predict(self.X_train)
        self.y_test_pred = self.model_pipeline.predict(self.X_test)

        if self._is_multi_output(self.Y_test):
            train_accuracy_per_target = {
                col: float(accuracy_score(self.Y_train[col], self.y_train_pred[:, idx]))
                for idx, col in enumerate(self.Y_train.columns)
            }
            test_accuracy_per_target = {
                col: float(accuracy_score(self.Y_test[col], self.y_test_pred[:, idx]))
                for idx, col in enumerate(self.Y_test.columns)
            }

            train_precision_weighted_per_target = {
                col: float(
                    precision_score(
                        self.Y_train[col],
                        self.y_train_pred[:, idx],
                        average="weighted",
                        zero_division=0,
                    )
                )
                for idx, col in enumerate(self.Y_train.columns)
            }
            test_precision_weighted_per_target = {
                col: float(
                    precision_score(
                        self.Y_test[col],
                        self.y_test_pred[:, idx],
                        average="weighted",
                        zero_division=0,
                    )
                )
                for idx, col in enumerate(self.Y_test.columns)
            }

            train_recall_weighted_per_target = {
                col: float(
                    recall_score(
                        self.Y_train[col],
                        self.y_train_pred[:, idx],
                        average="weighted",
                        zero_division=0,
                    )
                )
                for idx, col in enumerate(self.Y_train.columns)
            }
            test_recall_weighted_per_target = {
                col: float(
                    recall_score(
                        self.Y_test[col],
                        self.y_test_pred[:, idx],
                        average="weighted",
                        zero_division=0,
                    )
                )
                for idx, col in enumerate(self.Y_test.columns)
            }

            train_f1_weighted_per_target = {
                col: float(
                    f1_score(
                        self.Y_train[col],
                        self.y_train_pred[:, idx],
                        average="weighted",
                    )
                )
                for idx, col in enumerate(self.Y_train.columns)
            }
            test_f1_weighted_per_target = {
                col: float(
                    f1_score(
                        self.Y_test[col],
                        self.y_test_pred[:, idx],
                        average="weighted",
                    )
                )
                for idx, col in enumerate(self.Y_test.columns)
            }

            return {
                "train_accuracy_mean": float(
                    np.mean(list(train_accuracy_per_target.values()))
                ),
                "test_accuracy_mean": float(
                    np.mean(list(test_accuracy_per_target.values()))
                ),
                "train_precision_weighted_mean": float(
                    np.mean(list(train_precision_weighted_per_target.values()))
                ),
                "test_precision_weighted_mean": float(
                    np.mean(list(test_precision_weighted_per_target.values()))
                ),
                "train_recall_weighted_mean": float(
                    np.mean(list(train_recall_weighted_per_target.values()))
                ),
                "test_recall_weighted_mean": float(
                    np.mean(list(test_recall_weighted_per_target.values()))
                ),
                "train_f1_weighted_mean": float(
                    np.mean(list(train_f1_weighted_per_target.values()))
                ),
                "test_f1_weighted_mean": float(
                    np.mean(list(test_f1_weighted_per_target.values()))
                ),
                "train_accuracy_per_target": train_accuracy_per_target,
                "test_accuracy_per_target": test_accuracy_per_target,
                "train_precision_weighted_per_target": train_precision_weighted_per_target,
                "test_precision_weighted_per_target": test_precision_weighted_per_target,
                "train_recall_weighted_per_target": train_recall_weighted_per_target,
                "test_recall_weighted_per_target": test_recall_weighted_per_target,
                "train_f1_weighted_per_target": train_f1_weighted_per_target,
                "test_f1_weighted_per_target": test_f1_weighted_per_target,
            }

        return {
            "train_accuracy": float(accuracy_score(self.Y_train, self.y_train_pred)),
            "test_accuracy": float(accuracy_score(self.Y_test, self.y_test_pred)),
            "train_precision_weighted": float(
                precision_score(
                    self.Y_train,
                    self.y_train_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "test_precision_weighted": float(
                precision_score(
                    self.Y_test,
                    self.y_test_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "train_recall_weighted": float(
                recall_score(
                    self.Y_train,
                    self.y_train_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "test_recall_weighted": float(
                recall_score(
                    self.Y_test,
                    self.y_test_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "train_f1_weighted": float(
                f1_score(self.Y_train, self.y_train_pred, average="weighted")
            ),
            "test_f1_weighted": float(
                f1_score(self.Y_test, self.y_test_pred, average="weighted")
            ),
        }

    # -------------------- Confusion matrix engine --------------------
    def confusion_matrix_engine(
        self,
        target_col: str | None = None,
        normalize: str | None = None,
        save_fig: bool = False,
        folder_name: str = "Confusion_Matrix",
        file_name: str | None = None,
    ):
        """
        Plot and optionally save a confusion matrix for the fitted classifier.

        This method generates predictions on the test set and plots a confusion
        matrix using scikit-learn's ``ConfusionMatrixDisplay``.

        For single-output classification, the confusion matrix is computed
        directly from ``Y_test`` and predicted labels.

        For multi-output classification, a specific target column must be
        selected through ``target_col`` so that one confusion matrix can be
        plotted for that target at a time.

        Parameters
        ----------
        target_col : str or None, default=None
            Target-column name used for multi-output classification plotting.
            This parameter is required when ``Y_test`` is multi-output.
            It is ignored for single-output classification.

        normalize : str or None, default=None
            Normalization mode forwarded to ``sklearn.metrics.confusion_matrix()``.

            Common values include:
            - ``None``   : raw counts
            - ``"true"`` : normalize over true labels
            - ``"pred"`` : normalize over predicted labels
            - ``"all"``  : normalize over all entries

        save_fig : bool, default=False
            Whether to save the plotted figure as an image file.

        folder_name : str, default="Confusion_Matrix"
            Folder name created under ``report_root`` when ``save_fig=True``.

        file_name : str or None, default=None
            Custom output filename. If ``None``, an automatic filename is
            generated from the class name, optional target name, and timestamp.

        Returns
        -------
        None
            This method displays the plot with ``matplotlib.pyplot.show()`` and
            optionally saves it to disk.

        Raises
        ------
        ValueError
            If the model pipeline has not been trained.

        ValueError
            If test data is unavailable.

        ValueError
            If the task is multi-output classification and ``target_col`` is not
            provided.

        ValueError
            If ``target_col`` is not found in ``self.Y_test.columns``.

        Notes
        -----
        When ``save_fig=True``, the figure is saved under:
        ``os.path.join(report_root, folder_name)``

        The plot title includes the class name and, for multi-output cases, the
        selected target column name.

        The displayed value format is:
        - integer counts when ``normalize is None``
        - 2-decimal float format when normalization is enabled

        The plot is closed with ``plt.close()`` after display to reduce figure
        accumulation in repeated plotting workflows.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Train the model before confusion matrix plotting ‼️")

        if self.X_test is None or self.Y_test is None:
            raise ValueError(
                "⚠️ Test data is unavailable for confusion matrix plotting ‼️"
            )

        y_pred = self.model_pipeline.predict(self.X_test)

        # ---------- Multi-output ----------
        if self._is_multi_output(self.Y_test):
            if target_col is None:
                raise ValueError(
                    "⚠️ target_col is required for multi-output confusion matrix ‼️"
                )

            if target_col not in self.Y_test.columns:
                raise ValueError(f"⚠️ target_col '{target_col}' not found in Y_test ‼️")

            col_idx = list(self.Y_test.columns).index(target_col)
            y_true_plot = self.Y_test[target_col]
            y_pred_plot = y_pred[:, col_idx]
            title = f"{self.__class__.__name__} Confusion Matrix ({target_col})"

        # ---------- Single-output ----------
        else:
            y_true_plot = self.Y_test
            y_pred_plot = y_pred
            title = f"{self.__class__.__name__} Confusion Matrix"

        cm = confusion_matrix(y_true_plot, y_pred_plot, normalize=normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        plt.figure(figsize=(7, 6))
        disp.plot(values_format=".2f" if normalize else "d")
        plt.title(title)
        plt.tight_layout()

        if save_fig:
            save_folder = os.path.join(report_root, folder_name)
            os.makedirs(save_folder, exist_ok=True)

            if file_name is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                safe_suffix = f"_{target_col}" if target_col else ""
                file_name = f"{self.__class__.__name__}_confusion_matrix{safe_suffix}_{timestamp}.png"

            save_path = os.path.join(save_folder, file_name)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"🔥 Confusion matrix figure saved: {save_path}")

        plt.show()
        plt.close()


# -------------------- Bagging regressor model --------------------
class BaggingRegressor_Model(Bagging_Missioner):
    """
    Train and evaluate a bagging-based regression model.

    This class provides the model-layer workflow for regression tasks based on
    ``BaggingRegressor`` through the inherited bagging mission pipeline. It is
    responsible for declaring the task type, defining regression-specific
    scoring behavior, training the bagging ensemble, and evaluating fitted
    models.

    The class supports both single-output and multi-output regression tasks.
    For multi-output regression, evaluation metrics are computed separately for
    each target column and then aggregated by mean.

    Architecture Role
    -----------------
    This class belongs to the model layer in the project architecture.

    A typical responsibility split is:

    - base configuration layer
        Handles shared preprocessing, train/test splitting, pipeline building,
        and generic fitting or GridSearchCV workflows.

    - mission layer
        Handles bagging-specific estimator construction, validation, prediction
        helpers, and fitted-estimator access.

    - model layer
        Handles task declaration, scoring behavior, evaluation logic, and the
        user-facing training interface.

    Expected State
    --------------
    Before calling ``train()``, the inherited train/test split workflow is
    expected to have already prepared:

    - ``self.X_train``
    - ``self.X_test``
    - ``self.Y_train``
    - ``self.Y_test``

    Attributes
    ----------
    task : str
        Read-only property returning ``"regression"``.

    Notes
    -----
    This class delegates bagging model construction and fitting to the inherited
    mission-layer implementation and focuses on regression-specific model
    behavior only.

    The fitted model pipeline is stored in ``self.model_pipeline``. Predictions
    generated during evaluation are stored in ``self.y_train_pred`` and
    ``self.y_test_pred``.
    """

    # -------------------- Task type --------------------
    @property
    def task(self) -> str:
        """
        Return the task type handled by this model class.

        Returns
        -------
        str
            Fixed task identifier ``"regression"``.

        Notes
        -----
        This property is used by inherited mission-layer logic to determine
        whether regression-specific estimators, scoring behavior, and evaluation
        rules should be applied.
        """
        return "regression"

    # -------------------- Multi-output scoring --------------------
    @staticmethod
    def multioutput_scorer(y_true, y_pred) -> float:
        """
        Compute a regression score for single-output or multi-output targets.

        This scorer is designed for use in GridSearchCV or related
        model-selection workflows. For single-output regression, it returns the
        R² score directly. For multi-output regression, it computes the R² score
        separately for each target column and returns the arithmetic mean across
        all target columns.

        Parameters
        ----------
        y_true : pandas.Series, pandas.DataFrame, or numpy.ndarray
            Ground-truth target values.
            - If 1-dimensional, the task is treated as single-output regression.
            - If 2-dimensional, the task is treated as multi-output regression.

        y_pred : pandas.Series, pandas.DataFrame, or numpy.ndarray
            Predicted target values produced by the model.

        Returns
        -------
        float
            R² score for single-output regression, or the mean of per-target R²
            scores for multi-output regression.

        Notes
        -----
        For multi-output regression, ``y_true`` and ``y_pred`` are converted to
        DataFrame form when necessary so that target-column alignment can follow
        ``y_true.columns``.
        """
        if isinstance(y_true, pd.Series) or (
            isinstance(y_true, np.ndarray) and y_true.ndim == 1
        ):
            return float(r2_score(y_true, y_pred))

        if not isinstance(y_true, pd.DataFrame):
            y_true = pd.DataFrame(y_true)

        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y_true.columns)

        scores = [r2_score(y_true[col], y_pred[col]) for col in y_true.columns]
        return float(np.mean(scores)) if scores else 0.0

    # -------------------- Train bagging regressor --------------------
    def train(
        self,
        estimator: Any,
        n_estimators: int = 10,
        use_cv: bool = False,
        cv_folds: int = 5,
        scoring: str = "r2",
        max_samples: float | int = 1.0,
        max_features: float | int = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: Optional[int] = -1,
        param_grid: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
        use_preprocess: bool = True,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        cat_encoder: str = "ohe",
    ) -> Dict[str, Any]:
        """
        Train a bagging regressor and return a full training summary.

        This method fits a regression bagging ensemble using the inherited
        ``bagging_fit_engine()`` workflow, then evaluates the fitted model on
        both training and test sets through ``model_evaluation_engine()``.

        Parameters
        ----------
        estimator : Any
            Base estimator used inside the bagging ensemble.

        n_estimators : int, default=10
            Number of base estimators in the ensemble.

        use_cv : bool, default=False
            Whether to perform hyperparameter search with GridSearchCV on the
            outer pipeline.

        cv_folds : int, default=5
            Number of folds used for outer cross-validation when ``use_cv=True``.

        scoring : str, default="r2"
            Scikit-learn scoring name used for model selection in single-output
            regression tasks.

        max_samples : float or int, default=1.0
            Number or proportion of samples drawn from the training set to train
            each base estimator.

        max_features : float or int, default=1.0
            Number or proportion of features drawn from the input features to
            train each base estimator.

        bootstrap : bool, default=True
            Whether samples are drawn with replacement.

        bootstrap_features : bool, default=False
            Whether features are drawn with replacement.

        oob_score : bool, default=False
            Whether to compute out-of-bag estimates when bootstrap sampling is
            enabled.

        warm_start : bool, default=False
            Whether to reuse the solution of the previous fit call and add more
            estimators to the ensemble.

        n_jobs : int or None, default=-1
            Number of parallel jobs used where supported by the bagging
            estimator.

        param_grid : dict[str, Any] or None, default=None
            Parameter grid passed to GridSearchCV for outer pipeline search.

        random_state : int, default=42
            Random seed used for reproducibility where applicable.

        extra_steps : list[tuple[str, Any]] or None, default=None
            Optional pipeline steps inserted between preprocessing and the final
            bagging regressor in the outer pipeline.

        use_preprocess : bool, default=True
            Whether to build and use the outer preprocessing pipeline.

        categorical_cols : list[str] or None, default=None
            Categorical columns passed to the inherited preprocessing builder when
            outer preprocessing is enabled.

        numeric_cols : list[str] or None, default=None
            Numeric columns passed to the inherited preprocessing builder when
            outer preprocessing is enabled.

        cat_encoder : str, default="ohe"
            Encoding strategy passed to the inherited preprocessing builder when
            outer preprocessing is enabled.

        Returns
        -------
        dict[str, Any]
            Full training summary dictionary containing:

            - fit summary returned by ``bagging_fit_engine()``
            - ``feature_names_len`` : number of tracked transformed feature names
            - ``evaluation`` : evaluation dictionary returned by
              ``model_evaluation_engine()``

        Raises
        ------
        ValueError
            If ``X_train`` or ``Y_train`` is unavailable before training.

        Notes
        -----
        This method assumes that train/test splitting has already been completed.

        After fitting, the model is immediately evaluated on both train and test
        data. For multi-output regression, the returned evaluation includes both
        per-target metrics and mean aggregated metrics.
        """
        if self.X_train is None or self.Y_train is None:
            raise ValueError("⚠️ Run train_test_split_engine() before training ‼️")

        fit_summary = self.bagging_fit_engine(
            estimator=estimator,
            n_estimators=n_estimators,
            use_cv=use_cv,
            cv_folds=cv_folds,
            scoring=scoring,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            param_grid=param_grid,
            random_state=random_state,
            extra_steps=extra_steps,
            use_preprocess=use_preprocess,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            cat_encoder=cat_encoder,
        )

        eval_results = self.model_evaluation_engine()

        return {
            **fit_summary,
            "feature_names_len": (
                None if self.feature_names is None else len(self.feature_names)
            ),
            "evaluation": eval_results,
        }

    # -------------------- Regression evaluation --------------------
    def model_evaluation_engine(self) -> Dict[str, Any]:
        """
        Evaluate the fitted bagging regressor on training and test datasets.

        This method generates predictions from the fitted model pipeline for both
        ``X_train`` and ``X_test``, then computes regression metrics.

        Evaluation Behavior
        -------------------
        Single-output regression:
            - R²
            - MAE
            - MSE
            - RMSE

        Multi-output regression:
            - per-target R²
            - per-target MAE
            - per-target MSE
            - per-target RMSE
            - mean of each metric across all target columns

        Returns
        -------
        dict[str, Any]
            Evaluation-result dictionary.

            For single-output regression, the returned dictionary contains:
            - ``train_r2``
            - ``test_r2``
            - ``train_mae``
            - ``test_mae``
            - ``train_mse``
            - ``test_mse``
            - ``train_rmse``
            - ``test_rmse``

            For multi-output regression, the returned dictionary contains:
            - mean metrics across targets
            - per-target metric dictionaries for train and test data

        Raises
        ------
        ValueError
            If ``self.model_pipeline`` is not fitted or missing.

        ValueError
            If train/test feature or target data is unavailable.

        Notes
        -----
        Predictions are stored in:
        - ``self.y_train_pred``
        - ``self.y_test_pred``

        For multi-output regression, this method assumes that predictions are
        returned as a 2-dimensional array whose column order matches
        ``self.Y_train.columns`` and ``self.Y_test.columns``.

        RMSE is computed by applying the square root to MSE.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Train the model before evaluation ‼️")

        if (
            self.X_train is None
            or self.X_test is None
            or self.Y_train is None
            or self.Y_test is None
        ):
            raise ValueError("⚠️ Train/test data is unavailable for evaluation ‼️")

        self.y_train_pred = self.model_pipeline.predict(self.X_train)
        self.y_test_pred = self.model_pipeline.predict(self.X_test)

        if self._is_multi_output(self.Y_test):
            train_r2_per_target = {
                col: float(r2_score(self.Y_train[col], self.y_train_pred[:, idx]))
                for idx, col in enumerate(self.Y_train.columns)
            }
            test_r2_per_target = {
                col: float(r2_score(self.Y_test[col], self.y_test_pred[:, idx]))
                for idx, col in enumerate(self.Y_test.columns)
            }

            train_mae_per_target = {
                col: float(
                    mean_absolute_error(self.Y_train[col], self.y_train_pred[:, idx])
                )
                for idx, col in enumerate(self.Y_train.columns)
            }
            test_mae_per_target = {
                col: float(
                    mean_absolute_error(self.Y_test[col], self.y_test_pred[:, idx])
                )
                for idx, col in enumerate(self.Y_test.columns)
            }

            train_mse_per_target = {
                col: float(
                    mean_squared_error(self.Y_train[col], self.y_train_pred[:, idx])
                )
                for idx, col in enumerate(self.Y_train.columns)
            }
            test_mse_per_target = {
                col: float(
                    mean_squared_error(self.Y_test[col], self.y_test_pred[:, idx])
                )
                for idx, col in enumerate(self.Y_test.columns)
            }

            train_rmse_per_target = {
                col: float(np.sqrt(train_mse_per_target[col]))
                for col in self.Y_train.columns
            }
            test_rmse_per_target = {
                col: float(np.sqrt(test_mse_per_target[col]))
                for col in self.Y_test.columns
            }

            return {
                "train_r2_mean": float(np.mean(list(train_r2_per_target.values()))),
                "test_r2_mean": float(np.mean(list(test_r2_per_target.values()))),
                "train_mae_mean": float(np.mean(list(train_mae_per_target.values()))),
                "test_mae_mean": float(np.mean(list(test_mae_per_target.values()))),
                "train_mse_mean": float(np.mean(list(train_mse_per_target.values()))),
                "test_mse_mean": float(np.mean(list(test_mse_per_target.values()))),
                "train_rmse_mean": float(np.mean(list(train_rmse_per_target.values()))),
                "test_rmse_mean": float(np.mean(list(test_rmse_per_target.values()))),
                "train_r2_per_target": train_r2_per_target,
                "test_r2_per_target": test_r2_per_target,
                "train_mae_per_target": train_mae_per_target,
                "test_mae_per_target": test_mae_per_target,
                "train_mse_per_target": train_mse_per_target,
                "test_mse_per_target": test_mse_per_target,
                "train_rmse_per_target": train_rmse_per_target,
                "test_rmse_per_target": test_rmse_per_target,
            }

        train_mse = float(mean_squared_error(self.Y_train, self.y_train_pred))
        test_mse = float(mean_squared_error(self.Y_test, self.y_test_pred))

        return {
            "train_r2": float(r2_score(self.Y_train, self.y_train_pred)),
            "test_r2": float(r2_score(self.Y_test, self.y_test_pred)),
            "train_mae": float(mean_absolute_error(self.Y_train, self.y_train_pred)),
            "test_mae": float(mean_absolute_error(self.Y_test, self.y_test_pred)),
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_rmse": float(np.sqrt(train_mse)),
            "test_rmse": float(np.sqrt(test_mse)),
        }


# =================================================
