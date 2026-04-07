# -------------------- Import Modules --------------------
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from Apollo.Backbone.Ensemble_BaseConfig import EnsembleBaseModelConfig, report_root


# -------------------- AdaBoost Missioner --------------------
class AdaBoost_Missioner(EnsembleBaseModelConfig):
    """
    AdaBoost_Missioner
    ==================

    Mission-layer base class for AdaBoost ensemble models.

    This class extends `EnsembleBaseModelConfig` and centralizes the shared
    AdaBoost workflow used by child model-layer classes. Its purpose is to
    separate AdaBoost-specific construction and fitting logic from higher-level
    task-specific training and evaluation code.

    Responsibilities
    ----------------
    This class provides reusable support for:

    - validating the user-supplied base estimator
    - validating AdaBoost hyperparameters
    - building `AdaBoostClassifier` or `AdaBoostRegressor`
    - fitting AdaBoost through the inherited `fit_with_grid()` workflow
    - supporting optional GridSearchCV tuning
    - supporting optional outer preprocessing pipelines
    - generating predictions from the fitted pipeline
    - generating class probabilities for classification tasks
    - exposing the fitted AdaBoost estimator from the outer pipeline
    - returning a compact training summary dictionary

    Supported Tasks
    ---------------
    Classification
        Uses `sklearn.ensemble.AdaBoostClassifier`.

    Regression
        Uses `sklearn.ensemble.AdaBoostRegressor`.

    Architecture Role
    -----------------
    This class belongs to the mission layer in the project architecture.

    Typical responsibility split:

    - base config layer
        Handles common dataset management, preprocessing construction,
        pipeline assembly, train/test split handling, and shared fitting logic.

    - mission layer
        Handles algorithm-family-specific workflow such as validation,
        estimator construction, prediction helpers, and training summaries.

    - model layer
        Defines task identity, evaluation logic, scoring strategy,
        and user-facing train methods.

    Design Notes
    ------------
    AdaBoost is a sequential boosting ensemble method that repeatedly trains a
    single weak learner across multiple boosting rounds. Compared with ensemble
    methods such as Voting or Stacking, AdaBoost does not combine multiple
    different estimators in parallel. Instead, it refines performance by
    iteratively reweighting observations and aggregating the sequence of weak
    learners.

    This class supports two pipeline styles:

    1. Outer preprocessing enabled
    The inherited base configuration builds a preprocessing pipeline and places
    AdaBoost as the final estimator step.

    2. Outer preprocessing disabled
    The provided base estimator may already include its own preprocessing logic,
    and AdaBoost is fit directly through the inherited workflow.

    Child Class Expectations
    ------------------------
    Child classes are expected to define or provide:

    - `task`
        Must be either `"classification"` or `"regression"`.

    - task-appropriate scoring behavior
        Especially if multi-output or custom scoring is needed.

    - evaluation logic
        Such as classification or regression performance reporting.

    Expected State from Parent Layer
    --------------------------------
    This class relies on inherited attributes and workflow state maintained by
    `EnsembleBaseModelConfig`, including commonly used attributes such as:

    - `self.model_pipeline`
    - `self.feature_names`
    - `self.X_test`
    - `self.step_name`

    Parameter Grid Convention
    -------------------------
    When `use_cv=True`, parameter grids should follow the outer pipeline step
    naming convention defined by the parent configuration layer. In practice, this
    usually means using prefixed parameter names such as:

    - `classifier__n_estimators`
    - `classifier__learning_rate`
    - `regressor__n_estimators`
    - `regressor__learning_rate`

    depending on the active task and final pipeline step naming.

    Typical Workflow
    ----------------
    A child model class typically uses this class in the following order:

    1. define the task type
    2. provide a valid base estimator
    3. call `adaboost_fit_engine(...)`
    4. run task-specific evaluation
    5. optionally call prediction helper methods

    Raises
    ------
    ValueError
        If task configuration is invalid, required estimators are missing,
        hyperparameters are invalid, or prediction is requested before fitting.

    AttributeError
        If probability prediction is requested from a fitted pipeline that does not
        expose `predict_proba`.

    KeyError
        If the expected final AdaBoost step is not found in the fitted pipeline.

    Notes
    -----
    This class is designed for reuse and consistency across AdaBoost-based model
    implementations. It allows child classes to stay compact while preserving a
    clear separation of concerns across the project architecture.
    """

    # -------------------- Validation for estimator --------------------
    def _validate_estimator(
        self,
        estimator: Any,
    ):
        """
        Validate base estimator for AdaBoost.

        Parameters
        ----------
        estimator : Any
            Base estimator used as the weak learner inside AdaBoost.

            This estimator must be sklearn-compatible and must match the current task:

            - classification task -> classifier-compatible estimator
            - regression task -> regressor-compatible estimator

            The estimator may be a plain sklearn estimator or a pipeline object.

        Returns
        -------
        Any
            The validated estimator unchanged.

        Raises
        ------
        ValueError
            If `estimator` is None.

        Notes
        -----
        AdaBoost requires a single base estimator which will be trained repeatedly
        across boosting rounds. This method only checks for presence and does not
        perform deep sklearn capability validation.
        """
        if estimator is None:
            raise ValueError("⚠️ estimator cannot be None ‼️")
        return estimator

    # -------------------- Validation for AdaBoost parameters --------------------
    def _validate_adaboost_params(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
    ):
        """
        Validate AdaBoost hyperparameters.

        Parameters
        ----------
        n_estimators : int, default=50
            Number of boosting stages to perform.

            This value must be an integer greater than or equal to 1. Larger values
            increase ensemble capacity but may also increase training time and risk
            overfitting depending on the base estimator and dataset.

        learning_rate : float, default=1.0
            Shrinkage factor applied to each estimator contribution.

            This value must be a positive number. Smaller values reduce the influence
            of each boosting stage and are often tuned jointly with `n_estimators`.

        Raises
        ------
        ValueError
            If `n_estimators` is not an integer greater than or equal to 1.

        ValueError
            If `learning_rate` is not a positive numeric value.

        Notes
        -----
        This method validates only the core AdaBoost hyperparameters shared by both
        classification and regression variants.
        """
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("⚠️ n_estimators must be an integer >= 1 ‼️")

        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("⚠️ learning_rate must be a positive number ‼️")

    # -------------------- Build AdaBoost estimator --------------------
    def _build_adaboost_model(
        self,
        estimator: Any,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: int = 42,
    ):
        """
        Build an AdaBoost estimator according to the current task type.

        Parameters
        ----------
        estimator : Any
            Base estimator used as the weak learner for AdaBoost.

            The estimator must be sklearn-compatible and appropriate for the current
            task. For example, a classifier should be used for classification and a
            regressor should be used for regression.

        n_estimators : int, default=50
            Number of boosting stages.

        learning_rate : float, default=1.0
            Shrinkage factor applied to each estimator contribution.

        random_state : int, default=42
            Random seed used when constructing the AdaBoost model.

        Returns
        -------
        Any
            A constructed AdaBoost model instance:

            - `AdaBoostClassifier` when `self.task == "classification"`
            - `AdaBoostRegressor` when `self.task == "regression"`

        Raises
        ------
        ValueError
            If the provided estimator is invalid.

        ValueError
            If AdaBoost hyperparameters are invalid.

        ValueError
            If `self.task` is not `"classification"` or `"regression"`.

        Notes
        -----
        This method performs estimator validation and AdaBoost parameter validation
        before creating the final sklearn AdaBoost object.
        """
        checked_estimator = self._validate_estimator(estimator)
        self._validate_adaboost_params(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )

        if self.task == "classification":
            model = AdaBoostClassifier(
                estimator=checked_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            return model

        if self.task == "regression":
            model = AdaBoostRegressor(
                estimator=checked_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            return model

        raise ValueError("⚠️ task must be 'classification' or 'regression' ‼️")

    # -------------------- AdaBoost fit engine --------------------
    def adaboost_fit_engine(
        self,
        estimator: Any,
        n_estimators: int = 50,
        use_cv: bool = False,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
        learning_rate: float = 1.0,
        param_grid: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
        use_preprocess: bool = True,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        cat_encoder: str = "ohe",
    ) -> Dict[str, Any]:
        """
        Build and fit an AdaBoost ensemble model through the shared training workflow.

        Parameters
        ----------
        estimator : Any
            Base estimator used as the weak learner inside AdaBoost.

            This estimator must be sklearn-compatible and compatible with the current
            task type. It may be a plain estimator or a pipeline.

        n_estimators : int, default=50
            Number of boosting stages.

        use_cv : bool, default=False
            Whether to enable GridSearchCV during fitting.

        cv_folds : int, default=5
            Number of folds used for cross-validation when `use_cv=True`.

        scoring : str, default="f1_weighted"
            sklearn scoring name used for model selection in single-output tasks.

            Child classes handling more specialized tasks may override scoring behavior
            through inherited training logic.

        learning_rate : float, default=1.0
            Shrinkage factor applied to each estimator contribution.

        param_grid : Optional[Dict[str, Any]], default=None
            Parameter grid passed to GridSearchCV.

            Grid keys should follow the outer pipeline naming convention defined by the
            parent base configuration class. Typical examples include:

            Classification
                - `{"classifier__n_estimators": [30, 50, 100]}`
                - `{"classifier__learning_rate": [0.01, 0.1, 1.0]}`

            Regression
                - `{"regressor__n_estimators": [30, 50, 100]}`
                - `{"regressor__learning_rate": [0.01, 0.1, 1.0]}`

        random_state : int, default=42
            Random seed used for AdaBoost construction and inherited CV workflows.

        extra_steps : Optional[List[Tuple[str, Any]]], default=None
            Optional pipeline steps inserted between preprocessing and the final
            AdaBoost estimator in the outer pipeline.

        use_preprocess : bool, default=True
            Whether to build and use the outer preprocessing pipeline supplied by the
            base configuration layer.

            - True  -> outer preprocessing is included
            - False -> the estimator may already contain its own preprocessing logic

        categorical_cols : Optional[List[str]], default=None
            Categorical columns passed to the inherited preprocessor builder when
            outer preprocessing is enabled.

        numeric_cols : Optional[List[str]], default=None
            Numeric columns passed to the inherited preprocessor builder when outer
            preprocessing is enabled.

        cat_encoder : str, default="ohe"
            Encoding strategy passed to the inherited preprocessor builder when outer
            preprocessing is enabled.

        Returns
        -------
        Dict[str, Any]
            A compact training summary dictionary containing metadata about the fitted
            AdaBoost workflow, typically including:

            - input model type
            - task
            - base estimator class name
            - n_estimators
            - learning_rate
            - CV usage
            - CV folds
            - scoring
            - preprocessing usage
            - best parameters
            - best score
            - feature names

        Notes
        -----
        This method does not directly perform evaluation. It is intended to handle
        AdaBoost construction and fitting, while child model-layer classes typically
        perform task-specific evaluation afterward.

        The fitted pipeline is stored through the inherited training workflow.
        """
        self.input_model_type = (
            "AdaBoostClassifier"
            if self.task == "classification"
            else "AdaBoostRegressor"
        )

        adaboost_model = self._build_adaboost_model(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

        best_params, best_score = self.fit_with_grid(
            base_model=adaboost_model,
            param_grid=param_grid,
            use_cv=use_cv,
            cv_folds=cv_folds,
            scoring=scoring,
            random_state=random_state,
            extra_steps=extra_steps,
            use_preprocess=use_preprocess,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            cat_encoder=cat_encoder,
        )

        return {
            "input_model_type": self.input_model_type,
            "task": self.task,
            "base_estimator": type(estimator).__name__,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "use_cv": use_cv,
            "cv_folds": cv_folds if use_cv else None,
            "scoring": scoring,
            "use_preprocess": use_preprocess,
            "best_params": best_params,
            "best_score": best_score,
            "feature_names": self.feature_names,
        }

    # -------------------- Predict --------------------
    def predict_engine(
        self,
        X_data: Optional[pd.DataFrame] = None,
        preview: int = 5,
    ):
        """
        Generate predictions using the fitted AdaBoost pipeline.

        Parameters
        ----------
        X_data : Optional[pd.DataFrame], default=None
            Input features used for prediction.

            If None, the method attempts to use `self.X_test`. This is convenient for
            standard evaluation workflows after train/test split has already been
            prepared.

        preview : int, default=5
            Number of leading predictions to store in `self.prediction_preview` for
            quick inspection.

        Returns
        -------
        Any
            Prediction output returned by the fitted pipeline.

            - For regression tasks, this is typically an array-like collection of
            predicted numeric values.
            - For classification tasks, predictions are returned after optional
            target-label decoding. If target label encoding was applied during
            training, the returned values are typically the original class labels
            rather than encoded class indices.

            In multi-output classification workflows, the returned object may be a
            pandas DataFrame containing decoded target columns.

        Raises
        ------
        ValueError
            If the model pipeline has not been trained yet.

        ValueError
            If `X_data` is None and `self.X_test` is unavailable.

        Notes
        -----
        This method stores a small preview of predictions in ``self.prediction_preview``
        when possible, which can be useful for quick inspection, logging, or debugging.

        If target-side label encoding was applied during training, predictions are
        passed through ``self._maybe_decode_predictions(...)`` before preview caching
        and return.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Train the model before prediction ‼️")

        if X_data is None:
            if self.X_test is None:
                raise ValueError("⚠️ No X_data provided and X_test is unavailable ‼️")
            X_data = self.X_test

        preds = self.model_pipeline.predict(X_data)
        preds = self._maybe_decode_predictions(preds)

        try:
            self.prediction_preview = preds[:preview]
        except Exception:
            self.prediction_preview = preds

        return preds

    # -------------------- Predict probability --------------------
    def predict_proba_engine(
        self,
        X_data: Optional[pd.DataFrame] = None,
    ):
        """
        Predict class probabilities using the fitted AdaBoost classification pipeline.

        Parameters
        ----------
        X_data : Optional[pd.DataFrame], default=None
            Input features used for probability prediction.

            If None, the method attempts to use `self.X_test`.

        Returns
        -------
        Any
            Class probability predictions from the fitted AdaBoost classifier.

            The returned object is typically a 2D array-like structure where each row
            corresponds to a sample and each column corresponds to a class.

        Raises
        ------
        ValueError
            If the current task is not classification.

        ValueError
            If the model pipeline has not been trained yet.

        ValueError
            If `X_data` is None and `self.X_test` is unavailable.

        AttributeError
            If the fitted pipeline does not support `predict_proba`.

        Notes
        -----
        This method is only valid for classification workflows. It should not be used
        for regression tasks.
        """
        if self.task != "classification":
            raise ValueError("⚠️ predict_proba_engine() is only for classification ‼️")

        if self.model_pipeline is None:
            raise ValueError("⚠️ Train the model before predict_proba ‼️")

        if X_data is None:
            if self.X_test is None:
                raise ValueError("⚠️ No X_data provided and X_test is unavailable ‼️")
            X_data = self.X_test

        if not hasattr(self.model_pipeline, "predict_proba"):
            raise AttributeError(
                "⚠️ This fitted AdaBoost pipeline does not support predict_proba ‼️"
            )

        return self.model_pipeline.predict_proba(X_data)

    # -------------------- Fitted AdaBoost estimator getter --------------------
    def get_fitted_adaboost_estimator(self):
        """
        Return the fitted final AdaBoost estimator from the outer pipeline.

        This method retrieves the final AdaBoost estimator step stored inside
        ``self.model_pipeline.named_steps`` using ``self.step_name``.

        It is useful when direct access to the fitted sklearn
        ``AdaBoostClassifier`` or ``AdaBoostRegressor`` object is needed after
        the outer pipeline has already been built and fitted.

        Returns
        -------
        Any
            The fitted final AdaBoost estimator stored in the pipeline.
            This is typically an ``AdaBoostClassifier`` or ``AdaBoostRegressor``.

        Raises
        ------
        ValueError
            If ``self.model_pipeline`` is not fitted or unavailable.

        KeyError
            If the expected final estimator step name does not exist in
            ``self.model_pipeline.named_steps``.

        Notes
        -----
        This method returns only the final AdaBoost estimator, not the full
        preprocessing-plus-model pipeline. Use ``self.model_pipeline`` directly
        when access to the full outer pipeline is needed.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ model_pipeline is not fitted yet ‼️")

        if self.step_name not in self.model_pipeline.named_steps:
            raise KeyError(f"⚠️ Step '{self.step_name}' not found in model_pipeline ‼️")

        return self.model_pipeline.named_steps[self.step_name]

    # -------------------- Feature importance engine --------------------
    def feature_importance_engine(
        self,
        max_display: int = 20,
        save_fig: bool = False,
        folder_name: str = "Feature_Importance",
        file_name: str | None = None,
    ):
        """
        Compute and display feature importance for a fitted AdaBoost estimator.

        This method retrieves the fitted final AdaBoost estimator from the outer
        pipeline, extracts feature-importance values when supported, aligns them
        with the current feature names, builds a pandas DataFrame, plots the most
        important features as a horizontal bar chart, and optionally saves the
        figure to disk.

        Parameters
        ----------
        max_display : int, default=20
            Maximum number of top-ranked features displayed in the plot.

        save_fig : bool, default=False
            Whether to save the generated figure to disk.

        folder_name : str, default="Feature_Importance"
            Folder created under ``report_root`` when saving the figure.

        file_name : str or None, optional
            Custom output filename for the saved figure. If ``None``, an automatic
            timestamp-based filename is generated.

        Returns
        -------
        pandas.DataFrame
            Feature-importance table containing:

            - ``feature``
            - ``importance_mean``

            or equivalent importance columns depending on implementation details.

            The table is sorted by descending feature importance.

        Raises
        ------
        ValueError
            If no fitted AdaBoost estimator is available.

        ValueError
            If the fitted estimator does not expose ``feature_importances_``.

        ValueError
            If the importance vector length does not match the number of feature
            names.

        Notes
        -----
        In sklearn AdaBoost workflows, feature importance is available only when
        the fitted ensemble and its weak learners support feature-importance
        extraction.

        The returned DataFrame contains the full importance table, while the plot
        only shows the top ``max_display`` features.
        """
        fitted_model = self.get_fitted_adaboost_estimator()

        if fitted_model is None:
            raise ValueError("⚠️ No fitted AdaBoost estimator available ‼️")

        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else self.cleaned_X_data.columns.tolist()
        )
        n_features = len(feature_names)

        full_importances = []

        # ---------- Case 1: AdaBoost model itself exposes feature_importances_ ----------
        if hasattr(fitted_model, "feature_importances_"):
            importances = fitted_model.feature_importances_

            if len(importances) != n_features:
                raise ValueError(
                    f"⚠️ feature_importances_ length mismatch: "
                    f"{len(importances)} vs feature_names {n_features} ‼️"
                )

            importance_mean = np.asarray(importances, dtype=float)
            importance_std = np.zeros_like(importance_mean, dtype=float)

        # ---------- Case 2: aggregate sub-estimators ----------
        elif hasattr(fitted_model, "estimators_"):
            estimators = fitted_model.estimators_

            # sklearn AdaBoost usually stores per-estimator selected feature indices here
            estimators_features = getattr(fitted_model, "estimators_features_", None)

            for idx, est in enumerate(estimators):
                if not hasattr(est, "feature_importances_"):
                    continue

                local_imp = np.asarray(est.feature_importances_, dtype=float)
                full_imp = np.zeros(n_features, dtype=float)

                if estimators_features is not None:
                    feat_idx = np.asarray(estimators_features[idx])

                    if len(local_imp) != len(feat_idx):
                        raise ValueError(
                            f"⚠️ estimator {idx} importance length "
                            f"{len(local_imp)} does not match selected feature count "
                            f"{len(feat_idx)} ‼️"
                        )

                    full_imp[feat_idx] = local_imp

                else:
                    # fallback: only safe when local importance length already matches full feature count
                    if len(local_imp) != n_features:
                        raise ValueError(
                            "⚠️ Cannot align AdaBoost feature importances because "
                            "estimators_features_ is unavailable and local importance "
                            "length does not match full feature count ‼️"
                        )
                    full_imp = local_imp

                full_importances.append(full_imp)

            if not full_importances:
                raise ValueError(
                    "⚠️ Current AdaBoost estimator does not support feature importance ‼️"
                )

            full_importances = np.vstack(full_importances)
            importance_mean = np.mean(full_importances, axis=0)
            importance_std = np.std(full_importances, axis=0)

        else:
            raise ValueError(
                "⚠️ Current AdaBoost estimator does not support feature importance ‼️"
            )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": importance_mean,
                "importance_std": importance_std,
            }
        ).sort_values("importance_mean", ascending=False)

        display_df = importance_df.head(max_display).iloc[::-1]

        plt.figure(figsize=(10, max(5, len(display_df) * 0.35)))
        plt.barh(display_df["feature"], display_df["importance_mean"])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"{self.__class__.__name__} Feature Importance")
        plt.tight_layout()

        if save_fig:
            save_folder = os.path.join(report_root, folder_name)
            os.makedirs(save_folder, exist_ok=True)

            if file_name is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_name = (
                    f"{self.__class__.__name__}_feature_importance_{timestamp}.png"
                )

            save_path = os.path.join(save_folder, file_name)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"🔥 Feature importance figure saved: {save_path}")

        plt.show()
        return importance_df


# =================================================
