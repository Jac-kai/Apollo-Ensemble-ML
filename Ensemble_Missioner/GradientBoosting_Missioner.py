# -------------------- Import Modules --------------------
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import plot_tree

from Apollo.Backbone.Ensemble_BaseConfig import EnsembleBaseModelConfig, report_root


# -------------------- GradientBoosting Missioner --------------------
class GradientBoosting_Missioner(EnsembleBaseModelConfig):
    """
    GradientBoosting_Missioner
    ==========================

    Mission-layer base class for Gradient Boosting ensemble models.

    This class extends `EnsembleBaseModelConfig` and centralizes the shared
    workflow for sklearn Gradient Boosting estimators so that child model-layer
    classes can focus on task-specific concerns such as evaluation logic,
    task declaration, and scoring customization.

    Responsibilities
    ----------------
    This class provides reusable support for:

    - validating core Gradient Boosting hyperparameters
    - building `GradientBoostingClassifier` or `GradientBoostingRegressor`
    - fitting models through the inherited `fit_with_grid()` workflow
    - supporting optional GridSearchCV tuning
    - supporting optional outer preprocessing pipelines
    - generating predictions from the fitted pipeline
    - generating class probabilities for classification tasks
    - exposing the fitted final Gradient Boosting estimator from the pipeline
    - returning a compact training summary dictionary

    Supported Tasks
    ---------------
    Classification
        Uses `sklearn.ensemble.GradientBoostingClassifier`.

    Regression
        Uses `sklearn.ensemble.GradientBoostingRegressor`.

    Architecture Role
    -----------------
    This class belongs to the mission layer in the project architecture.

    A typical responsibility split is:

    - base config layer
        Handles shared data state, preprocessing construction, pipeline assembly,
        train/test split handling, and generic fitting / GridSearchCV logic.

    - mission layer
        Handles algorithm-family-specific model construction, validation,
        prediction helpers, and standardized training summaries.

    - model layer
        Defines the task identity, evaluation strategy, user-facing train methods,
        and any custom reporting logic.

    Design Notes
    ------------
    Gradient Boosting in sklearn is a tree-based boosting method that builds an
    ensemble sequentially, where each new stage attempts to improve the errors
    made by previous stages.

    Unlike AdaBoost, this interface does not require a user-supplied base
    estimator. Instead, model behavior is controlled directly through Gradient
    Boosting hyperparameters such as `n_estimators`, `learning_rate`,
    `subsample`, and tree-structure settings.

    This class supports two pipeline styles:

    1. Outer preprocessing enabled
    The inherited base configuration builds a preprocessing pipeline and places
    the Gradient Boosting estimator as the final step.

    2. Outer preprocessing disabled
    The estimator is fit directly through the inherited workflow without an
    outer preprocessing step.

    Child Class Expectations
    ------------------------
    Child classes are expected to define or provide:

    - `task`
        Must be either `"classification"` or `"regression"`.

    - task-specific scoring behavior
        Especially when custom or multi-output scoring is needed.

    - evaluation logic
        Such as classification metrics or regression metrics.

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
    naming convention defined by the parent configuration layer. In practice,
    this usually means using prefixed parameter names such as:

    - `classifier__n_estimators`
    - `classifier__learning_rate`
    - `classifier__subsample`
    - `classifier__max_depth`
    - `regressor__n_estimators`
    - `regressor__learning_rate`
    - `regressor__subsample`
    - `regressor__max_depth`

    depending on the active task and final estimator step name.

    Typical Workflow
    ----------------
    A child model class typically uses this class in the following order:

    1. define the task type
    2. call `gradient_boosting_fit_engine(...)`
    3. run task-specific evaluation
    4. optionally call prediction helper methods

    Raises
    ------
    ValueError
        If task configuration is invalid, hyperparameters are invalid, or
        prediction is requested before fitting.

    AttributeError
        If probability prediction is requested from a fitted pipeline that does
        not expose `predict_proba`.

    KeyError
        If the expected final Gradient Boosting step is not found in the fitted
        pipeline.

    Notes
    -----
    This class is designed to keep Gradient Boosting workflows consistent across
    child implementations while preserving a clear separation of concerns across
    the project architecture.
    """

    # -------------------- Validation for GradientBoosting parameters --------------------
    def _validate_gradient_boosting_params(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        max_depth: int = 3,
    ):
        """
        Validate core Gradient Boosting hyperparameters.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting stages.

            This value must be an integer greater than or equal to 1. Increasing this
            value may improve model capacity, but it also increases training time and
            may raise overfitting risk depending on the dataset and other parameters.

        learning_rate : float, default=0.1
            Shrinkage factor applied to the contribution of each boosting stage.

            Smaller values reduce the impact of each stage and are often tuned jointly
            with `n_estimators`.

        subsample : float, default=1.0
            Fraction of samples used for fitting each individual boosting stage.

            This value must be within the interval `(0, 1]`. Values below 1.0 enable
            stochastic gradient boosting behavior.

        max_depth : int, default=3
            Maximum depth of the individual regression trees used as weak learners.

            This value must be an integer greater than or equal to 1.

        Raises
        ------
        ValueError
            If `n_estimators` is not an integer greater than or equal to 1.

        ValueError
            If `learning_rate` is not a positive numeric value.

        ValueError
            If `subsample` is not within `(0, 1]`.

        ValueError
            If `max_depth` is not an integer greater than or equal to 1.

        Notes
        -----
        This method validates only a core subset of Gradient Boosting parameters that
        are shared across the classifier and regressor variants in this implementation.
        """
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("⚠️ n_estimators must be an integer >= 1 ‼️")

        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("⚠️ learning_rate must be a positive number ‼️")

        if not isinstance(subsample, (int, float)) or not (0 < subsample <= 1):
            raise ValueError("⚠️ subsample must be in (0, 1] ‼️")

        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("⚠️ max_depth must be an integer >= 1 ‼️")

    # -------------------- Build GradientBoosting estimator --------------------
    def _build_gradient_boosting_model(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        criterion: str = "friedman_mse",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        """
        Build a Gradient Boosting estimator according to the current task type.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting stages.

        learning_rate : float, default=0.1
            Shrinkage factor applied to the contribution of each boosting stage.

        subsample : float, default=1.0
            Fraction of training samples used for each boosting stage.

        criterion : str, default="friedman_mse"
            Split quality criterion used by the internal regression trees.

            The accepted value depends on the sklearn estimator variant and sklearn
            version in use.

        min_samples_split : int, default=2
            Minimum number of samples required to split an internal tree node.

        min_samples_leaf : int, default=1
            Minimum number of samples required to be at a leaf node.

        max_depth : int, default=3
            Maximum depth of the individual regression trees.

        random_state : int, default=42
            Random seed used when constructing the Gradient Boosting model.

        Returns
        -------
        Any
            A constructed Gradient Boosting model instance:

            - `GradientBoostingClassifier` when `self.task == "classification"`
            - `GradientBoostingRegressor` when `self.task == "regression"`

        Raises
        ------
        ValueError
            If core Gradient Boosting hyperparameters are invalid.

        ValueError
            If `self.task` is not `"classification"` or `"regression"`.

        Notes
        -----
        This method validates the core boosting parameters before constructing the
        final sklearn Gradient Boosting object.
        """
        self._validate_gradient_boosting_params(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            max_depth=max_depth,
        )

        if self.task == "classification":
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                subsample=subsample,
                criterion=criterion,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                random_state=random_state,
            )
            return model

        if self.task == "regression":
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                subsample=subsample,
                criterion=criterion,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                random_state=random_state,
            )
            return model

        raise ValueError("⚠️ task must be 'classification' or 'regression' ‼️")

    # -------------------- GradientBoosting fit engine --------------------
    def gradient_boosting_fit_engine(
        self,
        n_estimators: int = 100,
        use_cv: bool = False,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        criterion: str = "friedman_mse",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 3,
        param_grid: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
        use_preprocess: bool = True,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        cat_encoder: str = "ohe",
    ) -> Dict[str, Any]:
        """
        Build and fit a Gradient Boosting ensemble model through the shared training
        workflow.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting stages.

        use_cv : bool, default=False
            Whether to enable GridSearchCV during fitting.

        cv_folds : int, default=5
            Number of folds used for cross-validation when `use_cv=True`.

        scoring : str, default="f1_weighted"
            sklearn scoring name used for model selection in single-output tasks.

            Child classes with specialized tasks may override or extend scoring
            behavior through inherited training logic.

        learning_rate : float, default=0.1
            Shrinkage factor applied to the contribution of each boosting stage.

        subsample : float, default=1.0
            Fraction of training samples used for each boosting stage.

        criterion : str, default="friedman_mse"
            Split quality criterion used by the internal regression trees.

        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.

        min_samples_leaf : int, default=1
            Minimum number of samples required to remain in a leaf node.

        max_depth : int, default=3
            Maximum depth of the individual regression trees.

        param_grid : Optional[Dict[str, Any]], default=None
            Parameter grid passed to GridSearchCV.

            Grid keys should follow the outer pipeline naming convention defined by
            the parent base configuration class. Typical examples include:

            Classification
                - `{"classifier__n_estimators": [100, 200]}`
                - `{"classifier__learning_rate": [0.01, 0.1, 0.2]}`
                - `{"classifier__subsample": [0.8, 1.0]}`
                - `{"classifier__max_depth": [2, 3, 5]}`

            Regression
                - `{"regressor__n_estimators": [100, 200]}`
                - `{"regressor__learning_rate": [0.01, 0.1, 0.2]}`
                - `{"regressor__subsample": [0.8, 1.0]}`
                - `{"regressor__max_depth": [2, 3, 5]}`

        random_state : int, default=42
            Random seed used for model construction and inherited CV workflows.

        extra_steps : Optional[List[Tuple[str, Any]]], default=None
            Optional pipeline steps inserted between preprocessing and the final
            Gradient Boosting estimator in the outer pipeline.

        use_preprocess : bool, default=True
            Whether to build and use the outer preprocessing pipeline supplied by the
            base configuration layer.

            - True  -> outer preprocessing is included
            - False -> fitting is performed without outer preprocessing

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
            Gradient Boosting workflow, typically including:

            - input model type
            - task
            - n_estimators
            - learning_rate
            - subsample
            - criterion
            - min_samples_split
            - min_samples_leaf
            - max_depth
            - CV usage
            - CV folds
            - scoring
            - preprocessing usage
            - best parameters
            - best score
            - feature names

        Notes
        -----
        This method handles model construction and fitting but does not directly
        perform task-specific evaluation. Evaluation is typically implemented in child
        model-layer classes after training is completed.

        The fitted pipeline is stored through the inherited training workflow.
        """
        self.input_model_type = (
            "GradientBoostingClassifier"
            if self.task == "classification"
            else "GradientBoostingRegressor"
        )

        gb_model = self._build_gradient_boosting_model(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=random_state,
        )

        best_params, best_score = self.fit_with_grid(
            base_model=gb_model,
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
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "criterion": criterion,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
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
        Generate predictions using the fitted Gradient Boosting pipeline.

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
            Predicted values returned by the fitted pipeline.

            For classification tasks, this is typically an array-like collection of
            predicted class labels.
            For regression tasks, this is typically an array-like collection of
            predicted numeric values.

        Raises
        ------
        ValueError
            If the model pipeline has not been trained yet.

        ValueError
            If `X_data` is None and `self.X_test` is unavailable.

        Notes
        -----
        This method stores a small preview of predictions in `self.prediction_preview`
        when possible, which can be useful for logging, inspection, or debugging.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ Train the model before prediction ‼️")

        if X_data is None:
            if self.X_test is None:
                raise ValueError("⚠️ No X_data provided and X_test is unavailable ‼️")
            X_data = self.X_test

        preds = self.model_pipeline.predict(X_data)

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
        Predict class probabilities using the fitted Gradient Boosting classification
        pipeline.

        Parameters
        ----------
        X_data : Optional[pd.DataFrame], default=None
            Input features used for probability prediction.

            If None, the method attempts to use `self.X_test`.

        Returns
        -------
        Any
            Class probability predictions from the fitted Gradient Boosting classifier.

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
        This method is only valid for classification workflows and should not be used
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
                "⚠️ This fitted Gradient Boosting pipeline does not support predict_proba ‼️"
            )

        return self.model_pipeline.predict_proba(X_data)

    # -------------------- Fitted GradientBoosting estimator getter --------------------
    def get_fitted_gradient_boosting_estimator(self):
        """
        Return the fitted final Gradient Boosting estimator from the outer model
        pipeline.

        Returns
        -------
        Any
            The fitted Gradient Boosting estimator stored at the final pipeline step
            identified by `self.step_name`.

        Raises
        ------
        ValueError
            If `self.model_pipeline` has not been fitted yet.

        KeyError
            If the expected step name is not present in
            `self.model_pipeline.named_steps`.

        Notes
        -----
        This method is useful when direct access to the fitted Gradient Boosting
        estimator is needed, for example to inspect model-specific attributes after
        training.

        The returned object is the fitted final estimator inside the outer pipeline,
        not the full pipeline itself.
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
        Compute and display feature importance for a fitted Gradient Boosting estimator.

        This method retrieves the fitted final Gradient Boosting estimator from the
        outer pipeline, extracts feature-importance values from the estimator,
        aligns them with the current feature names, builds a pandas DataFrame,
        plots the most important features as a horizontal bar chart, and optionally
        saves the figure to disk.

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
            - ``importance_std``

            The table is sorted by descending feature importance.

        Raises
        ------
        ValueError
            If no fitted Gradient Boosting estimator is available.

        ValueError
            If the fitted estimator does not expose ``feature_importances_``.

        ValueError
            If the length of ``feature_importances_`` does not match the number of
            feature names.

        Notes
        -----
        In sklearn Gradient Boosting models, feature importance is typically exposed
        directly through the fitted estimator attribute ``feature_importances_``.

        Unlike bagging-based workflows, this method does not usually need to
        reconstruct feature importances across multiple independently sampled
        sub-estimators.

        The returned DataFrame contains the full feature-importance table, while
        the plot only shows the top ``max_display`` features.
        """
        fitted_model = self.get_fitted_gradient_boosting_estimator()

        if fitted_model is None:
            raise ValueError("⚠️ No fitted Bagging estimator available ‼️")

        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else self.cleaned_X_data.columns.tolist()
        )
        n_features = len(feature_names)

        full_importances = []

        # ---------- Case 1: bagging model itself exposes feature_importances_ ----------
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

            # sklearn bagging usually stores per-estimator selected feature indices here
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
                            "⚠️ Cannot align Bagging feature importances because "
                            "estimators_features_ is unavailable and local importance "
                            "length does not match full feature count ‼️"
                        )
                    full_imp = local_imp

                full_importances.append(full_imp)

            if not full_importances:
                raise ValueError(
                    "⚠️ Current Bagging estimator does not support feature importance ‼️"
                )

            full_importances = np.vstack(full_importances)
            importance_mean = np.mean(full_importances, axis=0)
            importance_std = np.std(full_importances, axis=0)

        else:
            raise ValueError(
                "⚠️ Current Bagging estimator does not support feature importance ‼️"
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

    # -------------------- Tree plot engine --------------------
    def tree_plot_engine(
        self,
        tree_indices: int | list[int] | tuple[int, ...] = 0,
        max_depth: int | None = 3,
        save_fig: bool = False,
        folder_name: str = "Tree_Plot",
        file_name: str | None = None,
    ):
        """
        Plot one or more fitted stage trees from a Gradient Boosting model.

        This method retrieves the fitted final Gradient Boosting estimator from the
        outer pipeline, validates the requested stage indices, and visualizes the
        selected boosting-stage trees one by one using ``sklearn.tree.plot_tree``.

        In sklearn Gradient Boosting models, the fitted weak learners are stored in
        ``estimators_``. For the workflows used here, each selected index typically
        corresponds to one boosting stage tree. When multiple indices are provided,
        each selected stage tree is plotted in sequence as an individual figure.

        Parameters
        ----------
        tree_indices : int or list[int] or tuple[int, ...], default=0
            Index or collection of indices identifying which fitted boosting-stage
            trees should be plotted.

            Examples:
            - ``0`` for a single boosting stage
            - ``[0, 1, 3]`` for multiple stages
            - ``(2, 4, 6)`` for multiple stages

        max_depth : int or None, default=3
            Maximum tree depth shown in the plotted visualization.

            - If an integer is provided, the displayed tree is truncated to that depth.
            - If ``None``, the full tree structure is displayed.

        save_fig : bool, default=False
            Whether each plotted tree figure should also be saved to disk.

        folder_name : str, default="Tree_Plot"
            Folder name created under ``report_root`` when ``save_fig=True``.

        file_name : str or None, default=None
            Optional custom output filename prefix used when saving figures.

            - If ``None``, each saved figure receives an automatically generated
            filename based on the class name, tree index, and timestamp.
            - If provided, the tree index is appended so that multiple selected
            stage trees are saved as separate files.

        Returns
        -------
        None
            The method displays one or more matplotlib figures and optionally saves
            them to disk.

        Raises
        ------
        ValueError
            If no fitted Gradient Boosting estimator is available.

        ValueError
            If the fitted estimator does not expose ``estimators_``.

        ValueError
            If ``tree_indices`` is empty after normalization.

        ValueError
            If any requested tree index is outside the valid boosting-stage range.

        Notes
        -----
        - This method visualizes one boosting-stage tree per figure.
        - Each selected stage tree is plotted separately rather than combined into
        subplots.
        - The current implementation assumes the sklearn ``estimators_`` structure
        used by the fitted Gradient Boosting estimator in this project workflow.
        - When ``save_fig=True``, files are saved under
        ``os.path.join(report_root, folder_name)``.
        - The figure title includes both the class name and the selected
        ``tree_index``.
        - Figures are closed with ``plt.close()`` after display to reduce plot
        accumulation during repeated interactive use.
        """
        fitted_model = self.get_fitted_gradient_boosting_estimator()

        if fitted_model is None:
            raise ValueError("⚠️ No fitted Gradient Boosting estimator available ‼️")

        if not hasattr(fitted_model, "estimators_"):
            raise ValueError("⚠️ Current estimator does not expose estimators_ ‼️")

        estimators_array = fitted_model.estimators_

        if isinstance(tree_indices, int):
            tree_indices = [tree_indices]
        else:
            tree_indices = list(tree_indices)

        if not tree_indices:
            raise ValueError("⚠️ tree_indices cannot be empty ‼️")

        total_stages = estimators_array.shape[0]

        for idx in tree_indices:
            if idx < 0 or idx >= total_stages:
                raise ValueError(
                    f"⚠️ tree_index {idx} is out of range (0 ~ {total_stages - 1}) ‼️"
                )

            tree_model = estimators_array[idx, 0]

            plt.figure(figsize=(18, 10))
            plot_tree(
                tree_model,
                feature_names=self.feature_names,
                filled=True,
                max_depth=max_depth,
                fontsize=8,
            )
            plt.title(f"{self.__class__.__name__} Tree Plot (tree_index={idx})")
            plt.tight_layout()

            if save_fig:
                save_folder = os.path.join(report_root, folder_name)
                os.makedirs(save_folder, exist_ok=True)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                if file_name is None:
                    out_name = f"{self.__class__.__name__}_tree_{idx}_{timestamp}.png"
                else:
                    stem, ext = os.path.splitext(file_name)
                    ext = ext or ".png"
                    out_name = f"{stem}_tree_{idx}{ext}"

                save_path = os.path.join(save_folder, out_name)
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"🔥 Tree plot figure saved: {save_path}")

            plt.show()
            plt.close()


# =================================================
