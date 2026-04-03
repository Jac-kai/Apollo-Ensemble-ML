# -------------------- Import Modules --------------------
from __future__ import annotations

from typing import Any, Dict


# -------------------- Voting Param Grids --------------------
def build_voting_classifier_param_grid() -> Dict[str, Any]:
    return {
        "classifier__voting": ["hard", "soft"],
        "classifier__weights": [
            [1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1],
            [1, 2, 1, 1, 1],
            [1, 1, 2, 1, 1],
            [1, 1, 1, 2, 1],
            [1, 1, 1, 1, 2],
            [2, 2, 1, 1, 1],
            [2, 1, 2, 1, 1],
            [1, 2, 2, 1, 1],
        ],
    }


def build_voting_regressor_param_grid() -> Dict[str, Any]:
    return {
        "regressor__weights": [
            [1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1],
            [1, 2, 1, 1, 1],
            [1, 1, 2, 1, 1],
            [1, 1, 1, 2, 1],
            [1, 1, 1, 1, 2],
            [2, 2, 1, 1, 1],
            [2, 1, 2, 1, 1],
            [1, 2, 2, 1, 1],
        ],
    }


# -------------------- Bagging Param Grids --------------------
def build_bagging_classifier_param_grid() -> Dict[str, Any]:
    return {
        "classifier__n_estimators": [10, 30, 50, 100],
        "classifier__max_samples": [0.7, 1.0],
        "classifier__max_features": [0.7, 1.0],
        "classifier__bootstrap": [True],
        "classifier__bootstrap_features": [False],
    }


def build_bagging_regressor_param_grid() -> Dict[str, Any]:
    return {
        "regressor__n_estimators": [10, 30, 50, 100],
        "regressor__max_samples": [0.7, 1.0],
        "regressor__max_features": [0.7, 1.0],
        "regressor__bootstrap": [True],
        "regressor__bootstrap_features": [False],
    }


# -------------------- AdaBoost Param Grids --------------------
def build_adaboost_classifier_param_grid() -> Dict[str, Any]:
    return {
        "classifier__n_estimators": [30, 50, 100],
        "classifier__learning_rate": [0.01, 0.1, 1.0],
    }


def build_adaboost_regressor_param_grid() -> Dict[str, Any]:
    return {
        "regressor__n_estimators": [30, 50, 100],
        "regressor__learning_rate": [0.01, 0.1, 1.0],
    }


# -------------------- GradientBoosting Param Grids --------------------
def build_gradientboosting_classifier_param_grid() -> Dict[str, Any]:
    return {
        "classifier__n_estimators": [50, 100],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__subsample": [0.8, 1.0],
        "classifier__max_depth": [2, 3, 5],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }


def build_gradientboosting_regressor_param_grid() -> Dict[str, Any]:
    return {
        "regressor__n_estimators": [50, 100],
        "regressor__learning_rate": [0.01, 0.1, 0.2],
        "regressor__subsample": [0.8, 1.0],
        "regressor__max_depth": [2, 3, 5],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2],
    }


# -------------------- Stacking Param Grids --------------------
def build_stacking_classifier_param_grid() -> Dict[str, Any]:
    return {
        "classifier__passthrough": [False, True],
        "classifier__stack_method": ["auto"],
        "classifier__final_estimator__C": [0.1, 1.0, 10.0],
    }


def build_stacking_regressor_param_grid() -> Dict[str, Any]:
    return {
        "regressor__passthrough": [False, True],
        "regressor__final_estimator__fit_intercept": [True, False],
    }


def build_stacking_regressor_ridge_param_grid() -> Dict[str, Any]:
    return {
        "regressor__passthrough": [False, True],
        "regressor__final_estimator__alpha": [0.1, 1.0, 10.0, 100.0],
        "regressor__final_estimator__fit_intercept": [True, False],
    }


# -------------------- Helper: dispatcher --------------------
def get_param_grid(model_type: str) -> Dict[str, Any]:
    """
    Return the default parameter-grid dictionary for a supported ensemble grid type.

    This dispatcher converts a normalized internal grid-type name into the
    corresponding default parameter-grid builder function and returns the
    generated parameter-grid dictionary.

    Parameters
    ----------
    model_type : str
        Internal ensemble grid type name.

        Supported values include:

        - ``"voting_classifier"``
        - ``"voting_regressor"``
        - ``"bagging_classifier"``
        - ``"bagging_regressor"``
        - ``"adaboost_classifier"``
        - ``"adaboost_regressor"``
        - ``"gradientboosting_classifier"``
        - ``"gradientboosting_regressor"``
        - ``"stacking_classifier"``
        - ``"stacking_regressor"``
        - ``"stacking_regressor_ridge"``

        The input is normalized with ``lower().strip()`` before dispatch.

    Returns
    -------
    Dict[str, Any]
        Default GridSearchCV parameter-grid dictionary for the requested model
        type.

        Example return values may include keys such as:

        - ``"classifier__n_estimators"``
        - ``"classifier__weights"``
        - ``"regressor__learning_rate"``
        - ``"regressor__final_estimator__alpha"``

    Raises
    ------
    ValueError
        If ``model_type`` is not one of the supported internal grid-type names.

    Notes
    -----
    This helper does not decide which Apollo model should use which grid.
    Its responsibility is only to dispatch a validated internal grid-type name
    to the correct builder function.

    The returned dictionary is intended for direct use as the ``param_grid``
    argument of ``GridSearchCV``.

    Workflow
    --------
    1. Normalize the incoming ``model_type`` string.
    2. Resolve the corresponding grid-builder function from the local mapping.
    3. Call the matched builder.
    4. Return the resulting parameter-grid dictionary.
    """
    model_type = model_type.lower().strip()

    mapping = {
        "voting_classifier": build_voting_classifier_param_grid,
        "voting_regressor": build_voting_regressor_param_grid,
        "bagging_classifier": build_bagging_classifier_param_grid,
        "bagging_regressor": build_bagging_regressor_param_grid,
        "adaboost_classifier": build_adaboost_classifier_param_grid,
        "adaboost_regressor": build_adaboost_regressor_param_grid,
        "gradientboosting_classifier": build_gradientboosting_classifier_param_grid,
        "gradientboosting_regressor": build_gradientboosting_regressor_param_grid,
        "stacking_classifier": build_stacking_classifier_param_grid,
        "stacking_regressor": build_stacking_regressor_param_grid,
        "stacking_regressor_ridge": build_stacking_regressor_ridge_param_grid,
    }

    if model_type not in mapping:
        raise ValueError(f"⚠️ Unsupported model_type: {model_type} ‼️")

    return mapping[model_type]()


# -------------------- Helper: get default param grid by Apollo model name --------------------
def get_default_param_grid_for_model(
    model_name: str,
    final_estimator_name: str | None = None,
) -> Dict[str, Any] | None:
    """
    Return the default parameter grid for a registered Apollo ensemble model.

    This helper translates a public Apollo model name into the corresponding
    internal grid-type name, then delegates grid construction to
    ``get_param_grid(...)``.

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

    final_estimator_name : str or None, optional
        Optional final-estimator name used only for model families whose
        default parameter-grid selection depends on the chosen final estimator.

        This argument is currently relevant only for ``"StackingRegressor"``.

        Supported behavior:

        - if ``final_estimator_name == "ridge"``, use the ridge-specific
          stacking regressor grid
        - otherwise, use the default stacking regressor grid

    Returns
    -------
    Dict[str, Any] or None
        Default GridSearchCV parameter-grid dictionary for the requested Apollo
        model when a compatible mapping exists.

        Returns ``None`` if ``model_name`` is not supported by the internal
        mapping.

    Notes
    -----
    This helper is a bridge between:

    1. Apollo public model names used in menus and registries, and
    2. internal grid-type names used by ``get_param_grid(...)``.

    For most model families, the mapping is one-to-one.

    Special Case
    ------------
    ``StackingRegressor`` supports multiple final estimators, so the selected
    default parameter grid depends on ``final_estimator_name``:

    - ``"ridge"`` -> ``"stacking_regressor_ridge"``
    - any other value -> ``"stacking_regressor"``

    This helper does not validate whether the chosen final estimator is
    actually compatible with the downstream stacking workflow. It only selects
    the matching default grid when possible.

    Workflow
    --------
    1. Map the Apollo model name to an internal grid-type name.
    2. Apply the special branching rule for ``StackingRegressor`` when needed.
    3. Return ``None`` if no supported mapping exists.
    4. Otherwise call ``get_param_grid(model_type)`` and return its result.
    """
    mapping = {
        "VotingClassifier": "voting_classifier",
        "VotingRegressor": "voting_regressor",
        "BaggingClassifier": "bagging_classifier",
        "BaggingRegressor": "bagging_regressor",
        "AdaBoostClassifier": "adaboost_classifier",
        "AdaBoostRegressor": "adaboost_regressor",
        "GradientBoostingClassifier": "gradientboosting_classifier",
        "GradientBoostingRegressor": "gradientboosting_regressor",
        "StackingClassifier": "stacking_classifier",
        "StackingRegressor": (
            "stacking_regressor_ridge"
            if final_estimator_name == "ridge"
            else "stacking_regressor"
        ),
    }

    model_type = mapping.get(model_name)
    if model_type is None:
        return None

    return get_param_grid(model_type)


# =================================================
