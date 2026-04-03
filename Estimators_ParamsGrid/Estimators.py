# -------------------- Import Modules --------------------
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# -------------------- Classification Estimators --------------------
# DecisionTree
def build_dt_classifier(
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
):
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


# RandomForest
def build_rf_classifier(
    n_estimators: int = 100,
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | None = None,
    random_state: int = 42,
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )


# KNN
def build_knn_classifier(
    n_neighbors: int = 5,
    weights: str = "uniform",
    algorithm: str = "auto",
    p: int = 2,
):
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        p=p,
    )


# SVC
def build_svc_classifier(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    probability: bool = True,
    random_state: int = 42,
):
    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        random_state=random_state,
    )


# Logistic
def build_logistic_classifier(
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    class_weight: str | None = None,
    random_state: int = 42,
):
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )


# -------------------- Regression Estimators --------------------
# DecisionTree
def build_dt_regressor(
    criterion: str = "squared_error",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
):
    return DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


# RandomForest
def build_rf_regressor(
    n_estimators: int = 100,
    criterion: str = "squared_error",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


# KNN
def build_knn_regressor(
    n_neighbors: int = 5,
    weights: str = "uniform",
    algorithm: str = "auto",
    p: int = 2,
):
    return KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        p=p,
    )


# SVR
def build_svr_regressor(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    epsilon: float = 0.1,
):
    return SVR(
        C=C,
        kernel=kernel,
        gamma=gamma,
        epsilon=epsilon,
    )


# Linear Regression
def build_linear_regressor():
    return LinearRegression()


# Ridge (stacking)
def build_ridge_regressor(
    alpha: float = 1.0,
    fit_intercept: bool = True,
    random_state: int = 42,
):
    return Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        random_state=random_state,
    )


# -------------------- Classification estimator menu map --------------------
CLASSIFIER_ESTIMATOR_BUILDERS = {
    1: ("dt", build_dt_classifier),
    2: ("rf", build_rf_classifier),
    3: ("knn", build_knn_classifier),
    4: ("svc", build_svc_classifier),
    5: ("logistic", build_logistic_classifier),
}


# -------------------- Regression estimator menu map --------------------
REGRESSOR_ESTIMATOR_BUILDERS = {
    1: ("dt", build_dt_regressor),
    2: ("rf", build_rf_regressor),
    3: ("knn", build_knn_regressor),
    4: ("svr", build_svr_regressor),
    5: ("linear", build_linear_regressor),
    6: ("ridge", build_ridge_regressor),
}


# -------------------- Helper: get estimator builder map --------------------
def get_estimator_builder_map(task_type: str) -> dict[int, tuple[str, callable]]:
    """
    Return the estimator-builder menu mapping for a given Apollo task type.

    This helper normalizes the requested task type and returns the
    corresponding menu mapping used by estimator-selection workflows.

    The returned mapping is a dictionary whose:

    - key is the numeric menu option shown to the user
    - value is a tuple ``(estimator_name, builder_function)``

    The builder function can be called immediately to construct a default
    sklearn estimator instance.

    Parameters
    ----------
    task_type : str
        Task type used to select the compatible estimator-builder mapping.

        Supported values are:

        - ``"classifier"``
        - ``"regressor"``

        The input is normalized with ``lower().strip()`` before validation.

    Returns
    -------
    dict[int, tuple[str, callable]]
        Estimator-builder menu mapping for the requested task type.

        For ``"classifier"``, the returned mapping is
        ``CLASSIFIER_ESTIMATOR_BUILDERS``.

        For ``"regressor"``, the returned mapping is
        ``REGRESSOR_ESTIMATOR_BUILDERS``.

        Example return structure::

            {
                1: ("dt", build_dt_classifier),
                2: ("rf", build_rf_classifier),
                3: ("knn", build_knn_classifier),
            }

    Raises
    ------
    ValueError
        If ``task_type`` is not one of the supported values.

    Notes
    -----
    This helper does not build estimators by itself. It only returns the
    correct menu mapping so downstream helpers such as
    ``select_single_estimator(...)`` can:

    1. display estimator choices,
    2. read the selected numeric option,
    3. resolve the estimator alias and builder function,
    4. build the estimator instance.

    Workflow
    --------
    1. Normalize the input ``task_type``.
    2. If the task is classification, return
       ``CLASSIFIER_ESTIMATOR_BUILDERS``.
    3. If the task is regression, return
       ``REGRESSOR_ESTIMATOR_BUILDERS``.
    4. Otherwise raise ``ValueError``.

    Examples
    --------
    Request the classifier estimator map::

        builder_map = get_estimator_builder_map("classifier")

    Example result::

        {
            1: ("dt", build_dt_classifier),
            2: ("rf", build_rf_classifier),
            3: ("knn", build_knn_classifier),
            4: ("svc", build_svc_classifier),
            5: ("logistic", build_logistic_classifier),
        }

    Request the regressor estimator map::

        builder_map = get_estimator_builder_map("regressor")

    Example result::

        {
            1: ("dt", build_dt_regressor),
            2: ("rf", build_rf_regressor),
            3: ("knn", build_knn_regressor),
            4: ("svr", build_svr_regressor),
            5: ("linear", build_linear_regressor),
            6: ("ridge", build_ridge_regressor),
        }
    """
    task_type = task_type.lower().strip()

    if task_type == "classifier":
        return CLASSIFIER_ESTIMATOR_BUILDERS

    if task_type == "regressor":
        return REGRESSOR_ESTIMATOR_BUILDERS

    raise ValueError(f"⚠️ Unsupported task_type: {task_type} ‼️")


# =================================================
