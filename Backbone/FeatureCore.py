# -------------------- Import Modules --------------------
import pandas as pd


# -------------------- Feature Core --------------------
class FeatureCore:
    """
    Core class for feature and target preparation in machine-learning workflows.

    `FeatureCore` is responsible for converting a tabular dataset into model-ready
    feature and target objects. It stores the source dataset, tracks the currently
    selected target column(s) and feature columns, validates user selections, and
    builds `X` and `y` for downstream machine-learning tasks.

    This class supports both single-target and multi-target workflows:

    - single target input, such as ``"Outcome"``
    - multiple target input, such as ``["Outcome", "RiskScore"]``

    Internally, selected target columns are normalized and stored as a list of
    column names. As a result, the generated `y` object is consistently returned
    as a pandas DataFrame, even when only one target column is selected. This
    design simplifies downstream handling for workflows that may support both
    single-output and multi-output learning.

    Attributes
    ----------
    target_data : pandas.DataFrame
        Source dataset used for feature and target selection.

    target_column : list[str] or None
        Selected target column names. The value is always stored as a list after
        validation, even when the original input contains only one column name.

    feature_columns : list[str] or None
        Selected feature column names used to construct `X`.

    X : pandas.DataFrame or None
        Feature matrix built from `feature_columns`.

    y : pandas.DataFrame or None
        Target matrix built from `target_column`. A DataFrame is returned for both
        single-target and multi-target cases.

    Notes
    -----
    This class only handles feature/target preparation and validation. Whether the
    resulting `y` is acceptable for model training depends on whether the chosen
    model supports single-output or multi-output targets.
    """

    # -------------------- Initialization --------------------
    def __init__(self, target_data: pd.DataFrame):
        """
        Initialize the feature-preparation core with a source dataset.

        Parameters
        ----------
        target_data : pandas.DataFrame
            Input dataset from which target and feature columns will be selected.

        Notes
        -----
        The provided dataset is stored directly in `self.target_data`. Selection state
        such as target columns, feature columns, `X`, and `y` is initialized as `None`
        until explicitly set or built through the class methods.
        """
        self.target_data = target_data
        self.target_column = None  # list[str] | None
        self.feature_columns = None  # list[str] | None
        self.X = None  # pd.DataFrame | None
        self.y = None  # pd.DataFrame | None

    # -------------------- Helper: Validation --------------------
    def _validation(self) -> bool:
        """
        Validate the current source dataset before feature/target operations.

        This helper checks whether `self.target_data` exists, is a pandas DataFrame,
        and is not empty.

        Returns
        -------
        bool
            `True` if the stored dataset is valid for feature/target processing,
            otherwise `False`.

        Notes
        -----
        This method is intended for internal use before operations such as setting
        target columns, setting feature columns, or building `X` and `y`.
        """
        if self.target_data is None:
            print("⚠️ No target data available ‼️")
            return False

        if not isinstance(self.target_data, pd.DataFrame):
            print("⚠️ Target data must be a pandas DataFrame ‼️")
            return False

        if self.target_data.empty:
            print("⚠️ Target data is empty ‼️")
            return False

        return True

    # -------------------- Reset feature state --------------------
    def reset_feature_state(self):
        """
        Reset the current feature-selection state.

        This method clears all previously stored feature/target selection results,
        including selected target columns, selected feature columns, and the built
        `X` and `y` objects.

        Returns
        -------
        None

        Notes
        -----
        This method does not modify the original dataset stored in `self.target_data`.
        It only resets the selection state so the user can perform feature/target
        selection again from a clean state.
        """
        self.target_column = None
        self.feature_columns = None
        self.X = None
        self.y = None
        print("☢️ Feature state has been reset ☢️")

    # -------------------- Set target column(s) --------------------
    def set_target_column(self, target_column: str | list[str]):
        """
        Set one or multiple target columns and build `y`.

        This method accepts either a single target column name or a list of target
        column names. The input is normalized into a cleaned list of unique column
        names, validated against the source dataset, and then stored in
        `self.target_column`.

        After successful validation, the method builds `self.y` from the selected
        target columns. The returned `y` is always a pandas DataFrame, including
        the case where only one target column is selected.

        Parameters
        ----------
        target_column : str or list[str]
            Column name or list of column names to use as target variables.

        Returns
        -------
        pandas.DataFrame or None
            Target DataFrame built from the selected target column(s) if successful;
            otherwise `None`.

        Validation Rules
        ----------------
        - `target_data` must exist and be a non-empty pandas DataFrame.
        - `target_column` must be either a string or a list of strings.
        - Empty strings are removed during cleaning.
        - At least one valid target column must remain after cleaning.
        - Duplicate target columns are not allowed.
        - All target columns must exist in the source dataset.

        Examples
        --------
        Single target input:

        >>> feature_core.set_target_column("Outcome")

        Multiple target input:

        >>> feature_core.set_target_column(["Outcome", "RiskScore"])

        Notes
        -----
        Even when only one target column is selected, `self.target_column` is stored
        as a list and `self.y` is returned as a DataFrame for consistency across
        single-target and multi-target workflows.
        """
        if not self._validation():
            return None

        # ---------- Normalize input to list[str] / Record target columns as cleaned target columns ----------
        if isinstance(target_column, str):
            cleaned_target_columns = (
                [target_column.strip()] if target_column.strip() else []
            )
        elif isinstance(target_column, list):
            if not all(
                isinstance(col, str) for col in target_column
            ):  # If target column is not string
                print("⚠️ All target column names must be strings ‼️")
                return None
            cleaned_target_columns = [
                col.strip() for col in target_column if col.strip()
            ]
        else:
            print("⚠️ Target column must be a string or list of strings ‼️")
            return None

        # ---------- Validate cleaned target columns ----------
        if not cleaned_target_columns:
            print("⚠️ Target column(s) must contain at least one valid column name ‼️")
            return None

        if len(cleaned_target_columns) != len(set(cleaned_target_columns)):
            print("⚠️ Target columns contain duplicates ‼️")
            return None

        invalid_columns = [
            col
            for col in cleaned_target_columns
            if col not in self.target_data.columns  # Check columns existed
        ]
        if invalid_columns:
            print(f"⚠️ Invalid target columns: {invalid_columns} ‼️")
            return None

        # ---------- Store targets ----------
        self.target_column = cleaned_target_columns
        self.y = self.target_data[self.target_column]

        print(
            f"\n🔥 Target column(s) set successfully ---> {self.target_column}\n{'-'*100}"
        )
        return self.y

    # -------------------- Set feature columns --------------------
    def set_feature_columns(self, feature_columns: list[str]):
        """
        Set the feature columns and build `X`.

        This method validates and stores the selected feature column names, then builds
        `self.X` from those columns in the source dataset.

        Parameters
        ----------
        feature_columns : list[str]
            Column names to use as independent variables / features.

        Returns
        -------
        pandas.DataFrame or None
            Feature DataFrame built from the selected feature columns if successful;
            otherwise `None`.

        Validation Rules
        ----------------
        - `target_data` must exist and be a non-empty pandas DataFrame.
        - `feature_columns` must be provided as a list of strings.
        - The feature column list must not be empty.
        - Empty strings are removed during cleaning.
        - At least one valid feature column must remain after cleaning.
        - Duplicate feature columns are not allowed.
        - All feature columns must exist in the source dataset.
        - If target columns have already been selected, feature columns must not
        overlap with any target column.

        Notes
        -----
        This method only stores explicitly selected feature columns. If no feature
        columns are provided later during `build_xy_data()`, all non-target columns
        may be selected automatically depending on the workflow.
        """
        if not self._validation():
            return None

        if not isinstance(feature_columns, list):
            print("⚠️ Feature columns must be provided as a list ‼️")
            return None

        if not feature_columns:
            print("⚠️ Feature columns must not be empty ‼️")
            return None

        if not all(isinstance(col, str) for col in feature_columns):
            print("⚠️ All feature column names must be strings ‼️")
            return None

        # ---------- Record feature columns as cleaned feature columns ----------
        cleaned_feature_columns = [
            col.strip() for col in feature_columns if col.strip()
        ]

        # ---------- Validate cleaned feature columns ----------
        if not cleaned_feature_columns:
            print("⚠️ Feature columns must contain at least one valid column name ‼️")
            return None

        if len(cleaned_feature_columns) != len(set(cleaned_feature_columns)):
            print("⚠️ Feature columns contain duplicates ‼️")
            return None

        invalid_columns = [
            col
            for col in cleaned_feature_columns
            if col not in self.target_data.columns
        ]
        if invalid_columns:
            print(f"⚠️ Invalid feature columns: {invalid_columns} ‼️")
            return None

        # ---------- Check target columns and feature columns overlapped ----------
        if self.target_column is not None:
            overlap_columns = [
                col for col in cleaned_feature_columns if col in self.target_column
            ]
            if overlap_columns:
                print(
                    f"⚠️ Target column(s) {self.target_column} must not be included "
                    f"in feature columns. Overlap: {overlap_columns} ‼️"
                )
                return None

        self.feature_columns = cleaned_feature_columns
        self.X = self.target_data[self.feature_columns]

        print(
            f"🔥 Feature columns set successfully ---> {self.feature_columns}\n{'-'*100}"
        )
        return self.X

    # -------------------- Build X and y --------------------
    def build_xy_data(self):
        """
        Build and return the feature matrix `X` and target matrix `y`.

        This method finalizes feature/target preparation after target columns have
        been selected. If feature columns have not been explicitly set, all columns
        from the source dataset except the selected target columns are automatically
        used as features.

        The method validates that target columns exist, that at least one feature
        column is available, and that no overlap exists between selected target
        columns and feature columns.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame] or None
            A tuple `(X, y)` if successful, where:

            - `X` is the feature DataFrame
            - `y` is the target DataFrame

            Returns `None` if validation fails or if `X` and `y` cannot be built.

        Workflow
        --------
        1. Validate the source dataset.
        2. Confirm that target columns have already been selected.
        3. If feature columns are not yet set, automatically select all non-target
        columns as features.
        4. Check that feature columns are not empty.
        5. Verify that target columns and feature columns do not overlap.
        6. Build `self.X` and `self.y`.
        7. Return `(self.X, self.y)`.

        Notes
        -----
        `y` is always returned as a pandas DataFrame, even when only one target column
        has been selected. This keeps output structure consistent for both single-
        target and multi-target workflows.

        Whether the returned `y` can be passed directly into a specific machine-
        learning model depends on whether that model supports single-output or
        multi-output targets.
        """
        if not self._validation():
            return None

        if self.target_column is None:
            print("⚠️ Target column has not been set ‼️")
            return None

        if self.feature_columns is None:
            self.feature_columns = [
                col for col in self.target_data.columns if col not in self.target_column
            ]

        if not self.feature_columns:
            print("⚠️ No feature columns available to build X ‼️")
            return None

        # ---------- Check feature and target columns overlapped ----------
        overlap_columns = [
            col for col in self.feature_columns if col in self.target_column
        ]
        if overlap_columns:
            print(
                f"⚠️ Target column(s) {self.target_column} must not be included "
                f"in feature columns. Overlap: {overlap_columns} ‼️"
            )
            return None

        # ---------- Record target and feature columns from target data  ----------
        self.X = self.target_data[self.feature_columns]
        self.y = self.target_data[self.target_column]

        print("🔥 X and y built successfully.")

        return self.X, self.y


# =================================================
