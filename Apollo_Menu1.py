# -------------------- Import Modules --------------------
import logging

from Apollo.Apollo_ML_Engine import ApolloEngine
from Apollo.Menu_Helper_Decorator import (
    column_list,
    input_int,
    input_list,
    menu_wrapper,
)

logger = logging.getLogger("Apollo")


# -------------------- Loaded ML Data Menu --------------------
@menu_wrapper("Loaded ML Data")
def loaded_ml_data_menu(apollo: ApolloEngine):
    """
    Interactively browse folders and files, then load a dataset into Apollo.

    This menu guides the user through the dataset-loading workflow by:
    1. listing available working-place folders,
    2. prompting the user to select a folder,
    3. listing available files inside that folder,
    4. prompting the user to select a file,
    5. calling ``ApolloEngine.ml_dataset_search(...)`` to load the dataset.

    If the dataset is loaded successfully, Apollo helper cores are also built
    by the engine so downstream feature-selection and model workflows can use
    the loaded data immediately.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance used to access the data-loading workflow.

    Returns
    -------
    None
        Returns ``None`` when the user exits the menu, when no folders are
        available, or when the workflow ends after a successful dataset load.

    Notes
    -----
    - If no working-place folders are found, the menu prints a warning and ends.
    - If the selected folder contains no files, the menu prints a warning and
      returns to the folder-selection loop.
    - If dataset loading fails, the menu prints a warning and allows the user
      to try again.
    - If the user cancels during folder or file selection, the menu exits
      immediately.
    """
    logger.info("Entered menu: Loaded ML Data")

    while True:
        # ---------- Show working-place folders ----------
        folders = apollo.hunter_core.working_place_searcher()
        if not folders:
            logger.warning("No working-place folders found")
            print("⚠️ No folders available ‼️")
            return None

        logger.info("Working-place folders loaded")

        print(f"\n----- 🔥 Folder Lists 🔥-----\n{'-'*50}")
        for i, folder in folders.items():
            print(f"📂 {i}. {folder}")

        # ---------- Select folder ----------
        selected_folder_num = input_int("🕯️ Select folder")
        if selected_folder_num is None:
            logger.info("Loaded ML Data menu exited at folder selection")
            return None

        logger.info("Selected folder number: %s", selected_folder_num)

        # ---------- Show files inside selected folder ----------
        files = apollo.hunter_core.files_searcher_from_folders(
            selected_folder_num=selected_folder_num,
        )
        if not files:
            logger.warning(
                "No files found in selected folder | folder_num=%s", selected_folder_num
            )
            print("⚠️ No files available in the selected folder ‼️")
            continue

        logger.info("Files loaded from folder number: %s", selected_folder_num)

        print(f"\n----- 🔥 File Lists 🔥-----\n{'-'*50}")
        for i, file in files.items():
            print(f"📄 {i}. {file}")

        # ---------- Select file ----------
        selected_file_num = input_int("🕯️ Select file")
        if selected_file_num is None:
            logger.info("Loaded ML Data menu exited at file selection")
            return None

        logger.info("Selected file number: %s", selected_file_num)

        # ---------- Load dataset ----------
        logger.info(
            "Start loading dataset | folder_num=%s | file_num=%s",
            selected_folder_num,
            selected_file_num,
        )

        loaded_data = apollo.ml_dataset_search(
            selected_folder_num=selected_folder_num,
            selected_file_num=selected_file_num,
        )

        if loaded_data is None:
            logger.warning(
                "Failed to load ML dataset | folder_num=%s | file_num=%s",
                selected_folder_num,
                selected_file_num,
            )
            print("⚠️ Failed to load ML dataset ‼️")
            continue

        logger.info(
            "ML dataset loaded successfully | folder_num=%s | file_num=%s",
            selected_folder_num,
            selected_file_num,
        )
        print(f"🔥 ML dataset loaded successfully.\n{'-' * 100}")
        return


# -------------------- Select Feature Menu --------------------
@menu_wrapper("Select Featand and Target")
def select_feature_target_menu(apollo: ApolloEngine):
    """
    Interactively select target and feature columns for Apollo workflows.

    This menu validates that source data and ``FeatureCore`` are available,
    shows the available dataset columns, and allows the user to configure
    target and feature columns by entering column indices.

    Workflow behavior
    -----------------
    1. Display available columns from the current source dataset.
    2. Prompt the user to select one or more target-column indices.
    3. Prompt the user to select feature-column indices.
    4. If no feature columns are entered, automatically use all non-target
       columns as features.
    5. Apply the selected target and feature columns through
       ``ApolloEngine.set_target_column(...)`` and
       ``ApolloEngine.set_feature_columns(...)``.
    6. Build X/Y data through ``ApolloEngine.build_xy_data()``.
    7. Show the selected target and feature columns and ask the user to
       confirm or reselect them.

    Parameters
    ----------
    apollo : ApolloEngine
        Active Apollo engine instance used to access source data and feature
        configuration workflows.

    Returns
    -------
    None
        Returns ``None`` when the user exits the menu, when required source
        data is unavailable, when no columns are available for selection, or
        after the selection is confirmed successfully.

    Notes
    -----
    - The menu supports both single-target and multi-target selection.
    - Target selections must be valid numeric column indices.
    - Feature selections must also be valid numeric indices and cannot overlap
      with the selected target columns.
    - When the user chooses to reselect, ``feature_core`` is rebuilt from the
      current source data so the selection workflow starts fresh.
    - If the user exits during target selection, feature selection, or final
      confirmation, the menu ends immediately.
    """
    logger.info("Entered menu: Select Feature")

    if apollo.source_data is None:
        logger.warning("Select Feature menu failed: no source data available")
        print("⚠️ No source data available. Please load data first ‼️")
        return

    if apollo.feature_core is None:
        logger.warning("Select Feature menu failed: FeatureCore not built")
        print("⚠️ FeatureCore has not been built. Please load data first ‼️")
        return

    while True:
        # ---------- Show available columns ----------
        col_map = column_list(apollo.source_data)
        if not col_map:
            logger.warning("No columns available for feature selection")
            print("⚠️ No columns available for selection ‼️")
            return

        # ---------- Select target column by index ----------
        target_input = input_list("🕯️ Select TARGET column index")
        if target_input == "__BACK__":
            logger.info("Select Feature menu exited at target selection")
            return

        if not target_input:
            logger.warning("Target column selection is empty")
            print("⚠️ TARGET column selection is required ‼️")
            continue

        if not all(str(item).isdigit() for item in target_input):
            logger.warning(
                "Invalid target selection: non-numeric input=%s", target_input
            )
            print("⚠️ TARGET selections must all be numeric indices ‼️")
            continue

        target_indices = [int(item) for item in target_input]

        if any(idx not in col_map for idx in target_indices):
            logger.warning("Target index out of range | input=%s", target_indices)
            print("⚠️ One or more TARGET indices are out of range ‼️")
            continue

        target_columns = [col_map[idx] for idx in target_indices]
        logger.info("Selected target columns: %s", target_columns)

        # ---------- Select feature columns by index ----------
        feature_input = input_list("🕯️ Select FEATURE column index(es)")
        if feature_input == "__BACK__":
            logger.info("Select Feature menu exited at feature selection")
            return

        if feature_input is None:
            logger.info(
                "No feature columns entered; using all non-target columns automatically | target=%s",
                target_columns,
            )
            print(
                "🔔 No feature columns entered. Using all non-target columns automatically."
            )

            feature_columns = [
                col for col in apollo.source_data.columns if col not in target_columns
            ]

        else:
            if not all(str(item).isdigit() for item in feature_input):
                logger.warning(
                    "Invalid feature selection: non-numeric input=%s", feature_input
                )
                print("⚠️ FEATURE selections must all be numeric indices ‼️")
                continue

            feature_indices = [int(item) for item in feature_input]

            if any(idx not in col_map for idx in feature_indices):
                logger.warning("Feature index out of range | input=%s", feature_indices)
                print("⚠️ One or more FEATURE indices are out of range ‼️")
                continue

            feature_columns = [col_map[idx] for idx in feature_indices]

            if any(col in target_columns for col in feature_columns):
                logger.warning(
                    "Invalid feature selection: feature columns overlap with target columns | target=%s | features=%s",
                    target_columns,
                    feature_columns,
                )
                print("⚠️ FEATURE columns cannot include TARGET columns ‼️")
                continue

            logger.info(
                "Selected feature columns | target=%s | features=%s",
                target_columns,
                feature_columns,
            )

        # ---------- Set target / feature in ApolloEngine ----------
        target_result = apollo.set_target_column(
            target_columns[0] if len(target_columns) == 1 else target_columns
        )
        if target_result is None:
            logger.warning("Failed to set target columns: %s", target_columns)
            print("⚠️ Failed to set target columns ‼️")
            continue

        feature_result = apollo.set_feature_columns(feature_columns)
        if feature_result is None:
            logger.warning("Failed to set feature columns: %s", feature_columns)
            print("⚠️ Failed to set feature columns ‼️")
            continue

        xy_data = apollo.build_xy_data()
        if xy_data is None:
            logger.warning(
                "Failed to build X and y | target=%s | features=%s",
                target_columns,
                feature_columns,
            )
            print("⚠️ Failed to build X and y ‼️")
            continue

        logger.info("Feature and target selection completed successfully")
        print("🔥 Feature and target selection completed.")
        print("👓 Show the selected features and targets.")

        # ---------- Show current feature selection ----------
        try:
            print(f"🎯 Target: {target_columns}")
            print(f"🧩 Features: {feature_columns}")
        except Exception:
            pass

        # ---------- Confirm the selection ----------
        while True:
            confirm = input_int("🕯️ (1) Confirm selection | (2) Reselect | (0) Back")

            if confirm == 1:
                logger.info("Feature selection confirmed")
                print("🔥 Current selection confirmed.")
                return

            elif confirm == 2:
                logger.info("Feature selection reset requested")
                apollo.feature_core = type(apollo.feature_core)(apollo.source_data)
                print("♻️ Feature selection has been reset. Please select again.")
                break

            elif confirm == 0 or confirm is None:
                logger.info("Exited Select Feature menu at confirmation step")
                return

            else:
                logger.warning("Invalid confirmation selection: %s", confirm)
                print("⚠️ Invalid selection ‼️")


# =================================================
