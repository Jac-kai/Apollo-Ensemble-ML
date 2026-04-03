# -------------------- Modules Import --------------------
import logging
import time

from Apollo.Apollo_Logging import apollo_init_logging
from Apollo.Apollo_Menu1 import loaded_ml_data_menu, select_feature_target_menu
from Apollo.Apollo_Menu2 import model_management_menu
from Apollo.Apollo_Menu3 import evaluation_menu
from Apollo.Apollo_ML_Engine import ApolloEngine
from Apollo.Menu_Helper_Decorator import input_int

logger = logging.getLogger("Apollo")


# -------------------- cornus_control --------------------
def apollo_control():
    """
    Run the main terminal control loop for the Apollo system.

    This function creates a single ``ApolloEngine`` instance and launches the
    top-level interactive menu for the Apollo ensemble-learning workflow.

    Through this menu, the user can:
    1. load a dataset,
    2. configure feature and target columns,
    3. open model management workflows,
    4. access evaluation-related tools,
    5. leave the Apollo system.

    The same engine instance is kept alive across menu selections so that
    loaded data, configured features, trained models, and evaluation state
    can be reused throughout the current session.

    Returns
    -------
    None

    Notes
    -----
    Menu selection is handled through ``input_int(...)``. If the user exits
    through cancel/back behavior, the control loop ends gracefully. Invalid
    menu selections are reported and the menu is shown again.
    """
    logger.info("Starting Apollo main control loop")

    apollo_engine = ApolloEngine()
    logger.info("ApolloEngine instance created")

    menu = [
        (1, "📨 Upload Data", loaded_ml_data_menu),
        (2, "🔎 Features and Targets", select_feature_target_menu),
        (3, "🧠 Models", model_management_menu),
        (4, "🪶 Evaluations", evaluation_menu),
        (0, "🍂 Leave System", None),
    ]
    menu_width = 35

    while True:
        logger.info("Displaying Apollo main menu")
        print("🏮  Apollo Main Menu 🏮 ".center(menu_width, "━"))

        for opt, action, _ in menu:
            print(f"{opt}. {action:<{menu_width-6}}")
        print("━" * menu_width)

        choice = input_int(f"🕯️  Select Services (🔅 {time.asctime()})⚡ ", default=-1)

        if choice is None:
            logger.info("Main menu exited by user cancel/back")
            print("🎶🎶🎶 Leaving Apollo Engine... Goodbye 🍁 Zack King")
            break

        if choice == -1:
            logger.warning("Invalid main menu selection received: %s", choice)
            print("⚠️ Invalid selection ‼️")
            continue

        logger.info("Main menu selection: %s", choice)

        for opt, label, func in menu:
            if choice == opt and func:
                logger.info("Dispatching main menu action: %s - %s", opt, label)
                func(apollo_engine)
                break
            if choice == 0 and opt == 0:
                logger.info("User selected Leave System")
                print("🎶🎶🎶 Leaving Apollo Engine... Goodbye 🍁 Zack King")
                return
        else:
            logger.warning("Main menu selection out of range: %s", choice)
            print("⚠️ Invalid selection ‼️")


# -------------------- Execute --------------------
if __name__ == "__main__":
    logger = apollo_init_logging()
    logger.info("Apollo logging initialized from main entry")
    apollo_control()


# -----------------------------------------
