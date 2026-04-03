# -------------------- Imported Modules -------------------
import logging
import os


# -------------------- Logging Setup --------------------
def apollo_init_logging() -> logging.Logger:
    """
    Initialize and configure the Apollo application logger.

    This function creates the default Apollo log folder and log file under the
    current project directory, configures a named logger, and attaches both a
    file handler and a console stream handler when they have not already been
    added.

    The logger uses a unified message format that includes timestamp, log
    level, logger name, and log message text. Repeated calls to this function
    do not duplicate existing handlers for the same log file or console stream.

    Returns
    -------
    logging.Logger
        Configured logger instance for the Apollo system.

    Notes
    -----
    - The log folder is created automatically if it does not already exist.
    - The log file is stored in ``Apollo_Logs/Apollo_Log.log`` under the
      current file's directory.
    - File logging uses UTF-8 with BOM encoding via ``utf-8-sig``.
    - Logger propagation is disabled to prevent duplicate log output through
      ancestor loggers.
    - This function is safe to call multiple times because it checks for
      existing handlers before adding new ones.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(project_root, "Apollo_Logs")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "Apollo_Log.log")

    logger = logging.getLogger("Apollo")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "") == log_file
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_file, encoding="utf-8-sig")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.info("Logging initialized to: %s", log_file)
    return logger


# -------------------- Execute --------------------
if __name__ == "__main__":
    logger = apollo_init_logging()


# --------------------------------------------------------
