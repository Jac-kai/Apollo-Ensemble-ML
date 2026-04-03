# -------------------- Modules Import --------------------
from functools import wraps


# -------------------- Menu Wrapper --------------------
def menu_wrapper(menu_name: str):
    """
    Wrap a menu function with a standard terminal header and basic error handling.

    This decorator factory creates a decorator that adds a unified execution style
    to menu functions used in the Zeus terminal interface. When the wrapped
    function is called, it first prints a formatted menu title banner, then
    executes the original function inside a ``try`` block.

    If execution succeeds, a simple completion message is printed and the original
    return value is passed through unchanged. If an exception occurs, the wrapper
    prints an error message and returns ``None`` instead of propagating the
    exception.

    Parameters
    ----------
    menu_name : str
        Human-readable menu title displayed before function execution and used in
        status and error messages.

    Returns
    -------
    callable
        A decorator that wraps the target menu function.

    Workflow
    --------
    1. Build a decorator using the provided ``menu_name``
    2. Wrap the target function with ``functools.wraps``
    3. Print a formatted menu banner before execution
    4. Execute the target function inside a ``try`` block
    5. Print a completion message if execution succeeds
    6. Print an error message and return ``None`` if execution fails

    Notes
    -----
    This decorator is intended for user-facing menu functions where graceful
    failure is preferred over full traceback interruption.

    Examples
    --------
    Use the decorator on a menu function::

        @menu_wrapper("Model Training")
        def train_model_menu(zeus):
            ...

    The decorated function will automatically show a title banner and catch
    unexpected execution errors.
    """

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            print(f"---------- 🔥  {menu_name} 🔥  ----------")
            try:
                result = func(*args, **kwargs)
                print(f"🕯️ 🕯️ 🕯️  Executing... {menu_name} 🕯️ 🕯️ 🕯️\n")
                return result
            except Exception as e:
                print(f"⚠️ [ERROR] {menu_name} failed: {e} ‼️")
                return None

        return wrapped

    return decorator


# -------------------- input_int --------------------
def input_int(prompt: str, default: int | None = None) -> int | None:
    """
    Read an integer value from terminal input.

    This helper prompts the user for numeric input intended to represent an
    integer selection, such as a menu number. The user may enter ``0`` to go back,
    or press ENTER to use the provided default value.

    If a valid integer is entered, that integer is returned. If the input is empty,
    the ``default`` value is returned. If the input cannot be converted to an
    integer, a warning message is shown and the default value is returned.

    Parameters
    ----------
    prompt : str
        Prompt text displayed before the standardized input hint.
    default : int | None, default=None
        Default integer returned when the user presses ENTER or when conversion
        fails.

    Returns
    -------
    int | None
        Parsed integer value, the provided default value, or ``None`` when the
        user chooses back.

    Notes
    -----
    This helper is designed for menu-style terminal workflows and treats ``0`` as
    a back/cancel signal.

    Examples
    --------
    Read a menu choice with a default value::

        choice = input_int("🕯️ Select service", default=1)
    """
    try:
        value = input(prompt + " (Number only | 0 to ↩️  BACK) ⚡ ").strip()

        if value == "0":
            return None

        return int(value) if value else default

    except ValueError:
        print(f"⚠️ Invalid input, using default {default} ‼️")
        return default


# -------------------- input_yesno --------------------
def input_yesno(prompt: str, default: bool = False) -> bool | None:
    """
    Read a yes/no decision from terminal input.

    This helper repeatedly prompts the user until a valid yes/no response is
    entered, unless the user presses ENTER to use the default value or enters
    ``0`` to go back.

    Accepted yes values are ``"y"`` and ``"yes"``.
    Accepted no values are ``"n"`` and ``"no"``.

    Parameters
    ----------
    prompt : str
        Prompt text displayed before the standardized input hint.
    default : bool, default=False
        Boolean value returned when the user presses ENTER without entering any
        explicit response.

    Returns
    -------
    bool | None
        ``True`` for yes, ``False`` for no, the default value for empty input, or
        ``None`` when the user chooses back.

    Notes
    -----
    This helper is intended for interactive menu confirmation prompts such as
    saving output, updating data in place, or enabling optional plotting features.

    Examples
    --------
    Ask whether to save output::

        save_fig = input_yesno("🕯️ Save figure", default=False)
    """
    while True:
        value = (
            input(prompt + " (y or yes/n or no | 0 to ↩️  BACK) ⚡ ").strip().lower()
        )

        if value == "0":
            return None
        if value == "":
            return default
        if value in ["y", "yes"]:
            return True
        if value in ["n", "no"]:
            return False

        print("⚠️ Invalid input, please enter y/yes or n/no ‼️")


# -------------------- input_list --------------------
def input_list(prompt: str) -> list[str] | str | None:
    """
    Read comma-separated terminal input and return cleaned text items as a list.

    This helper prompts the user to enter one or more comma-separated values in a
    single line. Each non-empty item is stripped of surrounding whitespace and the
    cleaned items are returned as a list of strings.

    Behavior
    --------
    - If the user presses ENTER without typing anything, the function returns
    ``None`` to indicate skipped input.
    - If the user enters ``0``, the function returns ``"__BACK__"`` to signal a
    back action to the caller.
    - If comma-separated values are entered, the function splits them by comma,
    strips surrounding whitespace from each item, removes empty items, and
    returns the cleaned list.
    - If an unexpected exception occurs while reading input, the function prints a
    warning message and returns ``None``.

    Parameters
    ----------
    prompt : str
        Prompt text displayed before the standardized input hint.

    Returns
    -------
    list[str] | str | None
        - ``list[str]`` when valid comma-separated values are entered
        - ``"__BACK__"`` when the user chooses back
        - ``None`` when the user skips input or input reading fails

    Workflow
    --------
    1. Show the prompt with comma-separated input instructions
    2. Read terminal input and strip surrounding whitespace
    3. Return ``"__BACK__"`` if the input is ``"0"``
    4. Return ``None`` if the input is empty
    5. Split the input by comma and clean each item
    6. Return the cleaned list of non-empty values

    Notes
    -----
    This helper is useful for interactive workflows that allow users to enter
    multiple column names, feature names, labels, categories, or configuration
    tokens in a single terminal prompt.

    Examples
    --------
    Read multiple column names::

        cols = input_list("🕯️ Enter selected columns")

    Possible input and output::

        age, salary, target   -> ["age", "salary", "target"]
        ENTER                 -> None
        0                     -> "__BACK__"
    """
    try:
        value = input(
            f"{prompt} (comma-separated) (ENTER to skip | 0 to ↩️  BACK) ⚡ "
        ).strip()

        if value == "0":
            return "__BACK__"

        return [v.strip() for v in value.split(",") if v.strip()] if value else None

    except Exception:
        print("⚠️ Failed to read list input, returning None ‼️")
        return None


# -------------------- index_list --------------------
def index_list(data: object) -> dict[int, object]:
    """
    Display a numbered index list for a dataset and return its mapping.

    This helper validates that the provided object exposes a usable ``index``
    attribute, then builds and prints a numbered mapping from 1-based display
    numbers to the object's real index labels. The resulting mapping is returned so
    the caller can translate user menu selections back to actual index values.

    Behavior
    --------
    - If ``data`` is ``None``, the function prints a warning and returns an empty
    dictionary.
    - If ``data`` does not expose an ``index`` attribute, the function prints a
    warning and returns an empty dictionary.
    - If the index exists but is empty, the function prints a warning and returns
    an empty dictionary.
    - If the index is valid, the function prints the numbered index list and
    returns the corresponding mapping.
    - If an unexpected exception occurs during processing, the function prints a
    warning message and returns an empty dictionary.

    Parameters
    ----------
    data : object
        Target object expected to expose an ``index`` attribute, typically a pandas
        ``DataFrame`` or ``Series``.

    Returns
    -------
    dict[int, object]
        Dictionary mapping 1-based display numbers to actual index labels. Returns
        an empty dictionary when no usable index information is available.

    Workflow
    --------
    1. Validate that input data exists
    2. Validate that the object has an ``index`` attribute
    3. Validate that the index is not empty
    4. Build a 1-based mapping from display number to true index label
    5. Print the formatted index list to the terminal
    6. Return the mapping to the caller

    Notes
    -----
    This helper is intended for terminal-based workflows where users choose rows by
    display number instead of typing full index labels manually.

    Examples
    --------
    Display row index options for later selection::

        idx_map = index_list(df)

    Possible returned mapping::

        {1: "A001", 2: "A002", 3: "A003"}

    A caller may then convert a user choice like ``2`` into the real index label
    ``"A002"``.
    """
    try:
        if data is None:
            print("⚠️ No data available ‼️")
            return {}

        if not hasattr(data, "index"):
            print("⚠️ Target data has no index attribute ‼️")
            return {}

        if len(data.index) == 0:
            print("⚠️ No index found in target data ‼️")
            return {}

        idx_map = {i: idx for i, idx in enumerate(data.index, 1)}

        print("🍁----- Index List -----🍁")
        for i, idx in idx_map.items():
            print(f"🐝 {i}. {idx}")
        print("-" * 40)

        return idx_map

    except Exception as e:
        print(f"⚠️ Failed to display index list: {e} ‼️")
        return {}


# -------------------- column_list --------------------
def column_list(data: object) -> dict[int, str]:
    """
    Display a numbered column list for a dataset and return its mapping.

    This helper validates that the provided object exposes usable ``columns`` and
    ``dtypes`` information, then builds and prints a numbered mapping from 1-based
    display numbers to real column names. Each printed entry includes the column
    name and its dtype so the user can inspect both structure and data type before
    making a selection.

    Behavior
    --------
    - If ``data`` is ``None``, the function prints a warning and returns an empty
    dictionary.
    - If ``data`` does not expose a ``columns`` attribute, the function prints a
    warning and returns an empty dictionary.
    - If no columns are available, the function prints a warning and returns an
    empty dictionary.
    - If valid column information exists, the function prints the numbered column
    list with dtype information and returns the corresponding mapping.
    - If an unexpected exception occurs during processing, the function prints a
    warning message and returns an empty dictionary.

    Parameters
    ----------
    data : object
        Target object expected to expose ``columns`` and ``dtypes`` attributes,
        typically a pandas ``DataFrame``.

    Returns
    -------
    dict[int, str]
        Dictionary mapping 1-based display numbers to actual column names. Returns
        an empty dictionary when no usable column information is available.

    Workflow
    --------
    1. Validate that input data exists
    2. Validate that the object has a ``columns`` attribute
    3. Validate that at least one column is available
    4. Build a 1-based mapping from display number to true column name
    5. Build a dtype lookup for each column
    6. Print the formatted column list to the terminal
    7. Return the mapping to the caller

    Notes
    -----
    This helper is commonly used in menu-driven data workflows where users select
    columns by number rather than typing full column names manually.

    Examples
    --------
    Display selectable columns for a DataFrame::

        col_map = column_list(df)

    Possible printed output::

        1. age (int64)
        2. salary (float64)
        3. city (object)

    Possible returned mapping::

        {1: "age", 2: "salary", 3: "city"}
    """
    try:
        if data is None:
            print("⚠️ No data available ‼️")
            return {}

        if not hasattr(data, "columns"):
            print("⚠️ Target data has no columns attribute ‼️")
            return {}

        if len(data.columns) == 0:
            print("⚠️ No columns found in target data ‼️")
            return {}

        col_map = {i: col for i, col in enumerate(data.columns, 1)}
        col_type_map = {col: str(dtype) for col, dtype in data.dtypes.items()}

        print(f"🍁----- Column List -----🍁")
        for i, col in col_map.items():
            print(f"🐝 {i}. {col} ({col_type_map[col]})")
        print("-" * 40)

        return col_map

    except Exception as e:
        print(f"⚠️ Failed to display column list: {e} ‼️")
        return {}


# -------------------- Helper: input text --------------------
def input_text_value(prompt: str) -> str | None:
    """
    Read a manually typed text value from terminal input.

    This helper prompts the user to enter an arbitrary text value in the terminal.
    The entered text is stripped of surrounding whitespace and returned as a
    string. If the user enters ``0``, the function treats that input as a back
    action and returns ``None``.

    Behavior
    --------
    - If the user enters ``0``, the function returns ``None`` to indicate back.
    - Otherwise, the function strips leading and trailing whitespace and returns
    the resulting text.
    - Empty text is returned as an empty string if the user presses ENTER.

    Parameters
    ----------
    prompt : str
        Prompt text displayed before the standardized input hint.

    Returns
    -------
    str | None
        The cleaned text entered by the user, or ``None`` when the user chooses
        back.

    Workflow
    --------
    1. Show the prompt with manual text-entry instructions
    2. Read terminal input
    3. Strip leading and trailing whitespace
    4. Return ``None`` if the input is ``"0"``
    5. Otherwise return the entered text

    Notes
    -----
    This helper is intended for terminal workflows that require free-text input,
    such as manually typing a column name, custom label, file keyword, rename
    target, or other user-defined text value.

    Examples
    --------
    Read a custom column name::

        target_col = input_text_value("🕯️ Enter target column")

    Possible input and output::

        target        -> "target"
        new_label     -> "new_label"
        0             -> None
    """
    value = input(f"{prompt} (typing manually) (0 to ↩️  BACK) ⚡ ").strip()

    if value == "0":
        return None

    return value


# -----------------------------------------
