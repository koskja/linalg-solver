import sympy


def pcformat(fstr, *vals):
    """
    Format a percent sign string with the given values.
    Example:
    >>> pcformat(r"%s + %s = %s", 1, 2, 3)
    "1 + 2 = 3"
    """
    formatted_vals = tuple(cformat(val) for val in vals)
    return fstr % formatted_vals


def cformat(val):
    if hasattr(val, "cformat") and callable(val.cformat):
        return val.cformat()
    if isinstance(val, str):
        return val
    if hasattr(val, "as_latex") and callable(val.as_latex):
        return val.as_latex()
    try:
        return sympy.latex(val)
    except Exception:  # this probably should be specific to some kind of exception
        pass
    return str(val)
