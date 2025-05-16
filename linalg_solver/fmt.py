from typing import List, Any
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


def cformat(val, arg_of=None):
    if hasattr(val, "cformat") and callable(val.cformat):
        return val.cformat(arg_of)
    if isinstance(val, str):
        return val
    if hasattr(val, "as_latex") and callable(val.as_latex):
        return val.as_latex()
    try:
        return sympy.latex(val)
    except Exception:  # this probably should be specific to some kind of exception
        pass
    return str(val)


def pretty_print_arithmetic(a: any, op: str, b: any) -> str:
    if op == "+":
        if b == 0:
            return cformat(a)
        if a == 0:
            return cformat(b)
        if b < 0:
            b = -b
        return pcformat(r"%s+%s", a, b)
    if op == "-":
        if b == 0:
            return cformat(a)
        if a == 0:
            return cformat(-b)
        if b < 0:
            b = -b
        return pcformat(r"%s-%s", a, b)
    if op == "*":
        if a == 0 or b == 0:
            return cformat(0)
        if a == 1:
            return cformat(b)
        if b == 1:
            return cformat(a)
        if b < 0:
            b = -b
            a = -a
        return pcformat(r"%s \times %s", a, b)


def make_latex_matrix(items: List[List[Any]]) -> str:
    start = r"\begin{pmatrix}"
    end = r"\end{pmatrix}"
    rows = [r" & ".join([cformat(item) for item in row]) for row in items]
    return start + (r"\\[0.1em]" + "\n").join(rows) + end


def make_latex_vector(items: List[Any]) -> str:
    start = r"\begin{pmatrix}"
    end = r"\end{pmatrix}"
    return start + (r"\\[0.1em]" + "\n").join([cformat(item) for item in items]) + end


def make_latex_augmented_matrix(items: List[List[Any]], bar_col: int = None) -> str:
    if len(items[0]) <= 1:
        return make_latex_matrix(items)
    if bar_col is None:
        bar_col = len(items[0]) - 1
    rows = [r" & ".join([cformat(item) for item in row]) for row in items]
    # Build the column format string with a vertical bar at bar_col
    n_cols = len(items[0])
    col_format = "".join([("|c" if j == bar_col else "c") for j in range(n_cols)])
    start = r"\left(\begin{array}{" + col_format + "}\n"
    end = "\n" + r"\end{array}\right)"
    return start + (r" \\[0.1em]" + "\n").join(rows) + end


def multi_add_vargs(*items: List[Any]) -> Any:
    return multi_add(list(items))


def multi_add(items: List[Any]) -> Any:
    if len(items) == 0:
        raise ValueError("At least one item is required")
    if len(items) == 1:
        return items[0]
    if hasattr(items[0], "multi_add") and callable(items[0].multi_add):
        return items[0].multi_add(*items[1:])
    return sum(items)


def scalar_mul(item: Any, scalar: Any) -> Any:
    if hasattr(item, "scalar_mul") and callable(item.scalar_mul):
        return item.scalar_mul(scalar)
    return item * scalar


def linear_comb(scalars: List[Any], items: List[Any]) -> Any:
    if len(scalars) != len(items):
        raise ValueError("Scalars and items must have the same length")
    return multi_add([scalar_mul(item, scalar) for scalar, item in zip(scalars, items)])


def make_latex_vertical_augmented_matrix(
    header_row_latex: str, matrix_items: List[List[Any]], num_cols: int
) -> str:
    assert num_cols > 0
    matrix_content_rows_latex = []
    for row_items in matrix_items:
        matrix_content_rows_latex.append(
            " & ".join([cformat(item) for item in row_items])
        )

    col_spec = "c" * num_cols

    if not matrix_items:  # Only header row (e.g., basis vectors are 0-dimensional)
        matrix_body_for_array = header_row_latex
    else:  # Header row, hline, then matrix item rows
        matrix_body_for_array = (
            header_row_latex + r" \\ \hline " + r" \\ ".join(matrix_content_rows_latex)
        )

    # Corrected f-string for LaTeX output
    return r"\left( \begin{array}{%s} %s \end{array} \right)" % (
        col_spec,
        matrix_body_for_array,
    )
