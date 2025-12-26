"""
Determinant computation using optimal process from Rust linalg_helper.

This module finds the optimal strategy for computing a sparse matrix's
determinant using the Rust library, then executes that strategy in Python.
"""

from __future__ import annotations

from typing import Any, List, Tuple, TYPE_CHECKING
import linalg_helper

from linalg_solver.permutation import RowColPermutation

from .log import log
from .fmt import cformat, multi_add, multi_mul, make_latex_matrix, pcformat

if TYPE_CHECKING:
    from .linalg import Matrix


def matrix_to_sparsity_pattern(matrix: "Matrix") -> List[List[bool]]:
    """Convert a Matrix to a boolean sparsity pattern for Rust."""
    return [[item != 0 for item in row] for row in matrix.items]


def check_sparsity(
    matrix: "Matrix",
    expected_nonzeros: List[Tuple[int, int]],
    rows: List[int],
    cols: List[int],
) -> None:
    """
    Verify that the matrix is at least as sparse as expected.

    The process expects certain positions to be non-zero. The actual matrix
    may have additional zeros (be sparser), but must not have non-zeros
    where the process expects zeros.

    Args:
        matrix: The matrix to check.
        expected_nonzeros: List of (row, col) tuples that are expected non-zero.
        rows: Row indices mapping local to actual rows.
        cols: Column indices mapping local to actual columns.

    Raises:
        ValueError: If the matrix has a non-zero where the process expects zero.
    """
    expected_set = set(expected_nonzeros)

    n_rows = len(rows)
    n_cols = len(cols)

    for local_r in range(n_rows):
        for local_c in range(n_cols):
            actual_r = rows[local_r]
            actual_c = cols[local_c]
            value = matrix.items[actual_r][actual_c]

            if value != 0 and (local_r, local_c) not in expected_set:
                raise ValueError(
                    r"Sparsity mismatch: matrix has non-zero at position (%s, %s) "
                    r"(local (%s, %s)) but the process expects zero there. "
                    r"Expected non-zeros: %s"
                    % (actual_r, actual_c, local_r, local_c, sorted(expected_nonzeros))
                )


def find_optimal_process(
    matrix: "Matrix",
) -> Tuple["linalg_helper.PyCost", "linalg_helper.PyProcess"]:
    """
    Find the optimal determinant computation process for a matrix.

    Args:
        matrix: The matrix to find the optimal process for.

    Returns:
        A tuple of (cost, process) where cost contains the operation counts
        and process describes the optimal computation strategy.
    """
    pattern = matrix_to_sparsity_pattern(matrix)
    result = linalg_helper.find_optimal_determinant_process(pattern)
    return result.cost, result.process


def execute_process(
    matrix: "Matrix",
    process: "linalg_helper.PyProcess",
    rows: List[int] = None,
    cols: List[int] = None,
    do_log: bool = False,
    sign: int = 1,
) -> Any:
    """
    Execute a determinant computation process on a matrix.

    Args:
        matrix: The full matrix containing the values.
        process: The PyProcess describing the computation strategy.
        rows: Row indices to use (defaults to all rows).
        cols: Column indices to use (defaults to all columns).
        do_log: Whether to log computation steps.
        sign: Current sign multiplier (for tracking row swaps).

    Returns:
        The computed determinant value.

    Raises:
        ValueError: If the matrix has non-zeros where the process expects zeros.
    """
    n = matrix.rows
    if rows is None:
        rows = list(range(n))
    if cols is None:
        cols = list(range(n))

    process_type = process.process_type

    # Verify the matrix is at least as sparse as the process expects.
    #
    # NOTE:
    # For transformation processes like AddRow, the Rust-side
    # `expected_nonzeros` corresponds to the *result* of the transformation
    # (after eliminating/swapping), not the input. We therefore validate
    # sparsity after applying the transformation in their respective executors.
    if process_type not in ("AddRow"):
        check_sparsity(matrix, process.expected_nonzeros, rows, cols)

    if process_type == "Direct":
        return _execute_direct(matrix, rows, cols, do_log, sign)
    elif process_type == "RowExpansion":
        return _execute_row_expansion(matrix, process, rows, cols, do_log, sign)
    elif process_type == "ColExpansion":
        return _execute_col_expansion(matrix, process, rows, cols, do_log, sign)
    elif process_type == "BlockTriangular":
        return _execute_block_triangular(matrix, process, rows, cols, do_log, sign)
    elif process_type == "AddRow":
        return _execute_add_row(matrix, process, rows, cols, do_log, sign)
    else:
        raise ValueError(r"Unknown process type: %s" % process_type)


def _get_element(
    matrix: "Matrix", rows: List[int], cols: List[int], i: int, j: int
) -> Any:
    """Get element at logical position (i, j) using row/col index mappings."""
    return matrix.items[rows[i]][cols[j]]


def _build_submatrix_items(
    matrix: "Matrix",
    rows: List[int],
    cols: List[int],
) -> List[List[Any]]:
    """Build a list of lists representing the submatrix for display."""
    return [
        [matrix.items[rows[i]][cols[j]] for j in range(len(cols))]
        for i in range(len(rows))
    ]


def _execute_direct(
    matrix: "Matrix",
    rows: List[int],
    cols: List[int],
    do_log: bool,
    sign: int,
) -> Any:
    """Execute direct determinant computation (for n <= 2)."""
    n = len(rows)

    if n == 0:
        if do_log:
            log(r"$\det([]) = 1$")
        return sign * 1

    if n == 1:
        val = _get_element(matrix, rows, cols, 0, 0)
        result = sign * val
        # Don't log 1x1 determinants - they're trivial
        return result

    if n == 2:
        a = _get_element(matrix, rows, cols, 0, 0)
        b = _get_element(matrix, rows, cols, 0, 1)
        c = _get_element(matrix, rows, cols, 1, 0)
        d = _get_element(matrix, rows, cols, 1, 1)
        det = a * d - b * c
        result = sign * det
        if do_log:
            submatrix_items = _build_submatrix_items(matrix, rows, cols)
            sign_str = "" if sign == 1 else "-"
            log(
                r"$$ \det%s = %s \cdot %s - %s \cdot %s = %s $$",
                make_latex_matrix(submatrix_items),
                cformat(a, arg_of="*"),
                cformat(d, arg_of="*"),
                cformat(b, arg_of="*"),
                cformat(c, arg_of="*"),
                cformat(result),
            )
        return result

    # For larger matrices, fall back to permutation expansion
    # This shouldn't happen if the Rust optimizer is working correctly
    import itertools
    from .permutation import Permutation

    perms = itertools.permutations(range(n))
    terms = []

    for p_tuple in perms:
        p_list = list(p_tuple)
        sigma = Permutation(p_list)
        perm_sign = sigma.sign()

        term = 1
        for i in range(n):
            term = term * _get_element(matrix, rows, cols, i, p_list[i])
        terms.append(perm_sign * term)

    return sign * multi_add(terms)


def _execute_row_expansion(
    matrix: "Matrix",
    process: "linalg_helper.PyProcess",
    rows: List[int],
    cols: List[int],
    do_log: bool,
    sign: int,
) -> Any:
    """Execute Laplace expansion along a row."""
    n = len(rows)
    expand_row = process.row
    minors = process.minors

    if do_log:
        submatrix_items = _build_submatrix_items(matrix, rows, cols)
        log(r"Provedeme rozvoj determinantu podle %s. řádku:", expand_row + 1)
        log(r"$$ \det%s $$", make_latex_matrix(submatrix_items))

    # Check for zero row
    if not minors:
        if do_log:
            log(r"Řádek %s je nulový, determinant je 0.", expand_row + 1)
        return 0

    terms = []
    term_strs = []
    remaining_rows = [r for i, r in enumerate(rows) if i != expand_row]

    for col_idx, subprocess in minors:
        remaining_cols = [c for i, c in enumerate(cols) if i != col_idx]

        element = _get_element(matrix, rows, cols, expand_row, col_idx)

        # Skip zero elements
        if element == 0:
            continue

        cofactor_sign = (-1) ** (expand_row + col_idx)

        minor_det = execute_process(
            matrix, subprocess, remaining_rows, remaining_cols, do_log=do_log
        )

        term = cofactor_sign * element * minor_det
        terms.append(term)

        if do_log:
            sign_str = "+" if cofactor_sign > 0 else "-"
            minor_items = _build_submatrix_items(matrix, remaining_rows, remaining_cols)
            log(
                r"$$ (-1)^{%s+%s} \cdot a_{%s,%s} \cdot M_{%s,%s} = %s \cdot %s \cdot \det%s = %s \cdot %s = %s $$",
                expand_row + 1,
                col_idx + 1,
                expand_row + 1,
                col_idx + 1,
                expand_row + 1,
                col_idx + 1,
                sign_str,
                cformat(element, arg_of="*"),
                make_latex_matrix(minor_items),
                cformat(element, arg_of="*"),
                cformat(minor_det, arg_of="*"),
                cformat(term),
            )
            term_strs.append(cformat(term, arg_of="+"))

    if not terms:
        return 0

    result = sign * multi_add(terms)

    if do_log:
        terms_sum_str = " + ".join(term_strs)
        log(r"$$ \det = %s = %s $$", terms_sum_str, cformat(result))

    return result


def _execute_col_expansion(
    matrix: "Matrix",
    process: "linalg_helper.PyProcess",
    rows: List[int],
    cols: List[int],
    do_log: bool,
    sign: int,
) -> Any:
    """Execute Laplace expansion along a column."""
    n = len(cols)
    expand_col = process.col
    minors = process.minors

    if do_log:
        submatrix_items = _build_submatrix_items(matrix, rows, cols)
        log(r"Provedeme rozvoj determinantu podle %s. sloupce:", expand_col + 1)
        log(r"$$ \det%s $$", make_latex_matrix(submatrix_items))

    # Check for zero column
    if not minors:
        if do_log:
            log(r"Sloupec %s je nulový, determinant je 0.", expand_col + 1)
        return 0

    terms = []
    term_strs = []
    remaining_cols = [c for i, c in enumerate(cols) if i != expand_col]

    for row_idx, subprocess in minors:
        remaining_rows = [r for i, r in enumerate(rows) if i != row_idx]

        element = _get_element(matrix, rows, cols, row_idx, expand_col)

        # Skip zero elements
        if element == 0:
            continue

        cofactor_sign = (-1) ** (row_idx + expand_col)

        minor_det = execute_process(
            matrix, subprocess, remaining_rows, remaining_cols, do_log=do_log
        )

        term = cofactor_sign * element * minor_det
        terms.append(term)

        if do_log:
            sign_str = "+" if cofactor_sign > 0 else "-"
            minor_items = _build_submatrix_items(matrix, remaining_rows, remaining_cols)
            log(
                r"$$ (-1)^{%s+%s} \cdot a_{%s,%s} \cdot M_{%s,%s} = %s \cdot %s \cdot \det%s = %s \cdot %s = %s $$",
                row_idx + 1,
                expand_col + 1,
                row_idx + 1,
                expand_col + 1,
                row_idx + 1,
                expand_col + 1,
                sign_str,
                cformat(element, arg_of="*"),
                make_latex_matrix(minor_items),
                cformat(element, arg_of="*"),
                cformat(minor_det, arg_of="*"),
                cformat(term),
            )
            term_strs.append(cformat(term, arg_of="+"))

    if not terms:
        return 0

    result = sign * multi_add(terms)

    if do_log:
        terms_sum_str = " + ".join(term_strs)
        log(r"$$ \det = %s = %s $$", terms_sum_str, cformat(result))

    return result


def czech_enumeration_join(l: list[str]) -> str:
    if len(l) == 0:
        return ""
    nonlast = l[:-1]
    joiner = " a " if len(nonlast) > 0 else ""
    return ", ".join(nonlast) + joiner + l[-1]


def _execute_block_triangular(
    matrix: "Matrix",
    process: "linalg_helper.PyProcess",
    rows: List[int],
    cols: List[int],
    do_log: bool,
    sign: int,
) -> Any:
    """Execute block triangular decomposition."""
    blocks = process.blocks
    row_perm = process.row_perm
    col_perm = process.col_perm

    # Map permutation indices back to actual row/col indices
    actual_row_perm = [rows[i] for i in row_perm]
    actual_col_perm = [cols[i] for i in col_perm]

    if do_log:
        submatrix_items = _build_submatrix_items(matrix, rows, cols)

        rc = RowColPermutation(row_perm, col_perm)
        perm, t = rc.try_tranpose()
        rp, cp = perm.to_rows_cols_permutations()

        steps = []
        if t:
            steps.append("transpozicí")
        if not rp.is_id():
            if transpose := rp.try_get_one_transpose():
                val = pcformat(
                    "prohozením řádků $%s$ a $%s$", transpose[0] + 1, transpose[1] + 1
                )
            else:
                val = pcformat("permutací řádků $%s$", rp)
            steps.append(val)
        if not cp.is_id():
            if transpose := cp.try_get_one_transpose():
                val = pcformat(
                    "prohozením sloupců  $%s$ a $%s$",
                    transpose[0] + 1,
                    transpose[1] + 1,
                )
            else:
                val = pcformat("permutací sloupců  $%s$", cp)
            steps.append(val)

        ut = all([_get_process_size(block) == 1 for block in blocks])
        tvar = "horního trojúhelníkového" if ut else "horního blokově trojúhelníkového"

        log("Matici %s převedeme do %s tvaru:", czech_enumeration_join(steps), tvar)

        # Build permuted matrix for display
        permuted_items = _build_submatrix_items(
            matrix, actual_row_perm, actual_col_perm
        )
        log(r"$$ %s $$", make_latex_matrix(permuted_items))
        log(r"Determinant je roven součinu determinantů diagonálních bloků.")

    block_dets = []
    offset = 0

    for i, block_process in enumerate(blocks):
        # Determine block size from the process
        block_size = _get_process_size(block_process)

        block_rows = actual_row_perm[offset : offset + block_size]
        block_cols = actual_col_perm[offset : offset + block_size]

        # Only log non-trivial blocks (size > 1)
        should_log_block = do_log and block_size > 1

        if should_log_block:
            block_items = _build_submatrix_items(matrix, block_rows, block_cols)
            log(r"Blok $B_{%s}$:", i + 1)
            log(r"$$ B_{%s} = %s $$", i + 1, make_latex_matrix(block_items))

        block_det = execute_process(
            matrix, block_process, block_rows, block_cols, do_log=should_log_block
        )
        block_dets.append(block_det)

        if should_log_block:
            log(r"$$ \det(B_{%s}) = %s $$", i + 1, cformat(block_det))

        offset += block_size

    result = sign * multi_mul(block_dets)

    if do_log:
        mul_str = r" \cdot ".join([cformat(d, arg_of="*") for d in block_dets])
        log(
            r"$$ \det = \prod_{i=1}^{%s} \det(B_i) = %s = %s $$",
            len(blocks),
            mul_str,
            cformat(result),
        )

    return result


def _get_process_size(process: "linalg_helper.PyProcess") -> int:
    """Determine the matrix size that a process operates on."""
    process_type = process.process_type

    if process_type == "Direct":
        return process.size or 0
    elif process_type == "RowExpansion":
        minors = process.minors
        if minors:
            # Size is 1 + size of any minor
            return 1 + _get_process_size(minors[0][1])
        return 1
    elif process_type == "ColExpansion":
        minors = process.minors
        if minors:
            return 1 + _get_process_size(minors[0][1])
        return 1
    elif process_type == "BlockTriangular":
        blocks = process.blocks
        return sum(_get_process_size(b) for b in blocks)
    elif process_type == "AddRow":
        return _get_process_size(process.result)
    else:
        return 0


def _is_polynomial(value: Any) -> bool:
    """Check if a value is a Polynomial."""
    from .polynomial import Polynomial

    return isinstance(value, Polynomial)


def _polynomial_to_sympy(poly: Any) -> Any:
    """Convert a Polynomial to a sympy expression."""
    from .polynomial import Polynomial
    import sympy

    if not isinstance(poly, Polynomial):
        return poly

    x = sympy.Symbol(poly.var)
    expr = 0
    for exp, coef in poly.powers.items():
        expr += coef * (x**exp)
    return expr


def _sympy_to_polynomial(expr: Any, var: str = r"\lambda") -> Any:
    """Convert a sympy polynomial expression back to a Polynomial."""
    from .polynomial import Polynomial
    import sympy

    # If it's just a number, return it directly
    if not hasattr(expr, "free_symbols") or not expr.free_symbols:
        return expr

    # Get the symbol
    symbols = list(expr.free_symbols)
    if len(symbols) != 1:
        # Multiple symbols, can't convert to our simple Polynomial
        return expr

    x = symbols[0]
    poly = sympy.Poly(expr, x)

    # Convert to our Polynomial format
    powers = {}
    for monom, coef in poly.as_dict().items():
        exp = monom[0]  # For univariate, it's a tuple with one element
        powers[exp] = coef

    return Polynomial(powers, var)


def _polynomial_safe_divide(numerator: Any, denominator: Any) -> Any:
    """
    Safely divide two values that may be Polynomials.

    Converts to sympy, performs exact polynomial division, and converts back.
    """
    from .polynomial import Polynomial
    import sympy

    # Determine the variable name for the result
    var = r"\lambda"
    if isinstance(numerator, Polynomial):
        var = numerator.var
    elif isinstance(denominator, Polynomial):
        var = denominator.var

    # Convert both to sympy
    num_sympy = _polynomial_to_sympy(numerator)
    denom_sympy = _polynomial_to_sympy(denominator)

    # Perform the division
    # Use cancel to simplify, then check if result is a polynomial
    result = sympy.cancel(num_sympy / denom_sympy)

    # Try to convert back to Polynomial if the result is a polynomial
    try:
        return _sympy_to_polynomial(sympy.expand(result), var)
    except Exception:
        # If conversion fails, return the sympy expression
        return result


def _execute_add_row(
    matrix: "Matrix",
    process: "linalg_helper.PyProcess",
    rows: List[int],
    cols: List[int],
    do_log: bool,
    sign: int,
) -> Any:
    """
    Execute an AddRow operation.

    This creates a modified view of the matrix where dst row has been
    modified by adding a multiple of src row to eliminate the pivot column.

    For polynomial elements, we avoid division by instead:
    - Multiplying dst row by src_pivot
    - Subtracting dst_pivot times src row
    - Dividing the final result by src_pivot
    """
    src = process.src
    dst = process.dst
    pivot_col = process.pivot_col
    result_process = process.result

    # Get the values needed for the row operation
    src_pivot = _get_element(matrix, rows, cols, src, pivot_col)
    dst_pivot = _get_element(matrix, rows, cols, dst, pivot_col)

    if src_pivot == 0:
        raise ValueError("AddRow: source pivot is zero")

    # Check if we need polynomial-safe operations (no division)
    use_polynomial_method = _is_polynomial(src_pivot) or _is_polynomial(dst_pivot)

    from copy import deepcopy

    modified_items = deepcopy(matrix.items)
    n_cols = len(cols)

    if do_log:
        submatrix_items = _build_submatrix_items(matrix, rows, cols)
        log(r"Úprava matice řádkovými operacemi:")
        log(r"$$ %s $$", make_latex_matrix(submatrix_items))

    if use_polynomial_method:
        # Polynomial-safe method: multiply dst by src_pivot, then subtract dst_pivot * src
        # This avoids polynomial division
        # Result: dst_new[j] = src_pivot * dst[j] - dst_pivot * src[j]
        # The determinant gets multiplied by src_pivot, so we divide at the end

        if do_log:
            log(
                r"Eliminace ve sloupci %s: $R_{%s} \leftarrow %s \cdot R_{%s} - %s \cdot R_{%s}$",
                pivot_col + 1,
                dst + 1,
                cformat(src_pivot, arg_of="*"),
                dst + 1,
                cformat(dst_pivot, arg_of="*"),
                src + 1,
            )

        for j in range(n_cols):
            src_val = matrix.items[rows[src]][cols[j]]
            dst_val = matrix.items[rows[dst]][cols[j]]
            modified_items[rows[dst]][cols[j]] = (
                src_pivot * dst_val - dst_pivot * src_val
            )

        # Create a temporary matrix-like object for the recursion
        class ModifiedMatrix:
            def __init__(self, items):
                self.items = items
                self.rows = len(items)
                self.cols = len(items[0]) if items else 0

        modified_matrix = ModifiedMatrix(modified_items)

        if do_log:
            new_submatrix_items = _build_submatrix_items(modified_matrix, rows, cols)
            log(r"Po úpravě:")
            log(r"$$ %s $$", make_latex_matrix(new_submatrix_items))

        # Compute sub-determinant (which is multiplied by src_pivot)
        check_sparsity(modified_matrix, result_process.expected_nonzeros, rows, cols)
        sub_det = execute_process(
            modified_matrix, result_process, rows, cols, do_log, sign
        )

        # Divide out the extra factor
        # For polynomials, we need to use polynomial division or sympy
        if do_log:
            log(
                r"Dělíme výsledek faktorem $%s$ z úpravy řádku.",
                cformat(src_pivot),
            )

        result = _polynomial_safe_divide(sub_det, src_pivot)
        return result

    else:
        # Standard method: compute scalar and add rows
        scalar = -dst_pivot / src_pivot

        if do_log:
            log(
                r"Přičteme $%s$-násobek řádku %s k řádku %s (eliminace ve sloupci %s):",
                cformat(scalar),
                src + 1,
                dst + 1,
                pivot_col + 1,
            )

        for j in range(n_cols):
            src_val = matrix.items[rows[src]][cols[j]]
            dst_val = matrix.items[rows[dst]][cols[j]]
            modified_items[rows[dst]][cols[j]] = dst_val + scalar * src_val

        # Create a temporary matrix-like object for the recursion
        class ModifiedMatrix:
            def __init__(self, items):
                self.items = items
                self.rows = len(items)
                self.cols = len(items[0]) if items else 0

        modified_matrix = ModifiedMatrix(modified_items)

        if do_log:
            new_submatrix_items = _build_submatrix_items(modified_matrix, rows, cols)
            log(r"Po úpravě:")
            log(r"$$ %s $$", make_latex_matrix(new_submatrix_items))

        check_sparsity(modified_matrix, result_process.expected_nonzeros, rows, cols)
        return execute_process(
            modified_matrix, result_process, rows, cols, do_log, sign
        )


def determinant(matrix: "Matrix", do_log: bool = True) -> Any:
    """
    Compute the determinant of a matrix using the optimal process.

    Args:
        matrix: The matrix to compute the determinant of.
        do_log: Whether to log computation steps.

    Returns:
        The determinant value.
    """
    if matrix.rows != matrix.cols:
        raise ValueError("Determinant requires a square matrix")

    n = matrix.rows
    if n == 0:
        if do_log:
            log(r"$\det([]) = 1$")
        return 1

    if do_log:
        log(r"Výpočet determinantu matice:")
        log(r"$$ \det%s $$", make_latex_matrix(matrix.items))

    # Find optimal process using Rust
    cost, process = find_optimal_process(matrix)

    if do_log:
        log(
            r"Optimální strategie: %s operací (%s násobení, %s sčítání)",
            cost.total,
            cost.multiplications,
            cost.additions,
        )

    # Execute the process
    result = execute_process(matrix, process, do_log=do_log)

    # execute_process already logs the result in a readable way
    # if do_log:
    #     log(r"$$ \boxed{\det = %s} $$", cformat(result))

    return result
