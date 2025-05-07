from copy import deepcopy
from typing import Dict, Iterator, List, Tuple, Any
import itertools
import sympy

from fmt import *
from log import log, nest_appending_logger
from permutation import Permutation
from polynomial import Polynomial


class Matrix:
    items: List[List[Any]]

    def __init__(self, items: List[List[Any]]):
        if not items:
            raise ValueError("Matrix cannot be empty")
        if not all(isinstance(row, list) for row in items):
            raise ValueError("Matrix items must be a list of lists")
        row_len = None
        if items:
            if not items[0]:
                if any(row for row in items):
                    raise ValueError("Matrix rows cannot be empty if columns exist")
                row_len = 0
            else:
                row_len = len(items[0])
                if not all(len(row) == row_len for row in items):
                    raise ValueError("All matrix rows must have the same length")
        else:
            row_len = 0

        self._cols = row_len if row_len is not None else (len(items[0]) if items else 0)

        self.items = items

    def __str__(self) -> str:
        return "\n".join([" ".join([str(item) for item in row]) for row in self.items])

    def cformat(self, _arg_of="") -> str:
        return make_latex_matrix(self.items)

    @property
    def rows(self) -> int:
        return len(self.items)

    @property
    def cols(self) -> int:
        if self.rows == 0:
            return self._cols
        return len(self.items[0])

    def get_row(self, i: int) -> List[any]:
        return self.items[i]

    def get_col(self, j: int) -> List[any]:
        return [row[j] for row in self.items]

    def inorder_slot_iter(self) -> Iterator[Tuple[int, int]]:
        for i in range(self.rows):
            for j in range(self.cols):
                yield (i, j)

    def __add__(self, other: "Matrix") -> "Matrix":
        return self.multi_add(other)

    def multi_add(self, *others: "Matrix") -> "Matrix":
        items = [self] + list(others)
        for i, item in enumerate(items):
            if item.rows != self.rows or item.cols != self.cols:
                raise ValueError(f"Matrix dimensions must match; mismatch at item {i}")
        res = Matrix([[0] * self.cols for _ in range(self.rows)])
        intermediate_slots = [[""] * self.cols for _ in range(self.rows)]
        logs = []
        for i, j in self.inorder_slot_iter():
            with nest_appending_logger(logs):
                intermediate_slots[i][j] = " + ".join(
                    [cformat(item.items[i][j], arg_of="+") for item in items]
                )
                res.items[i][j] = multi_add([item.items[i][j] for item in items])
        log(
            r"$$ %s = %s $$",
            make_latex_matrix(intermediate_slots),
            res,
        )
        if logs:
            log(r"with substeps: \\")
            for l in logs:
                log(r"%s \\", l)
        return res

    def scalar_mul(self, scalar: any) -> "Matrix":
        return Matrix([[item * scalar for item in row] for row in self.items])

    def __neg__(self) -> "Matrix":
        return self.scalar_mul(-1)

    def __sub__(self, other: "Matrix") -> "Matrix":
        return self + (-other)

    def __mul__(self, other) -> "Matrix":
        if not isinstance(other, Matrix):
            return self.scalar_mul(other)
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions must match")
        res = Matrix([[0] * other.cols for _ in range(self.rows)])
        intermediate_slots = [[""] * other.cols for _ in range(self.rows)]
        logs = []
        for i in range(self.rows):
            for j in range(other.cols):
                with nest_appending_logger(logs):
                    intermediate_slots[i][j] = " + ".join(
                        [
                            cformat(self.items[i][k], arg_of="*")
                            + " \\times "
                            + cformat(other.items[k][j], arg_of="*")
                            for k in range(self.cols)
                        ]
                    )
                    res.items[i][j] = multi_add(
                        [self.items[i][k] * other.items[k][j] for k in range(self.cols)]
                    )
        log(
            r"$$ %s \times %s = %s = %s $$",
            self,
            other,
            make_latex_matrix(intermediate_slots),
            res,
        )
        if logs:
            log(r"with substeps: \\")
            for l in logs:
                log(r"%s \\", l)
        return res

    def determinant(self, log_permutation_details: bool = False) -> Any:
        if self.rows != self.cols:
            raise ValueError("Determinant requires a square matrix")
        n = self.rows
        if n == 0:
            log(r"$$ \\det([]) = 1 $$ ")
            return 1

        perms = itertools.permutations(range(n))
        det_terms = []
        log_lines = []

        for p_tuple in perms:
            p_list = list(p_tuple)
            sigma = Permutation(p_list)
            sign = sigma.sign()

            term_prod = 1
            log_prod_term_elements = []
            prod_logs = []
            with nest_appending_logger(prod_logs):
                for i in range(n):
                    item = self.items[i][sigma(i)]
                    term_prod = term_prod * item
                    log_prod_term_elements.append(cformat(item, arg_of="*"))

            term_value = sign * term_prod
            det_terms.append(term_value)

            if (
                n <= 6
                and term_value != 0
                and not (
                    isinstance(term_value, Polynomial)
                    and all(coef == 0 for coef in term_value.powers.values())
                )
            ):
                sign_str = "+" if sign == 1 else "-"
                prod_str = r"\times ".join(log_prod_term_elements)
                term_contribution_str = pcformat(r"%s(%s)", sign_str, prod_str)

                if log_permutation_details:
                    # Build the log line step-by-step to avoid complex f-string escaping
                    perm_str = sigma.cformat()
                    log_line = r"%s & \qquad %s \\" % (perm_str, term_contribution_str)
                    log_lines.append(log_line)
                else:
                    # Original format
                    log_lines.append(term_contribution_str)

        sum_logs = []
        with nest_appending_logger(sum_logs):
            total_det = multi_add(det_terms)

        if n <= 6:
            # log_prefix = r"$$ \det%s = \sum_{\sigma \in S_%s} \text{sgn}(\sigma) \prod_{i=1}^{%s} A_{i, \sigma(i)}" % (self.cformat(), n, n)
            log_prefix = r"$$ \det%s" % (self.cformat())
            if log_permutation_details:
                log(r"%s = \begin{aligned}" % log_prefix)
                log(
                    r"\sigma \in S_{%s} & \qquad \text{sgn}(\sigma) \prod A_{i, \sigma(i)} \\"
                    % n
                )
                log(r"\hline")
                for line in log_lines:
                    log(line)
                log(r"\end{aligned} $$")
                log(r"$$ = %s $$" % cformat(total_det))
            else:
                log(
                    r"%s = %s = %s $$ "
                    % (log_prefix, " ".join(log_lines), cformat(total_det))
                )

            if sum_logs:
                log(r"with summation substeps: \\")
                for l in sum_logs:
                    log(r"%s \\", l)
        else:
            log(r"$$ \det(%s) = %s $$ ", self.cformat(), cformat(total_det))

        return total_det

    def to_block_matrix(self, row_splits: List[int], col_splits: List[int]) -> "Matrix":
        if not all(0 < split < self.rows for split in row_splits):
            raise ValueError(
                "Row splits must be within matrix dimensions (exclusive of 0 and rows)"
            )
        if not all(0 < split < self.cols for split in col_splits):
            raise ValueError(
                "Column splits must be within matrix dimensions (exclusive of 0 and cols)"
            )

        row_splits = sorted(list(set([0] + row_splits + [self.rows])))
        col_splits = sorted(list(set([0] + col_splits + [self.cols])))

        block_matrix = []
        for i in range(len(row_splits) - 1):
            row_start, row_end = row_splits[i], row_splits[i + 1]
            block_row = []
            for j in range(len(col_splits) - 1):
                col_start, col_end = col_splits[j], col_splits[j + 1]
                sub_items = [
                    row[col_start:col_end] for row in self.items[row_start:row_end]
                ]
                block_row.append(Matrix(sub_items))
            block_matrix.append(block_row)

        return Matrix(block_matrix)

    @classmethod
    def from_block_matrix(cls, blocks: List[List["Matrix"]] | "Matrix") -> "Matrix":
        if isinstance(blocks, Matrix):
            blocks = blocks.items
        if not blocks or not blocks[0]:
            return cls([[]])

        num_block_rows = len(blocks)
        num_block_cols = len(blocks[0])
        if not all(len(row) == num_block_cols for row in blocks):
            raise ValueError("All block rows must have the same number of blocks")

        col_widths = [blocks[0][j].cols for j in range(num_block_cols)]
        for i in range(1, num_block_rows):
            for j in range(num_block_cols):
                if blocks[i][j].cols != col_widths[j]:
                    raise ValueError(f"Inconsistent column width in block column {j}")

        row_heights = [blocks[i][0].rows for i in range(num_block_rows)]
        for i in range(num_block_rows):
            for j in range(1, num_block_cols):
                if blocks[i][j].rows != row_heights[i]:
                    raise ValueError(f"Inconsistent row height in block row {i}")

        new_items = []
        for i in range(num_block_rows):
            current_block_row_height = row_heights[i]
            for item_row_idx in range(current_block_row_height):
                new_row = []
                for j in range(num_block_cols):
                    new_row.extend(blocks[i][j].items[item_row_idx])
                new_items.append(new_row)

        return cls(new_items)

    @classmethod
    def zero(cls, rows: int, cols: int) -> "Matrix":
        return cls([[0] * cols for _ in range(rows)])

    @classmethod
    def identity(cls, size: int) -> "Matrix":
        return cls([[1 if i == j else 0 for j in range(size)] for i in range(size)])

    @classmethod
    def diagonal(cls, items: List[any] | any) -> "Matrix":
        res = cls.zero(len(items), len(items))
        for i, item in enumerate(items):
            res.items[i][i] = item
        return res

    def eigenvalues(self, real_only: bool = False) -> Dict[any, int]:
        if self.rows != self.cols:
            raise ValueError("Eigenvalues require a square matrix")
        n = self.rows
        lmbda = Polynomial({1: 1}, var=r"\lambda")
        lambda_identity = Matrix.diagonal([lmbda for _ in range(n)])
        logs = []
        with nest_appending_logger(logs):
            A_minus_lambda_I = self - lambda_identity
        log(
            r"Compute the characteristic matrix $A - \lambda I$: $$ A - \lambda I = %s - %s = %s $$",
            self,
            lambda_identity,
            A_minus_lambda_I,
        )

        log(r"Compute the characteristic polynomial $\det(A - \lambda I)$:")
        characteristic_poly = A_minus_lambda_I.determinant(log_permutation_details=True)
        log(
            r"The characteristic polynomial is: $$ p(\lambda) = %s $$",
            characteristic_poly,
        )

        # Log a factored form using Polynomial.factor_roots
        roots = characteristic_poly.radical_roots()
        if real_only:
            # Filter roots to only real values
            real_roots = {
                root: mult
                for root, mult in roots.items()
                if getattr(root, "is_real", None) is True
                or (isinstance(root, (int, float)) and not isinstance(root, bool))
            }
            roots = real_roots
        if roots:
            factors_dict = characteristic_poly.factor_roots(list(roots.items()))
            factors = []
            for factor_poly, mult in factors_dict.items():
                if mult == 1:
                    factors.append(cformat(factor_poly, arg_of="*"))
                else:
                    factors.append(
                        r"%s^{%d}" % (cformat(factor_poly, arg_of="^"), mult)
                    )
            factored_str = r" \times ".join(factors)
            log(r"A factored form: $$ p(\lambda) = %s $$", factored_str)

        eigenvalues_log_str = ", ".join(
            [f"${cformat(root)}$ (multiplicity {mult})" for root, mult in roots.items()]
        )
        field = "R" if real_only else "C"
        log(
            r"The eigenvalues (roots of $p(\lambda)$ in $\mathbb{%s}$) with their algebraic multiplicities are: %s",
            field,
            eigenvalues_log_str,
        )
        return roots

    @classmethod
    def new_vector(cls, items: List[any]) -> "Matrix":
        return cls([[i] for i in items])

    class AffineSubspace:
        def __init__(self, vec: List[any], mat: "Matrix"):
            self.vec = vec
            self.generators = mat

        def get_one(self) -> List[any]:
            return self.vec

        def cformat(self, arg_of="") -> str:
            if (
                self.generators is None
                or self.generators.rows == 0
                or self.generators.cols == 0
            ):
                return r" %s " % cformat(Matrix.new_vector(self.vec))
            all_zeros = all(v == 0 for v in self.vec)
            generators = []
            for i in range(self.generators.cols):
                generators.append(
                    cformat(Matrix.new_vector(self.generators.get_col(i)))
                )
            span = r" \text{span} \hspace{0.1em} \left\{ %s \right\} " % ", ".join(
                generators
            )
            return r" %s %s  " % (
                cformat(Matrix.new_vector(self.vec)) + " + " if not all_zeros else "",
                span,
            )

    class NoSolution:
        def __init__(self):
            pass

        def __repr__(self):
            return "NoSolution()"

        def cformat(self, arg_of=""):
            return r"\text{No solution}"

    def _q_find_preimage_of(self, vec: List[any]) -> "AffineSubspace | NoSolution":
        import sympy

        # Convert self.items and vec to sympy matrices
        A = sympy.Matrix(self.items)
        b = sympy.Matrix(vec)
        # Try to solve the system
        sol = sympy.linsolve((A, b))
        if not sol:
            return Matrix.NoSolution()
        # linsolve returns a FiniteSet of tuples (possibly with parameters)
        sol = list(sol)
        if not sol:
            return Matrix.NoSolution()
        # Take the first solution tuple
        s = sol[0]
        # If the solution is fully numeric, return as a single point
        if all(not hasattr(x, "free_symbols") or len(x.free_symbols) == 0 for x in s):
            return Matrix.AffineSubspace(list(s), Matrix.zero(len(s), 0))
        # Otherwise, extract particular solution and generators
        # Find all parameters (free symbols)
        params = set()
        for x in s:
            if hasattr(x, "free_symbols"):
                params |= x.free_symbols
        params = sorted(params, key=lambda x: str(x))
        # Build the particular solution (set all params to 0)
        subs = {p: 0 for p in params}
        particular = [x.subs(subs) for x in s]
        # Build generators: for each param, set that param to 1, others to 0
        generators = []
        for i, p in enumerate(params):
            subs = {q: 0 for q in params}
            subs[p] = 1
            gen = [x.subs(subs) - x.subs({q: 0 for q in params}) for x in s]
            generators.append(gen)
        if generators:
            gen_mat = Matrix([list(col) for col in zip(*generators)])
        else:
            gen_mat = Matrix.zero(len(s), 0)
        return Matrix.AffineSubspace(particular, gen_mat)

    def row_reduce(self, bar_col: int = None):
        """
        Perform Gaussian elimination (row reduction) on an augmented matrix.
        Returns the reduced matrix, pivot information, and logs.
        """
        # bar_col is the first column after the bar (the augmented column)
        from copy import deepcopy
        from fmt import make_latex_augmented_matrix

        A = deepcopy(self.items)
        m, n = len(A), len(A[0])
        pivot_i, pivot_j = 0, 0
        pivots = []
        bar_col = bar_col or n - 1
        intermediate_matrices = [make_latex_augmented_matrix(A, bar_col=bar_col)]
        intermediate_steps = []
        step = 0
        while pivot_i < m and pivot_j < bar_col:
            if A[pivot_i][pivot_j] == 0:
                swapped = False
                for i in range(pivot_i + 1, m):
                    if A[i][pivot_j] != 0:
                        A[pivot_i], A[i] = A[i], A[pivot_i]
                        intermediate_matrices.append(
                            make_latex_augmented_matrix(A, bar_col=bar_col)
                        )
                        intermediate_steps.append(
                            r"\textbf{S%s}: Swap rows $R_{%d}$ and $R_{%d}$"
                            % (step, pivot_i + 1, i + 1)
                        )
                        step += 1
                        swapped = True
                        break
                if not swapped:
                    pivot_j += 1
                    continue
            # Normalize pivot row
            factor = A[pivot_i][pivot_j]
            normalized = False
            if factor != 1:
                for j in range(pivot_j, n):
                    old_val = A[pivot_i][j]
                    A[pivot_i][j] = A[pivot_i][j] / factor
                    normalized = normalized or A[pivot_i][j] != old_val
            if normalized:
                intermediate_matrices.append(
                    make_latex_augmented_matrix(A, bar_col=bar_col)
                )
                intermediate_steps.append(
                    r"\textbf{N%s}: Normalize pivot row %s" % (step, pivot_i + 1)
                )
                step += 1
            # Eliminate entries below pivot
            first_nonzero_row = None
            eliminated = False
            for k in range(pivot_i + 1, m):
                factor = A[k][pivot_j]
                if factor == 0:
                    continue
                if first_nonzero_row is None:
                    first_nonzero_row = k
                for j in range(pivot_j, n):
                    old_val = A[k][j]
                    A[k][j] = A[k][j] - factor * A[pivot_i][j]
                    eliminated = eliminated or A[k][j] != old_val
            if first_nonzero_row is not None and eliminated:
                intermediate_matrices.append(
                    make_latex_augmented_matrix(A, bar_col=bar_col)
                )
                intermediate_steps.append(
                    r"\textbf{E%s}: Eliminate entries below pivot in column %s"
                    % (step, pivot_j + 1)
                )
                step += 1
            pivots.append((pivot_i, pivot_j))
            pivot_i += 1
            pivot_j += 1
        # Reverse elimination (above pivots)
        for idx in reversed(range(len(pivots))):
            row, col = pivots[idx]
            eliminated = False
            for k in range(row):
                factor = A[k][col]
                if factor == 0:
                    continue
                for j in range(col, n):
                    old_val = A[k][j]
                    A[k][j] = A[k][j] - factor * A[row][j]
                    eliminated = eliminated or A[k][j] != old_val
            if eliminated:
                intermediate_matrices.append(
                    make_latex_augmented_matrix(A, bar_col=bar_col)
                )
                intermediate_steps.append(
                    r"\textbf{E%s}: Eliminate above pivot in column %s"
                    % (step, col + 1)
                )
                step += 1
        return A, pivots, intermediate_matrices, intermediate_steps

    def _check_inconsistency(self, reduced_items, n, bar_col, log_fn=None):
        """
        Check for inconsistency in the reduced augmented matrix.
        Returns True if inconsistent, otherwise False. Optionally logs details.
        """
        from fmt import make_latex_augmented_matrix

        m = len(reduced_items)
        for i in range(m):
            if (
                all(abs(reduced_items[i][j]) == 0 for j in range(n))
                and abs(reduced_items[i][bar_col]) != 0
            ):
                if log_fn:
                    row_matrix = Matrix([reduced_items[i]])
                    log_fn(
                        r"\textbf{Inconsistent row detected (row %s):} $ %s $",
                        i + 1,
                        make_latex_augmented_matrix(row_matrix.items, bar_col=bar_col),
                    )
                    log_fn(
                        r"\[ \boxed{\text{The system is inconsistent: no solution.}} \]"
                    )
                return True
        return False

    def _extract_affine_subspace(self, reduced_items, pivots, n, bar_col, log_fn=None):
        """
        Given a reduced augmented matrix and pivots, extract the particular solution and nullspace generators.
        Optionally logs details.
        """
        from fmt import make_latex_vector

        m = len(reduced_items)
        pivots_row = [-1] * m  # pivot column for each row, -1 if none
        pivot_cols = set()
        for i, (row, col) in enumerate(pivots):
            pivots_row[row] = col
            pivot_cols.add(col)
        free_vars = [j for j in range(n) if j not in pivot_cols]
        if log_fn:
            log_fn(
                r"\textbf{Pivot columns:} $ %s$",
                ", ".join([f"x_{{{j+1}}}" for j in sorted(pivot_cols)]),
            )
            log_fn(
                r"\textbf{Free variables:} $ %s$",
                ", ".join([f"x_{{{j+1}}}" for j in free_vars]),
            )
        # Build particular solution (all free vars = 0)
        particular = [0] * n
        for i in range(m):
            if pivots_row[i] != -1:
                j = pivots_row[i]
                rhs = reduced_items[i][bar_col]
                # Subtract free var contributions (all zero for particular)
                particular[j] = rhs
        if log_fn:
            log_fn(
                r"\textbf{Particular solution (free vars = 0):} \[ %s \]",
                make_latex_vector(particular),
            )
        # Build nullspace generators (one for each free var)
        generators = []
        for idx, free_j in enumerate(free_vars):
            gen = [0] * n
            gen[free_j] = 1
            for i in range(m):
                if pivots_row[i] != -1:
                    j = pivots_row[i]
                    # The coefficient of free_j in row i
                    coeff = -reduced_items[i][free_j]
                    gen[j] = coeff
            if log_fn:
                log_fn(
                    r"\textbf{Nullspace generator for $x_{%s}$:} \[ %s \]",
                    free_j + 1,
                    make_latex_vector(gen),
                )
            generators.append(gen)
        if generators:
            gen_mat = Matrix([list(col) for col in zip(*generators)])
            if log_fn:
                log_fn(
                    r"\textbf{Generator matrix (nullspace basis):} \[ %s \]",
                    gen_mat.cformat(),
                )
        else:
            gen_mat = None
        return particular, gen_mat

    def find_preimage_of(
        self,
        vec: List[any],
        log_matrices: bool = False,
        log_steps: bool = False,
        log_result: bool = False,
    ) -> "AffineSubspace | NoSolution":
        """
        Returns the affine subspace of solutions to self * x = vec, or Matrix.NoSolution() if inconsistent.
        """
        if self.rows != len(vec):
            raise ValueError("Matrix dimensions must match")
        # If no logging, quietly use sympy
        if not log_matrices and not log_steps and not log_result:
            return self._q_find_preimage_of(vec)
        from fmt import make_latex_augmented_matrix
        from copy import deepcopy

        # Build augmented matrix
        A = deepcopy(self)
        for i in range(A.rows):
            A.items[i].append(vec[i])
        bar_col = A.cols - 1
        # Use row_reduce with logging
        reduced_items, pivots, intermediate_matrices, intermediate_steps = Matrix(
            A.items
        ).row_reduce(bar_col=bar_col)
        m, n_aug = len(reduced_items), len(reduced_items[0])
        n = n_aug - 1  # number of variables
        # Prepare intermediate matrix logs
        total_cols = 0
        last = []
        out = []
        for matrix in intermediate_matrices:
            total_cols += n_aug
            last.append(matrix)
            if total_cols > 10:
                out.append(last)
                last = [""]
                total_cols = 0
        if last:
            out.append(last)
        if log_matrices:
            log(r"Intermediate matrices:")
            out = [r" $$ " + r" \sim ".join(chunk) + r" $$ \\" for chunk in out]
            for line in out:
                log(r"%s", line)
        if log_steps:
            for step_desc in intermediate_steps:
                log(r"%s ", step_desc)
        logs = []
        with nest_appending_logger(logs):
            # Check for inconsistency
            inconsistent = self._check_inconsistency(
                reduced_items, n, bar_col, log_fn=log
            )
            if inconsistent:
                return Matrix.NoSolution()
            # Extract affine subspace
            particular, gen_mat = self._extract_affine_subspace(
                reduced_items, pivots, n, bar_col, log_fn=log
            )
        if log_result:
            log("\n".join(logs))
        return Matrix.AffineSubspace(particular, gen_mat)

    def inverse(
        self,
        log_matrices: bool = False,
        log_steps: bool = False,
        log_result: bool = False,
    ):
        """
        Returns the inverse of the matrix as a new Matrix, or Matrix.NoSolution() if singular.
        Uses sympy if no logging is requested, otherwise uses row reduction with logging.
        """
        if self.rows != self.cols:
            raise ValueError("Matrix must be square to invert.")
        n = self.rows
        # Fast path: no logging, use sympy
        if not log_matrices and not log_steps and not log_result:
            import sympy

            try:
                inv = sympy.Matrix(self.items).inv()
                return Matrix([list(inv.row(i)) for i in range(inv.rows)])
            except Exception:
                return Matrix.NoSolution()
        # Logging path: use row reduction
        from fmt import make_latex_augmented_matrix, make_latex_matrix
        from copy import deepcopy

        # Build augmented matrix [A | I]
        A = deepcopy(self)
        identity = Matrix.identity(n)
        aug_items = [A.items[i] + identity.items[i] for i in range(n)]
        bar_col = self.cols - 1
        # Use row_reduce with logging
        reduced_items, pivots, intermediate_matrices, intermediate_steps = Matrix(
            aug_items
        ).row_reduce(bar_col=bar_col + 1)
        n_aug = len(reduced_items[0])
        # Prepare intermediate matrix logs
        total_cols = 0
        last = []
        out = []
        for matrix in intermediate_matrices:
            total_cols += n_aug
            last.append(matrix)
            if total_cols > 10:
                out.append(last)
                last = [""]
                total_cols = 0
        if last:
            out.append(last)
        if log_matrices:
            log(r"Intermediate matrices:")
            out = [r" $$ " + r" \sim ".join(chunk) + r" $$ \\" for chunk in out]
            for line in out:
                log(r"%s", line)
        if log_steps:
            for step_desc in intermediate_steps:
                log(r"%s \\ ", step_desc)
        logs = []
        with nest_appending_logger(logs):
            # Check if left block is identity
            is_identity = True
            for i in range(n):
                for j in range(n):
                    if (i == j and abs(reduced_items[i][j] - 1) > 1e-12) or (
                        i != j and abs(reduced_items[i][j]) > 1e-12
                    ):
                        is_identity = False
                        break
                if not is_identity:
                    break
            if not is_identity:
                log(r"\[ \boxed{\text{The matrix is singular: no inverse.}} \]")
                return Matrix.NoSolution()
            # Extract right block as inverse
            inverse_items = [row[n:] for row in reduced_items]
            log(r"\textbf{Inverse matrix:} \[ %s \]", make_latex_matrix(inverse_items))
        if log_result:
            log("\n".join(logs))
        return Matrix(inverse_items)
