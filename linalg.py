from typing import Iterator, List, Tuple, Any
import itertools

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

    def eigenvalues(self, real_only: bool = False) -> List[Tuple[any, int]]:
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
