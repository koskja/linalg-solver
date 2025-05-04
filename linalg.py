from typing import Iterator, List, Tuple

from fmt import cformat, pcformat
from log import log, nest_appending_logger


def pretty_print_arithmetic(a: any, op: str, b: any) -> str:
    if op == "+":
        if b == 0:
            return pcformat(r"%s", a)
        if a == 0:
            return pcformat(r"%s", b)
        if b < 0:
            b = -b
        return pcformat(r"%s-%s", a, b)
    if op == "-":
        if b == 0:
            return pcformat(r"%s", a)
        if a == 0:
            return pcformat(r"%s", -b)
        if b < 0:
            b = -b
        return pcformat(r"%s+%s", a, b)
    if op == "*":
        if a == 0 or b == 0:
            return pcformat(r"%s", 0)
        if a == 1:
            return pcformat(r"%s", b)
        if b == 1:
            return pcformat(r"%s", a)
        if b < 0:
            b = -b
            a = -a
        return pcformat(r"%s \times %s", a, b)


def make_latex_matrix(items: List[List[any]]) -> str:
    start = r"\begin{pmatrix}"
    end = r"\end{pmatrix}"
    rows = [r" & ".join([cformat(item) for item in row]) for row in items]
    return start + (r"\\" + "\n").join(rows) + end


def multi_add_vargs(*items: List[any]) -> any:
    return multi_add(list(items))


def multi_add(items: List[any]) -> any:
    if len(items) == 0:
        raise ValueError("At least one item is required")
    if len(items) == 1:
        return items[0]
    if hasattr(items[0], "multi_add") and callable(items[0].multi_add):
        return items[0].multi_add(*items[1:])
    return sum(items)


def scalar_mul(item: any, scalar: any) -> any:
    if hasattr(item, "scalar_mul") and callable(item.scalar_mul):
        return item.scalar_mul(scalar)
    return item * scalar


def linear_comb(scalars: List[any], items: List[any]) -> any:
    if len(scalars) != len(items):
        raise ValueError("Scalars and items must have the same length")
    return multi_add([scalar_mul(item, scalar) for scalar, item in zip(scalars, items)])


class Matrix:
    items: List[List[any]]

    def __init__(self, items: List[List[any]]):
        self.items = items

    def __str__(self) -> str:
        return "\n".join([" ".join([str(item) for item in row]) for row in self.items])

    def cformat(self) -> str:
        return make_latex_matrix(self.items)

    @property
    def rows(self) -> int:
        return len(self.items)

    @property
    def cols(self) -> int:
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
                    [cformat(item.items[i][j]) for item in items]
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
        return Matrix([[scalar * item for item in row] for row in self.items])

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
                            cformat(self.items[i][k])
                            + " \\times "
                            + cformat(other.items[k][j])
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
