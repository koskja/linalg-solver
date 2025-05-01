from typing import Iterator, List, Tuple

from fmt import cformat, pcformat
from log import LoggerGuard, ignore_log, log, nest_appending_logger, nest_logger


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
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match")
        intermediate_slots = [
            [pcformat(r"%s + %s", item1, item2) for item1, item2 in zip(row1, row2)]
            for row1, row2 in zip(self.items, other.items)
        ]

        res = [[0] * self.cols for _ in range(self.rows)]
        logs = []
        for i, j in self.inorder_slot_iter():
            with nest_appending_logger(logs):
                res[i][j] = self.items[i][j] + other.items[i][j]
        res = Matrix(res)
        log(
            r"$$ %s + %s = %s = %s $$",
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
