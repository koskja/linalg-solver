from random import shuffle
from typing import List, Optional, Tuple, TYPE_CHECKING, overload


class Permutation:
    def __init__(self, perm: list[int]):
        n = len(perm)
        if set(perm) != set(range(n)):
            raise ValueError("Input list is not a valid permutation of 0..n-1")
        self.perm = perm

    def __call__(self, i: int) -> int:
        return self.perm[i]

    def __len__(self) -> int:
        return len(self.perm)

    def __eq__(self, other) -> bool:
        return self.perm == other.perm

    def __mul__(self, other: "Permutation") -> "Permutation":
        """Composition of permutations: (self * other)(i) = self(other(i))"""
        if len(self) != len(other):
            raise ValueError("Permutations must have same length")
        n = len(self)
        return Permutation([self(other(i)) for i in range(n)])

    @classmethod
    def id(cls, n: int):
        return cls(list(range(n)))

    def is_id(self):
        return self.id(len(self)) == self

    def _get_cycle_decomposition_and_count(self) -> tuple[list[list[int]], int]:
        n = len(self.perm)
        visited = [False] * n
        cycles = []
        num_cycles = 0
        for i in range(n):
            if not visited[i]:
                num_cycles += 1
                cycle = []
                j = i
                while not visited[j]:
                    visited[j] = True
                    cycle.append(j)
                    j = self.perm[j]
                cycles.append(cycle)
        return cycles, num_cycles

    def cycle_decomposition(self) -> list[list[int]]:
        all_cycles, _ = self._get_cycle_decomposition_and_count()
        return [cycle for cycle in all_cycles if len(cycle) > 1]

    def sign(self) -> int:
        n = len(self.perm)
        if n == 0:
            return 1
        _, num_cycles = self._get_cycle_decomposition_and_count()
        if (n - num_cycles) % 2 == 0:
            return 1
        else:
            return -1

    def cformat(self, arg_of=None) -> str:
        cycles = self.cycle_decomposition()
        if not cycles:
            return r"\text{id}"
        formatted_cycles = []
        for cycle in cycles:
            formatted_cycles.append(f"({' '.join(map(lambda x: str(x + 1), cycle))})")
        return "".join(formatted_cycles)

    @staticmethod
    def random(n: int) -> "Permutation":
        perm = list(range(n))
        shuffle(perm)
        return Permutation(perm)

    def cost(self) -> int:
        return sum(len(cycle) - 1 for cycle in self.cycle_decomposition())

    def try_get_one_transpose(self) -> Optional[tuple[int, int]]:
        cd = self.cycle_decomposition()
        c1 = [c for c in cd if len(c) == 2]
        c2 = [c for c in cd if len(c) > 2]
        if len(c1) == 1 and len(c2) == 0:
            return (c1[0][0], c1[0][1])
        return None


class RowColPermutation:
    """Permutation of rows and columns of a matrix. Represents permutation matrices P and Q, applied to A in the fashion of PAQ."""

    def __init__(
        self, row_perm: Permutation | list[int], col_perm: Permutation | list[int]
    ):
        make_perm = lambda val: (
            val if isinstance(val, Permutation) else Permutation(val)
        )
        self.row_perm = make_perm(row_perm)
        self.col_perm = make_perm(col_perm)

    def __len__(self) -> int:
        return len(self.row_perm)

    def __call__(self, i: int, j: int) -> Tuple[int, int]:
        return self.row_perm(i), self.col_perm(j)

    def __mul__(self, other: "RowColPermutation") -> "RowColPermutation":
        return RowColPermutation(
            self.row_perm * other.row_perm, other.col_perm * self.col_perm
        )

    def __eq__(self, other) -> bool:
        return self.row_perm == other.row_perm and self.col_perm == other.col_perm

    @classmethod
    def id(cls, n: int):
        return cls(Permutation.id(n), Permutation.id(n))

    def is_id(self):
        return self.row_perm.is_id() and self.col_perm.is_id()

    @classmethod
    def matrix_transpose(cls, n: int) -> "RowColPermutation":
        """Construct a permutation that transposes a matrix of size n x n"""
        reversed_range = lambda k: list(reversed(range(k)))
        return RowColPermutation(
            Permutation(reversed_range(n)), Permutation(reversed_range(n))
        )

    def with_transpose(self) -> "RowColPermutation":
        """Compose with matrix transpose"""
        return self * self.matrix_transpose(len(self))

    def cost(self) -> int:
        return self.row_perm.cost() + self.col_perm.cost()

    def try_tranpose(self) -> Tuple["RowColPermutation", bool]:
        new = self.with_transpose()
        old_cost = self.cost()
        new_cost = new.cost() + 1  # +1 for the transpose operation
        if new_cost < old_cost:
            return new, True
        else:
            return self, False

    def to_rows_cols_permutations(self) -> Tuple[Permutation, Permutation]:
        return self.row_perm, self.col_perm
