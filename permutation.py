from random import shuffle
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from linalg import Matrix


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

    def permutation_matrix(self) -> "Matrix":
        n = len(self.perm)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][self.perm[i]] = 1
        return Matrix(matrix)

    def cformat(self) -> str:
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
