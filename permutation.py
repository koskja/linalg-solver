from random import shuffle

class Permutation:
    def __init__(self, perm: list[int]):
        self.perm = perm
    
    def __call__(self, i: int) -> int:
        return self.perm[i]
    
    def __len__(self) -> int:
        return len(self.perm)

    def cycle_decomposition(self) -> list[list[int]]:
        n = len(self.perm)
        visited = [False] * n
        cycles = []
        for i in range(n):
            if not visited[i]:
                cycle = []
                j = i
                while not visited[j]:
                    visited[j] = True
                    cycle.append(j)
                    j = self.perm[j]
                if len(cycle) > 1: # Only include non-trivial cycles
                    cycles.append(cycle)
        return cycles

    def permutation_matrix(self) -> Matrix:
        n = len(self.perm)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][self.perm[i]] = 1
        return Matrix(matrix)

    def cformat(self) -> str:
        cycles = self.cycle_decomposition()
        if not cycles:
            return "\\text{id}"
        return "".join(f"({' '.join(map(str, cycle))})" for cycle in cycles)
    
    def random(n: int) -> 'Permutation':
        perm = list(range(n))
        shuffle(perm)
        return Permutation(perm)
    
    
