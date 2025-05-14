from copy import deepcopy
from typing import List, Tuple, Callable, Any
import random
from linalg import Matrix
from permutation import Permutation


class RandomMatrixBuilder:
    rank: int | None = None
    eigenvalues: List[Tuple[float, int]] | List[Tuple[float, List[int]]] | None = None
    jordan_blocks: List[Tuple[Any, int]] | None = None
    do_randomize_from_diagonal_form: bool = True
    num_rows: int | None = None
    num_cols: int | None = None
    dist: Callable[[], Any] | None = None

    @classmethod
    def new(cls, **kwargs) -> "RandomMatrixBuilder":
        builder = cls()
        for key, value in kwargs.items():
            setattr(builder, key, value)
        return builder

    def with_size(self, num_rows: int, num_cols: int) -> "RandomMatrixBuilder":
        self.num_rows = num_rows
        self.num_cols = num_cols
        return self

    def with_rank(self, rank: int) -> "RandomMatrixBuilder":
        self.rank = rank
        return self

    def with_dist(self, dist: Callable[[], Any]) -> "RandomMatrixBuilder":
        self.dist = dist
        return self

    def with_eigenvalues(
        self, eigenvalues: List[float] | List[Tuple[float, int]]
    ) -> "RandomMatrixBuilder":
        if isinstance(eigenvalues[0], tuple):
            self.eigenvalues = eigenvalues
        else:
            self.eigenvalues = [(eigenvalue, 1) for eigenvalue in eigenvalues]
        return self

    def with_jordan_blocks(
        self, blocks: List[Tuple[Any, int]]
    ) -> "RandomMatrixBuilder":
        self.jordan_blocks = blocks
        return self

    def is_square(self) -> bool:
        return self.num_rows == self.num_cols

    def assert_requirements(self) -> None:
        if self.eigenvalues is not None:
            assert self.is_square(), "Diagonalizable matrix must be square."
            assert (
                sum(e[1] for e in self.eigenvalues) == self.num_rows
            ), "Sum of eigenvalue multiplicities must match matrix size."
            assert self.rank is None, "Cannot specify both eigenvalues and rank."
            assert (
                self.jordan_blocks is None
            ), "Cannot specify both eigenvalues and Jordan blocks."
        if self.rank is not None:
            assert self.rank <= min(
                self.num_rows, self.num_cols
            ), "Rank cannot exceed min(num_rows, num_cols)."
            assert self.eigenvalues is None, "Cannot specify both rank and eigenvalues."
            assert (
                self.jordan_blocks is None
            ), "Cannot specify both rank and Jordan blocks."
        if self.jordan_blocks is not None:
            assert self.is_square(), "Jordan block matrix must be square."
            assert (
                sum(size for _, size in self.jordan_blocks) == self.num_rows
            ), "Sum of Jordan block sizes must match matrix size."
            assert (
                self.eigenvalues is None
            ), "Cannot specify both Jordan blocks and eigenvalues."
            assert self.rank is None, "Cannot specify both Jordan blocks and rank."

    def build_sized(self, num_rows: int, num_cols: int | None = None) -> Matrix:
        self.num_rows = num_rows
        self.num_cols = num_cols if num_cols is not None else num_rows
        return self.build()

    def build(self) -> Matrix:
        self.assert_requirements()
        if self.jordan_blocks is not None:
            return self.build_jordanized()
        if self.eigenvalues is not None:
            return self.build_diagonalizable()
        if self.rank is not None:
            if (
                self.rank == min(self.num_rows, self.num_cols)
                and self.num_rows == self.num_cols
            ):
                return self.build_full_rank()
            else:
                return self.build_rank()
        return self.build_random()

    def build_random(self) -> Matrix:
        dist = self.dist or (lambda: random.randint(-5, 5))
        return Matrix(
            [[dist() for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        )

    def build_full_rank(self) -> Matrix:
        dist = self.dist or (lambda: random.randint(-5, 5))
        N = self.num_rows
        while True:
            val = Matrix([[dist() for _ in range(N)] for _ in range(N)])
            if val.rank() == N:
                return val

    def build_rank(self) -> Matrix:
        dist = self.dist or (lambda: random.randint(-5, 5))
        rows, cols, rank = self.num_rows, self.num_cols, self.rank
        # Generate A (rows x rank) and B (rank x cols), both full rank
        while True:
            A = Matrix([[dist() for _ in range(rank)] for _ in range(rows)])
            if A.rank() == rank:
                break
        while True:
            B = Matrix([[dist() for _ in range(cols)] for _ in range(rank)])
            if B.rank() == rank:
                break
        return A * B

    def build_diagonalizable(self) -> Matrix:
        # Diagonal matrix with specified eigenvalues, randomized by similarity if requested
        N = self.num_rows
        diag = []
        for eig, mult in self.eigenvalues:
            diag.extend([eig] * mult)
        D = Matrix.diagonal(diag)
        if not self.do_randomize_from_diagonal_form:
            return D
        P = RandomMatrixBuilder.new().with_size(N, N).build_full_rank()
        P_inv = P.inverse()
        return P_inv * D * P

    def build_jordan(self) -> Matrix:
        N = self.num_rows
        total_size = sum(size for _, size in self.jordan_blocks)
        if total_size != N:
            raise ValueError(
                f"Sum of Jordan block sizes ({total_size}) must equal matrix size ({N})"
            )
        jordan_form = [[0 for _ in range(N)] for _ in range(N)]
        current_index = 0
        for eigenvalue, size in self.jordan_blocks:
            for i in range(size):
                jordan_form[current_index + i][current_index + i] = eigenvalue
                if i < size - 1:
                    jordan_form[current_index + i][current_index + i + 1] = 1
            current_index += size
        return Matrix(jordan_form)

    def build_jordanized(self) -> Matrix:
        # Randomly similar to a given Jordan form
        J = self.build_jordan()
        N = self.num_rows
        P = RandomMatrixBuilder.new().with_size(N, N).build_full_rank()
        P_inv = P.inverse()
        return P_inv * J * P


def raw_gen_rand_matrix(
    rows: int, cols: int, dist: Callable[[], Any] | None = None
) -> Matrix:
    return (
        RandomMatrixBuilder.new().with_size(rows, cols).with_dist(dist).build_random()
    )


def gen_regular_matrix(N: int, dist: Callable[[], Any] | None = None) -> Matrix:
    return RandomMatrixBuilder.new().with_size(N, N).with_dist(dist).build_full_rank()


def gen_matrix_with_rank(
    rows: int, cols: int, rank: int | None = None, dist: Callable[[], Any] | None = None
) -> Matrix:
    return (
        RandomMatrixBuilder.new()
        .with_size(rows, cols)
        .with_rank(rank or min(rows, cols))
        .with_dist(dist)
        .build_rank()
    )


def gen_jordan_matrix(N: int, blocks: List[Tuple[Any, int]]) -> Matrix:
    return (
        RandomMatrixBuilder.new()
        .with_size(N, N)
        .with_jordan_blocks(blocks)
        .build_jordan()
    )


def gen_matrix_with_jordan_blocks(
    N: int, blocks: List[Tuple[Any, int]], dist: Callable[[], Any] | None = None
) -> Matrix:
    return (
        RandomMatrixBuilder.new()
        .with_size(N, N)
        .with_jordan_blocks(blocks)
        .with_dist(dist)
        .build_jordanized()
    )


def gen_diagonalizable_matrix(
    N: int,
    eigenvalues: List[Tuple[float, int]] | None = None,
    dist: Callable[[], Any] | None = None,
) -> Matrix:
    if eigenvalues is None:
        eigenvalues = [
            (dist() if dist is not None else random.randint(-5, 5), 1) for _ in range(N)
        ]
    return (
        RandomMatrixBuilder.new()
        .with_size(N, N)
        .with_eigenvalues(eigenvalues)
        .with_dist(dist)
        .build_diagonalizable()
    )
