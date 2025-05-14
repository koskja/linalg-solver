from fmt import cformat
from linalg import Matrix
from log import log, nest_logger
import sympy
import random
from random_matrix import (
    gen_regular_matrix,
    gen_matrix_with_rank,
    gen_diagonalizable_matrix,
    gen_jordan_matrix,
    gen_matrix_with_jordan_blocks,
    raw_gen_rand_matrix,
)


def random_slr():
    randbool = lambda: random.randint(0, 1) == 1
    upper_bound = random.randint(1, 4)
    do_negatives = randbool()
    lower_bound = -upper_bound if do_negatives else 0
    r = lambda: sympy.Rational(random.randint(lower_bound, upper_bound), 1)
    n = random.randint(2, 4)
    log_level = random.randint(0, 3)
    for i in range(1):
        R = Matrix([[r() for _ in range(n)] for _ in range(n)])
        inv = R.inverse(
            log_matrices=log_level >= 1,
            log_steps=log_level >= 2,
            log_result=log_level >= 3,
        )
        log(r"\[ %s \]", inv)


def random_jordan(diagonal_only: bool = False, true_random: bool = False):
    n = random.randint(2, 5)
    blocks = []
    remaining_size = n
    while remaining_size > 0:
        eigenvalue = random.randint(-5, 5)
        if remaining_size == 1:
            block_size = 1
        else:
            block_size = random.randint(1, min(3, remaining_size))
        if diagonal_only:
            block_size = 1
        blocks.append((eigenvalue, block_size))
        remaining_size -= block_size
    if not true_random:
        J = gen_jordan_matrix(n, blocks)
    else:
        J = gen_matrix_with_jordan_blocks(n, blocks, lambda: random.randint(-2, 2))
    log(r"Generated a random \[%s \times %s\] Jordan normal form matrix:", n, n)
    log(r"\[ J = %s \]", J)
    block_description = []
    for eigenvalue, size in blocks:
        block_description.append(f"J_{{{size}}}({eigenvalue})")
    if not true_random:
        log(
            r"Jordan block structure: \[ J = \operatorname{diag}(%s) \]",
            ", ".join(block_description),
        )
    return J


def analyze_eigenvalues(matrix: Matrix):
    egvals = matrix.eigenvalues()
    for eigenvalue, multiplicity in egvals.items():
        eigenspace = matrix.find_eigenspace(eigenvalue)
        log(
            r"Eigenvalue $\lambda = %s$ has algebraic multiplicity %s and geometric multiplicity %s.",
            eigenvalue,
            multiplicity,
            eigenspace.dim(),
        )
        log(r"Eigenspace: \[ %s \]", eigenspace)


def main():
    with nest_logger() as lg:
        # Example: Full-rank random matrix
        A = gen_regular_matrix(3, lambda: random.randint(-2, 2))
        log(r"Random full-rank matrix: \[ A = %s \]", A)
        # Example: Random matrix with specified rank
        B = gen_matrix_with_rank(4, 5, 2, lambda: random.randint(-2, 2))
        log(r"Random 4x5 matrix of rank 2: \[ B = %s \]", B)
        # Example: Diagonalizable matrix with given eigenvalues
        eigs = [(2, 2), (3, 1)]
        C = gen_diagonalizable_matrix(3, eigs, lambda: random.randint(-2, 2))
        log(
            r"Random diagonalizable matrix with eigenvalues 2 (mult 2), 3 (mult 1): \[ C = %s \]",
            C,
        )
        # Example: Jordan normal form matrix
        D = gen_jordan_matrix(4, [(1, 2), (2, 2)])
        log(r"Jordan normal form matrix: \[ D = %s \]", D)
        # Example: Random matrix similar to a Jordan form
        E = gen_matrix_with_jordan_blocks(
            4, [(0, 2), (1, 2)], lambda: random.randint(-2, 2)
        )
        log(r"Random matrix similar to a Jordan form: \[ E = %s \]", E)
        # Example: Arbitrary random matrix
        F = raw_gen_rand_matrix(3, 4, lambda: random.randint(-2, 2))
        log(r"Arbitrary random 3x4 matrix: \[ F = %s \]", F)
        # Diagonalization example
        d = C.diagonalize()
        log(r"Diagonalization of C: %s", d)
    logs = str(lg)
    print(logs)


if __name__ == "__main__":
    main()
