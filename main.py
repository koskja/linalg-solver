from fmt import cformat
from linalg import (
    Matrix,
    gen_matrix_with_jordan_blocks,
    gen_jordan_matrix,
    gen_regular_matrix,
)
from log import log, nest_logger
import sympy
import random


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
    # Randomly determine the size of the matrix
    n = random.randint(2, 5)
    # Generate random Jordan blocks
    blocks = []
    remaining_size = n

    while remaining_size > 0:
        # Random eigenvalue between -5 and 5
        eigenvalue = random.randint(-5, 5)

        # Random block size (at least 1, at most what's remaining)
        if remaining_size == 1:
            block_size = 1
        else:
            block_size = random.randint(1, min(3, remaining_size))
        if diagonal_only:
            block_size = 1

        blocks.append((eigenvalue, block_size))
        remaining_size -= block_size

    # Generate the Jordan normal form matrix
    if not true_random:
        J = gen_jordan_matrix(n, blocks)
    else:
        J = gen_matrix_with_jordan_blocks(n, blocks, lambda: random.randint(-2, 2))
    # Display the matrix and its Jordan structure
    log(r"Generated a random \[%s \times %s\] Jordan normal form matrix:", n, n)
    log(r"\[ J = %s \]", J)

    # Display the Jordan block structure
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
        J = random_jordan(True, True)
        # J = gen_regular_matrix(3, lambda: random.randint(-2, 2))
        d = J.diagonalize()
        log(r"Diagonalization: %s", d)
    logs = str(lg)
    print(logs)


if __name__ == "__main__":
    main()
