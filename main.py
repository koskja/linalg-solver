from linalg_solver import *
import sympy
import random


def random_slr():
    randbool = lambda: random.randint(0, 1) == 1
    upper_bound = random.randint(1, 4)
    do_negatives = randbool()
    lower_bound = -upper_bound if do_negatives else 0
    r = lambda: sympy.Rational(random.randint(lower_bound, upper_bound), 1)
    n = random.randint(2, 5)
    log_level = random.randint(0, 4)
    for i in range(1):
        R = Matrix([[r() for _ in range(n)] for _ in range(n)])
        inv = R.inverse(
            log_matrices=log_level >= 1,
            log_steps=log_level >= 2,
            log_result=log_level >= 3,
        )
        log(r"\[ %s \]", inv)
        log(r"\newpage")
        # Generate a random vector and find its preimage
        m = random.randint(2, n)  # Choose a number of rows between 2 and n
        A = Matrix([[r() for _ in range(n)] for _ in range(m)])
        b = [r() for _ in range(m)]

        log(r"\textbf{Random matrix $A$:} \[ %s \]", A)
        log(r"\textbf{Random vector $b$:} \[ %s \]", Matrix.new_vector(b).cformat())
        log(r"\textbf{Finding preimage of $b$ under $A$:}")

        preimage = A.find_preimage_of(
            b,
            log_matrices=log_level >= 1,
            log_steps=log_level >= 2,
            log_result=log_level >= 3,
        )

        log(r"\textbf{Preimage:} \[ %s \]", preimage)
        log(r"\newpage")


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
        random_slr()
    logs = str(lg)
    print(logs)


if __name__ == "__main__":
    main()
