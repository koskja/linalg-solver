import random
import sympy

from linalg_solver.log import log, global_logger
from linalg_solver.linalg import Matrix
from linalg_solver.random_matrix import (
    gen_regular_matrix,
    gen_diagonalizable_matrix,
    gen_matrix_with_rank,
)
from linalg_solver.fmt import cformat


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _rationalize_matrix(mat: Matrix) -> Matrix:
    """Return a copy of *mat* with all numeric entries converted to SymPy Rationals."""

    return mat.self_map(
        lambda x: sympy.Rational(x) if isinstance(x, (int, float)) else x
    )


def _rationalize_vector(vec):
    """Convert all numeric items in *vec* to SymPy Rationals."""

    return [sympy.Rational(x) if isinstance(x, (int, float)) else x for x in vec]


# -----------------------------------------------------------------------------
# Example generation
# -----------------------------------------------------------------------------


def determinant_example():
    log(r"\section{Determinant}")
    A = _rationalize_matrix(gen_regular_matrix(3))
    log(r"Vstupní matice $A$: $%s$", A)
    det_val = A.determinant(log_permutation_details=True)
    log(r"\textbf{Determinant:} $%s$", det_val)


def inverse_example():
    log(r"\section{Inverze}")
    A = _rationalize_matrix(gen_regular_matrix(3))
    log(r"Vstupní matice $A$: $%s$", A)
    inv = A.inverse(log_matrices=True, log_steps=True, log_result=True)
    log(r"\textbf{Inverzní matice:} $%s$", inv)


def linear_system_example():
    log(r"\section{Lineární soustava}")
    A = _rationalize_matrix(gen_regular_matrix(3))
    b = [random.randint(-5, 5) for _ in range(3)]
    b = _rationalize_vector(b)
    log(r"Lineární soustava $A\,x=b$ s $A=%s$", A)
    sol = A.find_preimage_of(b, log_matrices=True, log_steps=True, log_result=True)
    if isinstance(sol, Matrix.NoSolution):
        log(r"\textbf{Množina řešení:} $%s$", sol)
    else:
        log(r"\textbf{Množina řešení:} $%s$", sol)


def eigenvalues_example():
    log(r"\section{Vlastní čísla}")
    eig_vals = [(-3, 1), (0, 1), (4, 1)]
    A = _rationalize_matrix(gen_diagonalizable_matrix(3, eigenvalues=eig_vals))
    log(r"Vstupní matice $A$: $%s$", A)
    eigs = A.eigenvalues()
    eig_summary = ", ".join(["%s^{%d}" % (cformat(e), m) for e, m in eigs.items()])
    log(r"\textbf{Vlastní čísla:} $%s$", eig_summary)


def diagonalization_example():
    log(r"\section{Diagonalizace}")
    eig_vals = [(5, 1), (2, 1), (-5, 1)]
    A = _rationalize_matrix(gen_diagonalizable_matrix(3, eigenvalues=eig_vals))
    log(r"Vstupní matice $A$: $%s$", A)
    diag_res = A.diagonalize()
    log(r"%s", diag_res)


def kernel_example():
    log(r"\section{Kernel}")
    A = _rationalize_matrix(gen_matrix_with_rank(3, 4, rank=2))
    log(r"Vstupní matice $A$: $%s$ \\", A)
    ker = A.find_preimage_of(
        [0] * 3, log_matrices=True, log_steps=True, log_result=True
    )
    log(r"\textbf{Báze jádra:}    $%s$", ker)


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def main():
    random.seed(2025)

    determinant_example()
    inverse_example()
    linear_system_example()
    eigenvalues_example()
    diagonalization_example()
    kernel_example()

    with open("output.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(global_logger.accum))


if __name__ == "__main__":
    main()
