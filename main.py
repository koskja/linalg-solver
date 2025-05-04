from linalg import Matrix
import log as logger_module


def main():
    with logger_module.nest_logger() as lg:
        A = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -2], [0, 0, -3, 0]]
        A = Matrix(A)
        B = Matrix([[5, 6], [7, 8]])
        C = Matrix([[2, 3], [8, 5]])
        D = Matrix([[A, B, C], [C, C, B], [A, B, A]])
        E = Matrix([[C, C, B], [B, A, A], [C, B, A]])
        # d = Matrix.from_block_matrix([[A, B], [C, A]]).determinant()
        e = A.eigenvalues(real_only=True)
    logs = str(lg)
    print(logs)


if __name__ == "__main__":
    main()
