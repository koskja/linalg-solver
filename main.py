from linalg import Matrix
import log as logger_module


def main():
    with logger_module.nest_logger() as lg:
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        C = Matrix([[2, 3], [8, 5]])
        D = Matrix([[A, B, C], [C, C, B], [A, B, A]])
        E = Matrix([[C, C, B], [B, A, A], [C, B, A]])
        F = D * E
    logs = str(lg)
    print(logs)


if __name__ == "__main__":
    main()
