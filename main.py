from linalg import Matrix
import log as logger_module


def main():
    with logger_module.nest_logger() as lg:
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        C = Matrix([[A, B], [B, A]])
        D = Matrix([[A, A], [B, B]])
        E = C + D
    logs = str(lg)
    print(r"$$ %s $$" % E.cformat())
    print(logs)


if __name__ == "__main__":
    main()
