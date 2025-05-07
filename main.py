from fmt import cformat
from linalg import Matrix
from log import log, nest_logger
import sympy
import random


def main():
    randbool = lambda: random.randint(0, 1) == 1
    upper_bound = random.randint(1, 4)
    do_negatives = randbool()
    lower_bound = -upper_bound if do_negatives else 0
    r = lambda: sympy.Rational(random.randint(lower_bound, upper_bound), 1)
    n = random.randint(2, 4)
    log_level = random.randint(0, 3)
    with nest_logger() as lg:
        for i in range(1):
            R = Matrix([[r() for _ in range(n)] for _ in range(n)])
            inv = R.inverse(
                log_matrices=log_level >= 1,
                log_steps=log_level >= 2,
                log_result=log_level >= 3,
            )
            log(r"\[ %s \]", inv)
    logs = str(lg)
    print(logs)


if __name__ == "__main__":
    main()
