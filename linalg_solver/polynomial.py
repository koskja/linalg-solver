import sympy
from .fmt import cformat
from typing import Dict, Tuple, List, Any


class Polynomial:
    powers: dict[int, any]
    var: str

    def __init__(self, powers: dict[int, any], var: str = "x"):
        self.powers = {k: v for k, v in powers.items() if v != 0}
        self.var = var

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (int, float)) and other == 0:
            return not self.powers
        if isinstance(other, Polynomial):
            return self.var == other.var and self.powers == other.powers
        return NotImplemented

    def __hash__(self) -> int:
        # Sort items to ensure deterministic hash
        items = tuple(sorted(self.powers.items()))
        return hash((self.var, items))

    def cformat(self, arg_of: str = None) -> str:
        res = ""
        for exp, coef in sorted(self.powers.items(), key=lambda x: -x[0]):
            if coef == 0:
                continue
            if cformat(coef).startswith("-"):
                res += "-"
                coef = -coef
            else:
                if res:
                    res += "+"
            coef_str = "" if coef == 1 and exp != 0 else cformat(coef)
            var_str = "" if exp == 0 else self.var
            pow_str = "" if exp <= 1 else r"^{%s}" % exp
            res += r"%s{%s}%s" % (coef_str, var_str, pow_str)
        if arg_of is None or arg_of == "+":
            return res
        if len(self.powers) <= 1 and not (res.startswith("-") and arg_of == "*"):
            return res
        return f"({res})"

    def multi_add(self, *others: "Polynomial") -> "Polynomial":
        res = self.powers.copy()
        for other in others:
            if not isinstance(other, Polynomial):
                other = Polynomial({0: other})
            elif other.var != self.var:
                raise TypeError(
                    f"Cannot add Polynomials with different variables: '{self.var}' and '{other.var}'"
                )
            for exp, coef in other.powers.items():
                res[exp] = res.get(exp, 0) + coef
        return Polynomial(res, self.var)

    def __radd__(self, other: "Polynomial") -> "Polynomial":
        return self.multi_add(other)

    def __add__(self, other: "Polynomial") -> "Polynomial":
        return self.multi_add(other)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        return self + (-other)

    def __neg__(self) -> "Polynomial":
        return Polynomial({exp: -coef for exp, coef in self.powers.items()}, self.var)

    def __mul__(self, other) -> "Polynomial":
        if not isinstance(other, Polynomial):
            return Polynomial(
                {exp: coef * other for exp, coef in self.powers.items()}, self.var
            )
        if other.var != self.var:
            raise TypeError(
                f"Cannot multiply Polynomials with different variables: '{self.var}' and '{other.var}'"
            )
        res = {}
        for exp1, coef1 in self.powers.items():
            for exp2, coef2 in other.powers.items():
                res[exp1 + exp2] = res.get(exp1 + exp2, 0) + coef1 * coef2
        return Polynomial(res, self.var)

    def remove_root(self, root: any) -> "Polynomial":
        """
        Removes a specified root from the polynomial by dividing by (x - root).

        Args:
            root: The root to remove.

        Returns:
            A new Polynomial representing the quotient.

        Raises:
            ValueError: If the provided value is not a root (division results in a non-zero remainder).
        """
        x = sympy.symbols(self.var)
        # Convert self to sympy.Poly
        coeffs = [
            self.powers.get(i, 0)
            for i in range(max(self.powers.keys(), default=-1) + 1)
        ][::-1]
        if not coeffs:  # Handle zero polynomial
            return Polynomial({}, self.var)
        p = sympy.Poly(coeffs, x)

        # Create the divisor polynomial (x - root)
        divisor = sympy.Poly(x - root, x)

        # Perform polynomial division
        quotient, remainder = sympy.div(p, divisor)

        # Check if the remainder is zero (or numerically close to zero)
        if not sympy.simplify(remainder).is_zero:
            raise ValueError(
                f"{root} is not a root of the polynomial, division resulted in remainder {remainder}"
            )

        # Convert quotient back to Polynomial object
        return Polynomial._sympy_poly_to_polynomial(quotient, self.var)

    def factor_roots(self, roots: List[Tuple[any, int]]) -> Dict["Polynomial", int]:
        res = self
        for root, mult in roots:
            for _ in range(mult):
                res = res.remove_root(root)
        reduced = {Polynomial({0: -root, 1: 1}, self.var): mult for root, mult in roots}
        if len(res.powers) == 1 and res.powers.get(0, 1) == 1:
            return reduced
        return {res: 1} | reduced

    def __rmul__(self, other: "Polynomial") -> "Polynomial":
        return self * other

    def radical_roots(self):
        x = sympy.symbols(self.var)
        coeffs = [
            self.powers.get(i, 0)
            for i in range(max(self.powers.keys(), default=-1) + 1)
        ][::-1]
        p = sympy.Poly(coeffs, x)
        return sympy.roots(p, multiple=False)

    @staticmethod
    def _sympy_poly_to_polynomial(sympy_p: sympy.Poly, var_name: str) -> "Polynomial":
        """Converts a sympy.Poly object to a Polynomial instance."""
        # Extract coefficients and exponents from the sympy polynomial
        sympy_dict = sympy_p.as_dict()
        if not sympy_dict:  # Handle zero polynomial
            return Polynomial({}, var_name)
        # sympy.Poly.as_dict() returns {(exp,): coeff} for univariate polynomials
        powers = {exp[0]: coeff for exp, coeff in sympy_dict.items() if coeff != 0}
        return Polynomial(powers, var_name)
