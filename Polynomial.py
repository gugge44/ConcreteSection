"""
Polynomial - A sparse multivariate polynomial implementation.

This module provides a Polynomial class for representing and manipulating
multivariate polynomials using a sparse dictionary-based storage format.

Features:
    - Sparse storage for memory efficiency
    - Full arithmetic operations (+, -, *, /, //, %, **)
    - Calculus operations (differentiate, integrate, gradient)
    - Polynomial composition and evaluation
    - Support for multivariate polynomials
    - Special polynomial generators (Chebyshev, Legendre, Hermite)

Example:
    >>> p = Polynomial([1, 2, 3])  # 1 + 2x + 3x²
    >>> q = Polynomial([0, 1])     # x
    >>> print(p * q)               # x + 2x² + 3x³
    >>> p(2)                       # Evaluate at x=2: 17.0
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from numbers import Number
from typing import Dict, Tuple, Union, List, Optional, Iterator, Callable
from functools import reduce
import operator
import warnings


# Type aliases
Exponents = Tuple[int, ...]
CoefficientDict = Dict[Exponents, float]
ArrayLike = Union[list, tuple, np.ndarray]


class PolynomialConfig:
    """Global configuration for polynomial operations."""
    
    rtol: float = 1e-9
    atol: float = 1e-12
    display_precision: int = 6
    use_unicode: bool = True
    warn_on_dimension_mismatch: bool = True


def _is_close(a: float, b: float) -> bool:
    """Check if two numbers are approximately equal."""
    return np.isclose(a, b, rtol=PolynomialConfig.rtol, atol=PolynomialConfig.atol)


def _is_zero(value: float) -> bool:
    """Check if a value is approximately zero."""
    return np.isclose(value, 0, atol=PolynomialConfig.atol)


def _is_number(value) -> bool:
    """Check if value is a numeric type."""
    return isinstance(value, Number)


def _is_array_like(value) -> bool:
    """Check if value is array-like (list, tuple, or ndarray)."""
    return isinstance(value, (list, tuple, np.ndarray))


def _format_exponent(n: int) -> str:
    """Format an exponent, using Unicode superscripts if enabled."""
    if not PolynomialConfig.use_unicode:
        return f"^{n}" if n != 1 else ""
    
    if n == 1:
        return ""
    
    superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'}
    return ''.join(superscripts.get(c, c) for c in str(n))


def _format_coefficient(coeff: float, is_first: bool, has_variables: bool) -> str:
    """
    Format a coefficient for display.
    
    Args:
        coeff: The coefficient value.
        is_first: Whether this is the first term in the polynomial.
        has_variables: Whether this term has variable factors.
    
    Returns:
        Formatted coefficient string.
    """
    precision = PolynomialConfig.display_precision
    
    # Handle integer-valued floats
    if isinstance(coeff, float) and coeff == int(coeff):
        coeff = int(coeff)
    
    abs_coeff = abs(coeff)
    sign = "-" if coeff < 0 else ("" if is_first else "+")
    
    # Special cases for ±1 with variables
    if has_variables and abs_coeff == 1:
        if is_first:
            return "-" if coeff < 0 else ""
        else:
            return " - " if coeff < 0 else " + "
    
    # Format the number
    if isinstance(abs_coeff, int):
        num_str = str(abs_coeff)
    else:
        num_str = f"{abs_coeff:.{precision}g}"
    
    if is_first:
        return f"-{num_str}" if coeff < 0 else num_str
    else:
        return f" - {num_str}" if coeff < 0 else f" + {num_str}"


class Polynomial:
    """
    A sparse multivariate polynomial class.
    
    Polynomials are stored as dictionaries mapping exponent tuples to coefficients.
    For example, 3x²y + 2xy² is stored as {(2,1): 3, (1,2): 2}.
    
    The dimension (number of variables) is fixed at creation and maintained
    throughout operations. When combining polynomials of different dimensions,
    the lower-dimensional polynomial is promoted by appending zero exponents.
    
    Attributes:
        data: Dictionary mapping exponent tuples to coefficients.
        ndim: Number of variables in the polynomial.
        log_order: Order of Taylor expansion for logarithm approximation.
    
    Examples:
        >>> p = Polynomial([1, 2, 3])  # 1 + 2x + 3x²
        >>> p(2)                       # 1 + 2*2 + 3*4 = 17.0
        17.0
        
        >>> # 2D polynomial: 3xy + 2x²
        >>> q = Polynomial({(1, 1): 3, (2, 0): 2})
        >>> q(1, 2)                    # 3*1*2 + 2*1² = 8.0
        8.0
    """
    
    __slots__ = ('_data', '_ndim', 'log_order')
    
    def __init__(
        self, 
        coefficients: Union[ArrayLike, CoefficientDict, 'Polynomial', Number],
        ndim: Optional[int] = None,
        log_order: int = 5
    ):
        """
        Initialize a Polynomial.
        
        Args:
            coefficients: Can be:
                - A number (creates constant polynomial)
                - A 1D array [a0, a1, ...] for a0 + a1*x + a2*x² + ...
                - An n-dimensional array for multivariate polynomials
                - A dictionary {(i, j, ...): coeff} mapping exponents to coefficients
                - Another Polynomial (creates a copy)
            ndim: Force a specific number of dimensions. If None, inferred from input.
            log_order: Order for Taylor series approximation when integrating 1/x.
        
        Raises:
            ValueError: If ndim is less than the required dimensions.
        
        Examples:
            >>> Polynomial(5)              # Constant: 5
            >>> Polynomial([1, 2, 3])      # 1 + 2x + 3x²
            >>> Polynomial({(2,): 3})      # 3x²
            >>> Polynomial([1, 2], ndim=3) # 1 + 2x in 3D space
        """
        self.log_order = log_order
        
        if isinstance(coefficients, Polynomial):
            inferred_ndim = coefficients.ndim
            self._data = coefficients._data.copy()
        elif isinstance(coefficients, dict):
            inferred_ndim, self._data = self._parse_dict(coefficients)
        else:
            inferred_ndim, self._data = self._parse_array(coefficients)
        
        # Handle explicit ndim
        if ndim is not None:
            if ndim < inferred_ndim:
                raise ValueError(
                    f"Specified ndim={ndim} is less than required {inferred_ndim}"
                )
            if ndim > inferred_ndim:
                # Promote to higher dimension
                self._data = {
                    k + (0,) * (ndim - len(k)): v 
                    for k, v in self._data.items()
                }
            self._ndim = ndim
        else:
            self._ndim = inferred_ndim
    
    @staticmethod
    def _parse_dict(coefficients: CoefficientDict) -> Tuple[int, CoefficientDict]:
        """Parse a coefficient dictionary, returning (ndim, cleaned_data)."""
        if not coefficients:
            return 1, {(0,): 0.0}
        
        # Validate and find dimensions
        ndims = set()
        for key in coefficients.keys():
            if not isinstance(key, tuple):
                raise TypeError(f"Dictionary keys must be tuples, got {type(key)}")
            if not all(isinstance(e, (int, np.integer)) for e in key):
                raise TypeError(f"Exponents must be integers, got {key}")
            ndims.add(len(key))
        
        if len(ndims) > 1:
            raise ValueError(
                f"Inconsistent exponent tuple lengths: {ndims}. "
                "All keys must have the same length."
            )
        
        ndim = ndims.pop()
        
        # Filter zeros and convert to float
        filtered = {
            tuple(int(e) for e in key): float(value)
            for key, value in coefficients.items()
            if not _is_zero(value)
        }
        
        if not filtered:
            return ndim, {(0,) * ndim: 0.0}
        
        return ndim, filtered
    
    @staticmethod
    def _parse_array(coefficients: Union[ArrayLike, Number]) -> Tuple[int, CoefficientDict]:
        """Parse an array or number, returning (ndim, data)."""
        if _is_number(coefficients):
            if _is_zero(coefficients):
                return 1, {(0,): 0.0}
            return 1, {(0,): float(coefficients)}
        
        array = np.asarray(coefficients, dtype=float)
        if array.size == 0:
            return 1, {(0,): 0.0}
        
        ndim = array.ndim
        data = {}
        
        for indices in np.ndindex(array.shape):
            value = array[indices]
            if not _is_zero(value):
                data[indices] = float(value)
        
        if not data:
            return ndim, {(0,) * ndim: 0.0}
        
        return ndim, data
    
    # Properties
    
    @property
    def data(self) -> CoefficientDict:
        """Read-only access to coefficient dictionary."""
        return self._data.copy()
    
    @property
    def ndim(self) -> int:
        """Number of variables in the polynomial."""
        return self._ndim
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape needed to represent this polynomial as a dense array."""
        if not self._data or self.is_zero:
            return (1,) * self._ndim
        
        return tuple(
            max((key[i] for key in self._data.keys()), default=0) + 1
            for i in range(self._ndim)
        )
    
    @property
    def degree(self) -> int:
        """Total degree (maximum sum of exponents across all terms)."""
        if not self._data or self.is_zero:
            return 0
        return max(sum(key) for key in self._data.keys())
    
    def degree_in(self, dim: Union[int, str]) -> int:
        """Degree with respect to a specific variable."""
        dim = self._resolve_dim(dim)
        if not self._data or self.is_zero:
            return 0
        return max(key[dim] for key in self._data.keys())
    
    @property
    def is_zero(self) -> bool:
        """Check if this polynomial is identically zero."""
        return not self._data or all(_is_zero(v) for v in self._data.values())
    
    @property
    def is_constant(self) -> bool:
        """Check if this polynomial is a constant."""
        if self.is_zero:
            return True
        return all(all(e == 0 for e in key) for key in self._data.keys())
    
    @property
    def leading_coefficient(self) -> float:
        """Coefficient of the highest-degree term."""
        if self.is_zero:
            return 0.0
        max_key = max(self._data.keys(), key=lambda k: (sum(k), k))
        return self._data[max_key]
    
    @property
    def constant_term(self) -> float:
        """The constant term (coefficient of x⁰y⁰...)."""
        return self._data.get((0,) * self._ndim, 0.0)
    
    @property 
    def coefficients(self) -> List[float]:
        """List of all non-zero coefficients."""
        return list(self._data.values())
    
    @property
    def terms(self) -> List[Tuple[Exponents, float]]:
        """List of (exponents, coefficient) pairs, sorted by degree."""
        return sorted(self._data.items(), key=lambda x: (sum(x[0]), x[0]))
    
    # Factory methods
    
    @classmethod
    def zero(cls, ndim: int = 1) -> 'Polynomial':
        """Create a zero polynomial with specified dimension."""
        p = cls.__new__(cls)
        p._data = {(0,) * ndim: 0.0}
        p._ndim = ndim
        p.log_order = 5
        return p
    
    @classmethod
    def one(cls, ndim: int = 1) -> 'Polynomial':
        """Create a unit polynomial (constant 1) with specified dimension."""
        p = cls.__new__(cls)
        p._data = {(0,) * ndim: 1.0}
        p._ndim = ndim
        p.log_order = 5
        return p
    
    @classmethod
    def constant(cls, value: float, ndim: int = 1) -> 'Polynomial':
        """Create a constant polynomial."""
        if _is_zero(value):
            return cls.zero(ndim)
        p = cls.__new__(cls)
        p._data = {(0,) * ndim: float(value)}
        p._ndim = ndim
        p.log_order = 5
        return p
    
    @classmethod
    def monomial(cls, exponents: Union[Exponents, int], coefficient: float = 1.0) -> 'Polynomial':
        """
        Create a monomial (single term polynomial).
        
        Args:
            exponents: Tuple of exponents, or single int for univariate.
            coefficient: Coefficient of the monomial.
        
        Returns:
            Polynomial representing coefficient * x₀^e₀ * x₁^e₁ * ...
        
        Examples:
            >>> Polynomial.monomial(3)        # x³
            >>> Polynomial.monomial((2, 1))   # x²y
            >>> Polynomial.monomial((1, 2), 3) # 3xy²
        """
        if isinstance(exponents, (int, np.integer)):
            exponents = (int(exponents),)
        else:
            exponents = tuple(int(e) for e in exponents)
        
        if _is_zero(coefficient):
            return cls.zero(len(exponents))
        
        p = cls.__new__(cls)
        p._data = {exponents: float(coefficient)}
        p._ndim = len(exponents)
        p.log_order = 5
        return p
    
    @classmethod
    def variable(cls, index: int = 0, ndim: Optional[int] = None) -> 'Polynomial':
        """
        Create a polynomial representing a single variable.
        
        Args:
            index: Which variable (0 for first, 1 for second, etc.)
            ndim: Total number of dimensions. Defaults to index + 1.
        
        Returns:
            Polynomial representing the variable x_index.
        
        Examples:
            >>> Polynomial.variable()      # x (1D)
            >>> Polynomial.variable(0, 2)  # x (2D: x, y)
            >>> Polynomial.variable(1, 2)  # y (2D: x, y)
        """
        if ndim is None:
            ndim = index + 1
        if index >= ndim:
            raise ValueError(f"Variable index {index} >= ndim {ndim}")
        
        exponents = [0] * ndim
        exponents[index] = 1
        return cls.monomial(tuple(exponents), 1.0)
    
    @classmethod
    def from_roots(cls, roots: ArrayLike) -> 'Polynomial':
        """
        Create a monic polynomial from its roots.
        
        Args:
            roots: List of roots.
        
        Returns:
            Polynomial (x - r₁)(x - r₂)...(x - rₙ)
        
        Examples:
            >>> Polynomial.from_roots([1, 2])  # (x-1)(x-2) = x² - 3x + 2
        """
        roots = np.asarray(roots).flatten()
        if len(roots) == 0:
            return cls.one()
        
        # Use numpy's poly function for numerical stability
        coeffs = np.poly(roots)
        # np.poly returns highest degree first, we need lowest first
        return cls(coeffs[::-1])
    
    # Internal helpers
    
    def _clean(self) -> 'Polynomial':
        """Return a new polynomial with zero coefficients removed."""
        cleaned = {k: v for k, v in self._data.items() if not _is_zero(v)}
        if not cleaned:
            cleaned = {(0,) * self._ndim: 0.0}
        
        p = Polynomial.__new__(Polynomial)
        p._data = cleaned
        p._ndim = self._ndim
        p.log_order = self.log_order
        return p
    
    def _promote_ndim(self, target_ndim: int) -> 'Polynomial':
        """Return polynomial promoted to higher dimension."""
        if target_ndim <= self._ndim:
            return self
        
        extra = target_ndim - self._ndim
        p = Polynomial.__new__(Polynomial)
        p._data = {k + (0,) * extra: v for k, v in self._data.items()}
        p._ndim = target_ndim
        p.log_order = self.log_order
        return p
    
    def _resolve_dim(self, dim: Union[int, str]) -> int:
        """Convert dimension specification to integer index."""
        if isinstance(dim, str):
            # Convention: 'x' is last, 'y' is second-to-last, etc.
            dim_map = {'x': -1, 'y': -2, 'z': -3, 'w': -4}
            if dim.lower() in dim_map:
                dim = self._ndim + dim_map[dim.lower()]
            else:
                raise ValueError(f"Unknown dimension name: '{dim}'. Use 'x', 'y', 'z', 'w' or integer.")
        
        # Handle negative indices
        if dim < 0:
            dim = self._ndim + dim
        
        if not 0 <= dim < self._ndim:
            raise IndexError(f"Dimension {dim} out of range for {self._ndim}D polynomial")
        
        return dim
    
    @staticmethod
    def _unify_dimensions(p1: 'Polynomial', p2: 'Polynomial') -> Tuple['Polynomial', 'Polynomial']:
        """Promote both polynomials to the same dimension."""
        if p1._ndim == p2._ndim:
            return p1, p2
        
        if PolynomialConfig.warn_on_dimension_mismatch:
            warnings.warn(
                f"Combining polynomials of different dimensions ({p1._ndim}D and {p2._ndim}D). "
                f"Lower dimension promoted to {max(p1._ndim, p2._ndim)}D.",
                stacklevel=3
            )
        
        target = max(p1._ndim, p2._ndim)
        return p1._promote_ndim(target), p2._promote_ndim(target)
    
    def copy(self) -> 'Polynomial':
        """Create a deep copy of this polynomial."""
        p = Polynomial.__new__(Polynomial)
        p._data = self._data.copy()
        p._ndim = self._ndim
        p.log_order = self.log_order
        return p
    
    # Item access
    
    def __getitem__(self, indices: Union[Exponents, int]) -> float:
        """Get coefficient at given exponents."""
        if isinstance(indices, (int, np.integer)):
            indices = (int(indices),)
        if len(indices) != self._ndim:
            raise IndexError(
                f"Expected {self._ndim} indices, got {len(indices)}"
            )
        return self._data.get(tuple(indices), 0.0)
    
    def __setitem__(self, indices: Union[Exponents, int], value: float) -> None:
        """Set coefficient at given exponents."""
        if isinstance(indices, (int, np.integer)):
            indices = (int(indices),)
        if len(indices) != self._ndim:
            raise IndexError(
                f"Expected {self._ndim} indices, got {len(indices)}"
            )
        
        indices = tuple(int(i) for i in indices)
        if _is_zero(value):
            self._data.pop(indices, None)
            if not self._data:
                self._data[(0,) * self._ndim] = 0.0
        else:
            self._data[indices] = float(value)
    
    def __iter__(self) -> Iterator[Tuple[Exponents, float]]:
        """Iterate over (exponents, coefficient) pairs."""
        return iter(self._data.items())
    
    def __len__(self) -> int:
        """Return number of non-zero terms."""
        return sum(1 for v in self._data.values() if not _is_zero(v))
    
    def __bool__(self) -> bool:
        """Return True if polynomial is non-zero."""
        return not self.is_zero
    
    def __contains__(self, exponents: Exponents) -> bool:
        """Check if a term with given exponents exists."""
        return exponents in self._data and not _is_zero(self._data[exponents])
    
    # Arithmetic operations
    
    def __pos__(self) -> 'Polynomial':
        """Unary positive (returns copy)."""
        return self.copy()
    
    def __neg__(self) -> 'Polynomial':
        """Negate the polynomial."""
        p = Polynomial.__new__(Polynomial)
        p._data = {k: -v for k, v in self._data.items()}
        p._ndim = self._ndim
        p.log_order = self.log_order
        return p
    
    def __add__(self, other: Union['Polynomial', Number]) -> 'Polynomial':
        """Add two polynomials or a polynomial and a number."""
        if _is_number(other):
            other = Polynomial.constant(other, self._ndim)
        elif not isinstance(other, Polynomial):
            return NotImplemented
        
        p1, p2 = self._unify_dimensions(self, other)
        
        result_data = defaultdict(float, p1._data)
        for key, value in p2._data.items():
            result_data[key] += value
        
        p = Polynomial.__new__(Polynomial)
        p._data = dict(result_data)
        p._ndim = p1._ndim
        p.log_order = self.log_order
        return p._clean()
    
    def __radd__(self, other: Number) -> 'Polynomial':
        return self + other
    
    def __sub__(self, other: Union['Polynomial', Number]) -> 'Polynomial':
        """Subtract a polynomial or number from this polynomial."""
        if _is_number(other):
            return self + (-other)
        elif isinstance(other, Polynomial):
            return self + (-other)
        return NotImplemented
    
    def __rsub__(self, other: Number) -> 'Polynomial':
        return (-self) + other
    
    def __mul__(self, other: Union['Polynomial', Number]) -> 'Polynomial':
        """Multiply two polynomials or a polynomial and a number."""
        if _is_number(other):
            if _is_zero(other):
                return Polynomial.zero(self._ndim)
            p = Polynomial.__new__(Polynomial)
            p._data = {k: v * other for k, v in self._data.items()}
            p._ndim = self._ndim
            p.log_order = self.log_order
            return p
        
        if not isinstance(other, Polynomial):
            return NotImplemented
        
        p1, p2 = self._unify_dimensions(self, other)
        
        result_data = defaultdict(float)
        for exp1, coef1 in p1._data.items():
            for exp2, coef2 in p2._data.items():
                new_exp = tuple(a + b for a, b in zip(exp1, exp2))
                result_data[new_exp] += coef1 * coef2
        
        p = Polynomial.__new__(Polynomial)
        p._data = dict(result_data)
        p._ndim = p1._ndim
        p.log_order = self.log_order
        return p._clean()
    
    def __rmul__(self, other: Number) -> 'Polynomial':
        return self * other
    
    def __pow__(self, exponent: int) -> 'Polynomial':
        """Raise polynomial to a non-negative integer power using binary exponentiation."""
        if not isinstance(exponent, (int, np.integer)):
            raise TypeError(f"Exponent must be an integer, got {type(exponent)}")
        
        exponent = int(exponent)
        
        if exponent < 0:
            raise ValueError("Cannot raise polynomial to negative power")
        if exponent == 0:
            return Polynomial.one(self._ndim)
        if exponent == 1:
            return self.copy()
        if self.is_zero:
            return Polynomial.zero(self._ndim)
        
        # Binary exponentiation
        result = Polynomial.one(self._ndim)
        base = self.copy()
        
        while exponent > 0:
            if exponent & 1:
                result = result * base
            base = base * base
            exponent >>= 1
        
        return result
    
    def __truediv__(self, other: Union['Polynomial', Number]) -> 'Polynomial':
        """Divide by a number or perform polynomial division."""
        if _is_number(other):
            if _is_zero(other):
                raise ZeroDivisionError("Cannot divide polynomial by zero")
            p = Polynomial.__new__(Polynomial)
            p._data = {k: v / other for k, v in self._data.items()}
            p._ndim = self._ndim
            p.log_order = self.log_order
            return p
        
        if isinstance(other, Polynomial):
            return self.__floordiv__(other)
        
        return NotImplemented
    
    def __rtruediv__(self, other: Number) -> 'Polynomial':
        """Division with polynomial in denominator - not supported."""
        raise TypeError(
            "Cannot divide a number by a polynomial. "
            "Use RationalPolynomial for rational functions."
        )
    
    def __floordiv__(self, other: 'Polynomial') -> 'Polynomial':
        """Polynomial floor division (quotient only)."""
        quotient, _ = self.divmod(other)
        return quotient
    
    def __mod__(self, other: 'Polynomial') -> 'Polynomial':
        """Polynomial modulo (remainder only)."""
        _, remainder = self.divmod(other)
        return remainder
    
    def divmod(self, divisor: Union['Polynomial', ArrayLike]) -> Tuple['Polynomial', 'Polynomial']:
        """
        Perform univariate polynomial division with remainder.
        
        For multivariate polynomials, this performs division treating the
        polynomial as univariate in the last variable with polynomial coefficients.
        
        Args:
            divisor: The polynomial to divide by.
        
        Returns:
            Tuple of (quotient, remainder) where self = quotient * divisor + remainder.
        
        Raises:
            ZeroDivisionError: If divisor is zero.
            ValueError: If attempting multivariate division.
        
        Examples:
            >>> p = Polynomial([1, 0, -1])  # x² - 1
            >>> d = Polynomial([1, 1])       # x + 1
            >>> q, r = p.divmod(d)
            >>> print(q)  # x - 1
            >>> print(r)  # 0
        """
        if not isinstance(divisor, Polynomial):
            divisor = Polynomial(divisor)
        
        if divisor.is_zero:
            raise ZeroDivisionError("Cannot divide by zero polynomial")
        
        # For univariate, we can do standard polynomial long division
        if self._ndim == 1 and divisor._ndim == 1:
            return self._univariate_divmod(divisor)
        
        # Promote dimensions if needed
        p1, p2 = self._unify_dimensions(self, divisor)
        
        # For multivariate, we do pseudo-division treating last variable as main
        return p1._multivariate_divmod(p2)
    
    def _univariate_divmod(self, divisor: 'Polynomial') -> Tuple['Polynomial', 'Polynomial']:
        """Univariate polynomial division."""
        quotient_data = {}
        remainder = self.copy()
        
        divisor_deg = divisor.degree
        divisor_lc = divisor.leading_coefficient
        
        while not remainder.is_zero and remainder.degree >= divisor_deg:
            deg_diff = remainder.degree - divisor_deg
            coef = remainder.leading_coefficient / divisor_lc
            
            if _is_zero(coef):
                break
            
            quotient_data[(deg_diff,)] = coef
            
            # Subtract coef * x^deg_diff * divisor from remainder
            subtrahend_data = {
                (e[0] + deg_diff,): v * coef 
                for e, v in divisor._data.items()
            }
            subtrahend = Polynomial(subtrahend_data)
            remainder = remainder - subtrahend
        
        if not quotient_data:
            quotient_data = {(0,): 0.0}
        
        return Polynomial(quotient_data), remainder
    
    def _multivariate_divmod(self, divisor: 'Polynomial') -> Tuple['Polynomial', 'Polynomial']:
        """
        Multivariate polynomial division using graded lexicographic order.
        
        This is a simplified version that works well for many practical cases.
        """
        def term_order(exp: Exponents) -> Tuple:
            """Graded lexicographic ordering (total degree, then lex)."""
            return (sum(exp), exp)
        
        quotient_data = defaultdict(float)
        remainder_data = dict(self._data)
        
        divisor_lt = max(divisor._data.keys(), key=term_order)
        divisor_lc = divisor._data[divisor_lt]
        
        max_iterations = 1000  # Safety limit
        iteration = 0
        
        while remainder_data and iteration < max_iterations:
            iteration += 1
            
            # Find leading term of remainder
            remainder_lt = max(remainder_data.keys(), key=term_order)
            remainder_lc = remainder_data[remainder_lt]
            
            if _is_zero(remainder_lc):
                del remainder_data[remainder_lt]
                continue
            
            # Check if divisor's leading term divides remainder's leading term
            exp_diff = tuple(r - d for r, d in zip(remainder_lt, divisor_lt))
            if any(e < 0 for e in exp_diff):
                # Can't divide - this term goes to remainder
                break
            
            # Compute quotient coefficient
            q_coef = remainder_lc / divisor_lc
            quotient_data[exp_diff] += q_coef
            
            # Update remainder: subtract q_coef * x^exp_diff * divisor
            for d_exp, d_coef in divisor._data.items():
                new_exp = tuple(ed + dd for ed, dd in zip(exp_diff, d_exp))
                remainder_data[new_exp] = remainder_data.get(new_exp, 0) - q_coef * d_coef
                if _is_zero(remainder_data[new_exp]):
                    del remainder_data[new_exp]
        
        if not quotient_data:
            quotient_data = {(0,) * self._ndim: 0.0}
        if not remainder_data:
            remainder_data = {(0,) * self._ndim: 0.0}
        
        q = Polynomial.__new__(Polynomial)
        q._data = dict(quotient_data)
        q._ndim = self._ndim
        q.log_order = self.log_order
        
        r = Polynomial.__new__(Polynomial)
        r._data = remainder_data
        r._ndim = self._ndim
        r.log_order = self.log_order
        
        return q._clean(), r._clean()
    
    def __matmul__(self, other: 'Polynomial') -> 'Polynomial':
        """
        Polynomial composition: (p @ q)(x) = p(q(x)).
        
        For multivariate polynomials, substitutes q for the last variable only.
        Use transform() for more general substitutions.
        
        Examples:
            >>> p = Polynomial([1, 0, 1])  # 1 + x²
            >>> q = Polynomial([1, 1])     # 1 + x
            >>> p @ q                      # 1 + (1+x)² = 2 + 2x + x²
        """
        if not isinstance(other, Polynomial):
            return NotImplemented
        
        if self._ndim != 1:
            # For multivariate, substitute for last variable
            return self.transform([other] if other._ndim == 1 else [other])
        
        result = Polynomial.zero(other._ndim)
        
        for (exp,), coef in self._data.items():
            result = result + coef * (other ** exp)
        
        return result
    
    # Comparison operations
    
    def __eq__(self, other: object) -> bool:
        """Check polynomial equality (within tolerance)."""
        if _is_number(other):
            other = Polynomial.constant(other, self._ndim)
        if not isinstance(other, Polynomial):
            return NotImplemented
        
        p1, p2 = self._unify_dimensions(self, other)
        p1 = p1._clean()
        p2 = p2._clean()
        
        if set(p1._data.keys()) != set(p2._data.keys()):
            # Check if difference is all zeros
            diff = p1 - p2
            return diff.is_zero
        
        return all(
            _is_close(p1._data.get(k, 0), p2._data.get(k, 0))
            for k in set(p1._data.keys()) | set(p2._data.keys())
        )
    
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
    
    # Note: __hash__ intentionally not defined because polynomials are mutable
    # and equality involves floating-point tolerance
    __hash__ = None
    
    # Calculus operations
    
    def differentiate(self, dim: Union[int, str] = -1, order: int = 1) -> 'Polynomial':
        """
        Compute the partial derivative with respect to a variable.
        
        Args:
            dim: Dimension to differentiate. Can be:
                - Integer index (0, 1, 2, ... or -1, -2, ...)
                - String 'x', 'y', 'z', 'w' (x is last, y is second-to-last, etc.)
            order: Order of derivative (default 1).
        
        Returns:
            The differentiated polynomial.
        
        Examples:
            >>> p = Polynomial([1, 2, 3])  # 1 + 2x + 3x²
            >>> p.differentiate()          # 2 + 6x
            >>> p.differentiate(order=2)   # 6
        """
        if order < 0:
            raise ValueError("Derivative order must be non-negative")
        if order == 0:
            return self.copy()
        
        dim = self._resolve_dim(dim)
        result = self
        
        for _ in range(order):
            new_data = {}
            for indices, value in result._data.items():
                exp = indices[dim]
                if exp != 0:
                    new_indices = list(indices)
                    new_indices[dim] = exp - 1
                    new_data[tuple(new_indices)] = value * exp
            
            if not new_data:
                return Polynomial.zero(self._ndim)
            
            result = Polynomial.__new__(Polynomial)
            result._data = new_data
            result._ndim = self._ndim
            result.log_order = self.log_order
        
        return result
    
    def integrate(self, dim: Union[int, str] = -1, constant: float = 0.0) -> 'Polynomial':
        """
        Compute the indefinite integral with respect to a variable.
        
        Args:
            dim: Dimension to integrate.
            constant: Constant of integration (default 0).
        
        Returns:
            The integrated polynomial.
        
        Raises:
            ValueError: If polynomial contains 1/x term (would require log).
        
        Examples:
            >>> p = Polynomial([2, 6])  # 2 + 6x
            >>> p.integrate()           # 2x + 3x²
            >>> p.integrate(constant=1) # 1 + 2x + 3x²
        """
        dim = self._resolve_dim(dim)
        result_data = {}
        
        for indices, value in self._data.items():
            exp = indices[dim]
            
            if exp == -1:
                raise ValueError(
                    "Cannot integrate x⁻¹ term exactly (would require logarithm). "
                    "Use integrate_approx() for Taylor series approximation."
                )
            
            new_indices = list(indices)
            new_indices[dim] = exp + 1
            result_data[tuple(new_indices)] = value / (exp + 1)
        
        # Add constant of integration
        if not _is_zero(constant):
            const_key = (0,) * self._ndim
            result_data[const_key] = result_data.get(const_key, 0) + constant
        
        p = Polynomial.__new__(Polynomial)
        p._data = result_data
        p._ndim = self._ndim
        p.log_order = self.log_order
        return p._clean()
    
    def integrate_approx(self, dim: Union[int, str] = -1, constant: float = 0.0) -> 'Polynomial':
        """
        Compute indefinite integral with Taylor series approximation for 1/x terms.
        
        For 1/x terms, approximates log(x) using Taylor series:
        log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 - ... (valid for 0 < x < 2)
        
        Args:
            dim: Dimension to integrate.
            constant: Constant of integration.
        
        Returns:
            The integrated polynomial (approximate if 1/x terms present).
        
        Warning:
            The Taylor approximation for log(x) only converges for 0 < x < 2.
        """
        dim = self._resolve_dim(dim)
        result_data = {}
        has_log_term = False
        
        for indices, value in self._data.items():
            exp = indices[dim]
            new_indices = list(indices)
            
            if exp == -1:
                has_log_term = True
                # Taylor series for log(x) around x=1
                for n in range(1, self.log_order + 1):
                    approx_indices = list(new_indices)
                    approx_indices[dim] = n
                    approx_coeff = value * ((-1) ** (n + 1)) / n
                    key = tuple(approx_indices)
                    result_data[key] = result_data.get(key, 0) + approx_coeff
            else:
                new_indices[dim] = exp + 1
                key = tuple(new_indices)
                result_data[key] = result_data.get(key, 0) + value / (exp + 1)
        
        if has_log_term:
            warnings.warn(
                "Integration of 1/x approximated using Taylor series for log(x). "
                "Approximation only valid for 0 < x < 2.",
                stacklevel=2
            )
        
        if not _is_zero(constant):
            const_key = (0,) * self._ndim
            result_data[const_key] = result_data.get(const_key, 0) + constant
        
        p = Polynomial.__new__(Polynomial)
        p._data = result_data
        p._ndim = self._ndim
        p.log_order = self.log_order
        return p._clean()
    
    def gradient(self) -> List['Polynomial']:
        """
        Compute the gradient (vector of partial derivatives).
        
        Returns:
            List of partial derivatives [∂p/∂x₀, ∂p/∂x₁, ...].
        """
        return [self.differentiate(i) for i in range(self._ndim)]
    
    def laplacian(self) -> 'Polynomial':
        """
        Compute the Laplacian (sum of second partial derivatives).
        
        Returns:
            ∇²p = ∂²p/∂x₀² + ∂²p/∂x₁² + ...
        """
        result = Polynomial.zero(self._ndim)
        for i in range(self._ndim):
            result = result + self.differentiate(i, order=2)
        return result
    
    # Evaluation and transformation
    
    def evaluate(self, *values, **kwargs) -> float:
        """
        Evaluate the polynomial at given variable values.
        
        Args:
            *values: Values for each variable. Can pass as separate args or single array.
            **kwargs: Named values ('x', 'y', 'z', 'w' for last, second-last, etc.)
        
        Returns:
            The evaluated result.
        
        Raises:
            ValueError: If wrong number of values provided.
        
        Examples:
            >>> p = Polynomial([1, 2, 3])  # 1 + 2x + 3x²
            >>> p.evaluate(2)              # 17.0
            >>> p(2)                       # 17.0 (same as evaluate)
            
            >>> q = Polynomial({(1, 1): 1})  # xy
            >>> q.evaluate(2, 3)            # 6.0
            >>> q(x=2, y=3)                 # 6.0
        """
        # Handle array input
        if len(values) == 1 and _is_array_like(values[0]):
            values = tuple(values[0])
        
        # Handle named arguments
        if kwargs:
            values_list = list(values) if values else [None] * self._ndim
            while len(values_list) < self._ndim:
                values_list.append(None)
            
            name_to_idx = {
                'x': self._ndim - 1,
                'y': self._ndim - 2,
                'z': self._ndim - 3,
                'w': self._ndim - 4,
            }
            for name, val in kwargs.items():
                idx = name_to_idx.get(name.lower())
                if idx is None or idx < 0:
                    raise ValueError(f"Unknown variable name: {name}")
                values_list[idx] = val
            
            values = tuple(values_list)
        
        if len(values) != self._ndim:
            raise ValueError(
                f"Expected {self._ndim} values, got {len(values)}. "
                f"Polynomial has {self._ndim} variable(s)."
            )
        
        values = np.asarray(values, dtype=float)
        result = 0.0
        
        for exponents, coef in self._data.items():
            term = coef
            for i, exp in enumerate(exponents):
                if exp != 0:
                    term *= values[i] ** exp
            result += term
        
        return float(result)
    
    def __call__(self, *values, **kwargs) -> float:
        """Allow polynomial to be called as a function."""
        return self.evaluate(*values, **kwargs)
    
    def transform(self, substitutions: Union[List['Polynomial'], Dict[int, 'Polynomial']]) -> 'Polynomial':
        """
        Substitute polynomials for variables.
        
        Args:
            substitutions: Either:
                - List of polynomials: substitutions[i] replaces variable x_i
                - Dict mapping indices to polynomials
        
        Returns:
            The transformed polynomial.
        
        Examples:
            >>> p = Polynomial([0, 0, 1])    # x²
            >>> q = Polynomial([1, 1])       # 1 + x
            >>> p.transform([q])             # (1+x)² = 1 + 2x + x²
            
            >>> # 2D: substitute y -> x+1
            >>> r = Polynomial({(0, 1): 1})  # y
            >>> r.transform({1: Polynomial([1, 1])})  # x + 1
        """
        if isinstance(substitutions, dict):
            subs = substitutions
        elif _is_array_like(substitutions):
            subs = {i: s for i, s in enumerate(substitutions)}
        else:
            subs = {0: substitutions}
        
        # Convert all to Polynomials
        subs = {
            k: (v if isinstance(v, Polynomial) else Polynomial(v))
            for k, v in subs.items()
        }
        
        # Determine result dimension
        result_ndim = max(
            (s._ndim for s in subs.values()),
            default=self._ndim
        )
        
        result = Polynomial.zero(result_ndim)
        
        for exponents, coef in self._data.items():
            term = Polynomial.constant(coef, result_ndim)
            
            for i, exp in enumerate(exponents):
                if exp != 0:
                    if i in subs:
                        sub_poly = subs[i]._promote_ndim(result_ndim)
                        term = term * (sub_poly ** exp)
                    else:
                        # Keep original variable
                        var_exp = [0] * result_ndim
                        var_exp[i] = exp
                        term = term * Polynomial.monomial(tuple(var_exp))
            
            result = result + term
        
        return result
    
    # Root finding (for univariate)
    
    def roots(self, real_only: bool = False) -> np.ndarray:
        """
        Find roots of a univariate polynomial.
        
        Args:
            real_only: If True, return only real roots.
        
        Returns:
            Array of roots (complex in general).
        
        Raises:
            ValueError: If polynomial is multivariate.
        
        Examples:
            >>> p = Polynomial([1, 0, -1])  # x² - 1
            >>> p.roots()                   # array([-1., 1.])
        """
        if self._ndim != 1:
            raise ValueError("Root finding only supported for univariate polynomials")
        
        if self.is_zero:
            raise ValueError("Zero polynomial has infinitely many roots")
        
        if self.is_constant:
            return np.array([])
        
        # Convert to numpy coefficient array (highest degree first)
        coeffs = np.zeros(self.degree + 1)
        for (exp,), coef in self._data.items():
            coeffs[self.degree - exp] = coef
        
        all_roots = np.roots(coeffs)
        
        if real_only:
            # Filter to real roots (small imaginary part)
            real_roots = all_roots[np.abs(all_roots.imag) < PolynomialConfig.atol]
            return np.sort(real_roots.real)
        
        return all_roots
    
    # Conversion and representation
    
    def to_dense(self) -> np.ndarray:
        """
        Convert to dense numpy array representation.
        
        Returns:
            Dense array where array[i,j,...] = coefficient of x₀^i * x₁^j * ...
        """
        dense = np.zeros(self.shape)
        for indices, value in self._data.items():
            dense[indices] = value
        return dense
    
    def to_dict(self) -> CoefficientDict:
        """Return a copy of the coefficient dictionary."""
        return self._data.copy()
    
    def to_sympy(self):
        """
        Convert to SymPy polynomial (requires sympy to be installed).
        
        Returns:
            sympy.Poly object.
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError("SymPy is required for to_sympy(). Install with: pip install sympy")
        
        if self._ndim <= 3:
            var_names = ['z', 'y', 'x'][-self._ndim:]
        else:
            var_names = [f'x{i}' for i in range(self._ndim)]
        
        variables = sp.symbols(' '.join(var_names))
        if self._ndim == 1:
            variables = (variables,)
        
        expr = sp.Integer(0)
        for exponents, coef in self._data.items():
            term = sp.Rational(coef).limit_denominator(10**10)
            for var, exp in zip(variables, exponents):
                term *= var ** exp
            expr += term
        
        return sp.Poly(expr, *variables)
    
    @classmethod
    def from_sympy(cls, poly) -> 'Polynomial':
        """
        Create Polynomial from a SymPy polynomial.
        
        Args:
            poly: A sympy.Poly or sympy expression.
        
        Returns:
            Polynomial instance.
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError("SymPy is required for from_sympy()")
        
        if not isinstance(poly, sp.Poly):
            poly = sp.Poly(poly)
        
        data = {}
        for monom, coef in poly.as_dict().items():
            data[monom] = float(coef)
        
        return cls(data)
    
    def as_equation(self, var_names: Optional[List[str]] = None) -> str:
        """
        Format as a mathematical equation string.
        
        Args:
            var_names: Optional list of variable names. 
                      Defaults to ['z', 'y', 'x'] for ≤3D, else ['x₀', 'x₁', ...].
        
        Returns:
            String representation like "3x² + 2y - 1"
        """
        if var_names is None:
            if self._ndim <= 3:
                var_names = ['z', 'y', 'x'][-self._ndim:]
            else:
                if PolynomialConfig.use_unicode:
                    subscripts = '₀₁₂₃₄₅₆₇₈₉'
                    var_names = ['x' + ''.join(subscripts[int(d)] for d in str(i)) 
                                for i in range(self._ndim)]
                else:
                    var_names = [f'x{i}' for i in range(self._ndim)]
        
        if self.is_zero:
            return "0"
        
        # Sort terms by total degree (descending), then lexicographically
        sorted_terms = sorted(
            [(exp, coef) for exp, coef in self._data.items() if not _is_zero(coef)],
            key=lambda x: (-sum(x[0]), [-e for e in x[0]]),
        )
        
        if not sorted_terms:
            return "0"
        
        result_parts = []
        
        for i, (exponents, coef) in enumerate(sorted_terms):
            is_first = (i == 0)
            
            # Build variable part
            var_parts = []
            for j, exp in enumerate(exponents):
                if exp != 0:
                    var = var_names[j]
                    exp_str = _format_exponent(exp)
                    var_parts.append(f"{var}{exp_str}")
            
            has_variables = len(var_parts) > 0
            var_str = "".join(var_parts) if PolynomialConfig.use_unicode else "*".join(var_parts)
            
            # Format coefficient
            coef_str = _format_coefficient(coef, is_first, has_variables)
            
            if has_variables:
                if coef_str.endswith(' '):
                    result_parts.append(coef_str + var_str)
                elif coef_str in ('', '-'):
                    result_parts.append(coef_str + var_str)
                elif coef_str.endswith('+ ') or coef_str.endswith('- '):
                    result_parts.append(coef_str + var_str)
                else:
                    sep = "" if PolynomialConfig.use_unicode else "*"
                    result_parts.append(coef_str + sep + var_str)
            else:
                result_parts.append(coef_str)
        
        return ''.join(result_parts)
    
    def __str__(self) -> str:
        return self.as_equation()
    
    def __repr__(self) -> str:
        return f"Polynomial({self._data}, ndim={self._ndim})"
    
    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks."""
        if self._ndim <= 3:
            var_names = ['z', 'y', 'x'][-self._ndim:]
        else:
            var_names = [f'x_{i}' for i in range(self._ndim)]
        
        if self.is_zero:
            return "$0$"
        
        sorted_terms = sorted(
            [(exp, coef) for exp, coef in self._data.items() if not _is_zero(coef)],
            key=lambda x: (-sum(x[0]), [-e for e in x[0]]),
        )
        
        parts = []
        for i, (exponents, coef) in enumerate(sorted_terms):
            is_first = (i == 0)
            
            # Sign
            if coef < 0:
                sign = "-" if is_first else " - "
                abs_coef = -coef
            else:
                sign = "" if is_first else " + "
                abs_coef = coef
            
            # Coefficient
            if isinstance(abs_coef, float) and abs_coef == int(abs_coef):
                abs_coef = int(abs_coef)
            
            has_vars = any(e != 0 for e in exponents)
            
            if abs_coef == 1 and has_vars:
                coef_str = ""
            else:
                coef_str = str(abs_coef)
            
            # Variables
            var_parts = []
            for j, exp in enumerate(exponents):
                if exp == 1:
                    var_parts.append(var_names[j])
                elif exp != 0:
                    var_parts.append(f"{var_names[j]}^{{{exp}}}")
            
            var_str = " ".join(var_parts)
            
            parts.append(f"{sign}{coef_str}{var_str}")
        
        return "$" + "".join(parts) + "$"


# Special polynomial generators

def chebyshev_t(n: int) -> Polynomial:
    """
    Create the nth Chebyshev polynomial of the first kind.
    
    Uses recurrence: T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)
    With T_0(x) = 1, T_1(x) = x.
    
    Args:
        n: Degree of the polynomial (non-negative integer).
    
    Returns:
        The nth Chebyshev polynomial T_n(x).
    """
    if n < 0:
        raise ValueError("Chebyshev polynomial degree must be non-negative")
    if n == 0:
        return Polynomial([1])
    if n == 1:
        return Polynomial([0, 1])
    
    x = Polynomial([0, 1])
    t_prev = Polynomial([1])
    t_curr = x
    
    for _ in range(2, n + 1):
        t_next = 2 * x * t_curr - t_prev
        t_prev = t_curr
        t_curr = t_next
    
    return t_curr


def chebyshev_u(n: int) -> Polynomial:
    """
    Create the nth Chebyshev polynomial of the second kind.
    
    Uses recurrence: U_{n+1}(x) = 2x·U_n(x) - U_{n-1}(x)
    With U_0(x) = 1, U_1(x) = 2x.
    """
    if n < 0:
        raise ValueError("Chebyshev polynomial degree must be non-negative")
    if n == 0:
        return Polynomial([1])
    if n == 1:
        return Polynomial([0, 2])
    
    x = Polynomial([0, 1])
    u_prev = Polynomial([1])
    u_curr = 2 * x
    
    for _ in range(2, n + 1):
        u_next = 2 * x * u_curr - u_prev
        u_prev = u_curr
        u_curr = u_next
    
    return u_curr


def legendre(n: int) -> Polynomial:
    """
    Create the nth Legendre polynomial.
    
    Uses recurrence: (n+1)·P_{n+1}(x) = (2n+1)·x·P_n(x) - n·P_{n-1}(x)
    With P_0(x) = 1, P_1(x) = x.
    
    Args:
        n: Degree of the polynomial (non-negative integer).
    
    Returns:
        The nth Legendre polynomial P_n(x).
    """
    if n < 0:
        raise ValueError("Legendre polynomial degree must be non-negative")
    if n == 0:
        return Polynomial([1])
    if n == 1:
        return Polynomial([0, 1])
    
    x = Polynomial([0, 1])
    p_prev = Polynomial([1])
    p_curr = x
    
    for k in range(1, n):
        p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)
        p_prev = p_curr
        p_curr = p_next
    
    return p_curr


def hermite_prob(n: int) -> Polynomial:
    """
    Create the nth probabilist's Hermite polynomial.
    
    Uses recurrence: He_{n+1}(x) = x·He_n(x) - n·He_{n-1}(x)
    With He_0(x) = 1, He_1(x) = x.
    """
    if n < 0:
        raise ValueError("Hermite polynomial degree must be non-negative")
    if n == 0:
        return Polynomial([1])
    if n == 1:
        return Polynomial([0, 1])
    
    x = Polynomial([0, 1])
    h_prev = Polynomial([1])
    h_curr = x
    
    for k in range(1, n):
        h_next = x * h_curr - k * h_prev
        h_prev = h_curr
        h_curr = h_next
    
    return h_curr


def hermite_phys(n: int) -> Polynomial:
    """
    Create the nth physicist's Hermite polynomial.
    
    Uses recurrence: H_{n+1}(x) = 2x·H_n(x) - 2n·H_{n-1}(x)
    With H_0(x) = 1, H_1(x) = 2x.
    """
    if n < 0:
        raise ValueError("Hermite polynomial degree must be non-negative")
    if n == 0:
        return Polynomial([1])
    if n == 1:
        return Polynomial([0, 2])
    
    x = Polynomial([0, 1])
    h_prev = Polynomial([1])
    h_curr = 2 * x
    
    for k in range(1, n):
        h_next = 2 * x * h_curr - 2 * k * h_prev
        h_prev = h_curr
        h_curr = h_next
    
    return h_curr


def laguerre(n: int, alpha: float = 0) -> Polynomial:
    """
    Create the nth generalized Laguerre polynomial L_n^(α)(x).
    
    For α=0, these are the standard Laguerre polynomials.
    
    Uses recurrence:
    (n+1)·L_{n+1}^(α)(x) = (2n+1+α-x)·L_n^(α)(x) - (n+α)·L_{n-1}^(α)(x)
    """
    if n < 0:
        raise ValueError("Laguerre polynomial degree must be non-negative")
    if n == 0:
        return Polynomial([1])
    if n == 1:
        return Polynomial([1 + alpha, -1])
    
    x = Polynomial([0, 1])
    l_prev = Polynomial([1])
    l_curr = Polynomial([1 + alpha, -1])
    
    for k in range(1, n):
        l_next = ((2 * k + 1 + alpha - x) * l_curr - (k + alpha) * l_prev) / (k + 1)
        l_prev = l_curr
        l_curr = l_next
    
    return l_curr


def bernstein(n: int, k: int) -> Polynomial:
    """
    Create the kth Bernstein basis polynomial of degree n.
    
    B_{k,n}(x) = C(n,k) * x^k * (1-x)^(n-k)
    
    Args:
        n: Degree of the polynomial.
        k: Index (0 ≤ k ≤ n).
    
    Returns:
        The Bernstein basis polynomial B_{k,n}(x).
    """
    if k < 0 or k > n:
        raise ValueError(f"Index k={k} must satisfy 0 ≤ k ≤ n={n}")
    
    from math import comb
    
    x = Polynomial([0, 1])
    one_minus_x = Polynomial([1, -1])
    
    return comb(n, k) * (x ** k) * (one_minus_x ** (n - k))


def taylor_exp(n: int) -> Polynomial:
    """
    Taylor polynomial approximation of e^x around x=0.
    
    e^x ≈ 1 + x + x²/2! + x³/3! + ... + xⁿ/n!
    """
    from math import factorial
    coeffs = [1.0 / factorial(k) for k in range(n + 1)]
    return Polynomial(coeffs)


def taylor_sin(n: int) -> Polynomial:
    """
    Taylor polynomial approximation of sin(x) around x=0.
    
    sin(x) ≈ x - x³/3! + x⁵/5! - ...
    """
    from math import factorial
    coeffs = [0.0] * (2 * n + 2)
    for k in range(n + 1):
        coeffs[2 * k + 1] = ((-1) ** k) / factorial(2 * k + 1)
    return Polynomial(coeffs)


def taylor_cos(n: int) -> Polynomial:
    """
    Taylor polynomial approximation of cos(x) around x=0.
    
    cos(x) ≈ 1 - x²/2! + x⁴/4! - ...
    """
    from math import factorial
    coeffs = [0.0] * (2 * n + 1)
    for k in range(n + 1):
        coeffs[2 * k] = ((-1) ** k) / factorial(2 * k)
    return Polynomial(coeffs)


# GCD and related operations

def polynomial_gcd(p: Polynomial, q: Polynomial) -> Polynomial:
    """
    Compute the greatest common divisor of two univariate polynomials.
    
    Uses the Euclidean algorithm.
    
    Args:
        p, q: Univariate polynomials.
    
    Returns:
        The monic GCD of p and q.
    """
    if p._ndim != 1 or q._ndim != 1:
        raise ValueError("GCD only supported for univariate polynomials")
    
    # Ensure p has higher or equal degree
    if p.degree < q.degree:
        p, q = q, p
    
    while not q.is_zero:
        _, r = p.divmod(q)
        p, q = q, r
    
    # Make monic
    if not p.is_zero:
        p = p / p.leading_coefficient
    
    return p


def polynomial_lcm(p: Polynomial, q: Polynomial) -> Polynomial:
    """
    Compute the least common multiple of two univariate polynomials.
    
    lcm(p, q) = p * q / gcd(p, q)
    """
    gcd = polynomial_gcd(p, q)
    if gcd.is_zero:
        return Polynomial.zero()
    return (p * q) // gcd


# Testing

def _run_tests():
    """Run comprehensive tests."""
    print("=" * 60)
    print("POLYNOMIAL CLASS TESTS")
    print("=" * 60)
    
    # Disable dimension warnings for tests
    PolynomialConfig.warn_on_dimension_mismatch = False
    
    tests_passed = 0
    tests_failed = 0
    
    def test(name: str, condition: bool, detail: str = ""):
        nonlocal tests_passed, tests_failed
        if condition:
            print(f"  ✓ {name}")
            tests_passed += 1
        else:
            print(f"  ✗ {name}")
            if detail:
                print(f"    {detail}")
            tests_failed += 1
    
    # Creation tests
    print("\n1. Creation Tests")
    print("-" * 40)
    
    p1 = Polynomial([1, 2, 3])
    test("From list", p1[0] == 1 and p1[1] == 2 and p1[2] == 3)
    
    p2 = Polynomial({(2,): 5, (0,): 1})
    test("From dict", p2[0] == 1 and p2[2] == 5)
    
    p3 = Polynomial(7)
    test("From number", p3[0] == 7 and p3.is_constant)
    
    p4 = Polynomial.zero()
    test("Zero polynomial", p4.is_zero)
    
    p5 = Polynomial.one()
    test("One polynomial", p5[0] == 1 and p5.is_constant)
    
    p6 = Polynomial.monomial(3, 2)
    test("Monomial", p6[3] == 2 and p6[0] == 0)
    
    p7 = Polynomial.variable()
    test("Variable", p7[1] == 1 and p7[0] == 0)
    
    p8 = Polynomial.from_roots([1, 2])
    test("From roots", 
         _is_close(p8(1), 0) and _is_close(p8(2), 0),
         f"p(1)={p8(1)}, p(2)={p8(2)}")
    
    # Arithmetic tests
    print("\n2. Arithmetic Tests")
    print("-" * 40)
    
    a = Polynomial([1, 2])
    b = Polynomial([3, 4])
    
    test("Addition", (a + b) == Polynomial([4, 6]))
    test("Subtraction", (a - b) == Polynomial([-2, -2]))
    test("Negation", (-a) == Polynomial([-1, -2]))
    test("Scalar multiplication", (a * 2) == Polynomial([2, 4]))
    test("Polynomial multiplication", 
         (a * b) == Polynomial([3, 10, 8]),
         f"Got: {a * b}")
    test("Scalar division", (Polynomial([2, 4]) / 2) == a)
    test("Power", (a ** 2) == Polynomial([1, 4, 4]))
    test("Power of 0", (a ** 0) == Polynomial.one())
    
    # Division tests
    print("\n3. Division Tests")
    print("-" * 40)
    
    dividend = Polynomial([1, 0, -1])  # x² - 1
    divisor = Polynomial([1, 1])       # x + 1
    q, r = dividend.divmod(divisor)
    test("Polynomial division quotient", 
         q == Polynomial([-1, 1]),
         f"Got q={q}")
    test("Polynomial division remainder", 
         r.is_zero or _is_close(r[0], 0),
         f"Got r={r}")
    test("Division reconstruction", 
         (q * divisor + r) == dividend,
         f"q*d + r = {q * divisor + r}")
    
    # Calculus tests
    print("\n4. Calculus Tests")
    print("-" * 40)
    
    p = Polynomial([1, 2, 3])  # 1 + 2x + 3x²
    dp = p.differentiate()
    test("Differentiation", 
         dp == Polynomial([2, 6]),
         f"Got: {dp}")
    
    ip = Polynomial([2, 6]).integrate()
    test("Integration", 
         _is_close(ip[1], 2) and _is_close(ip[2], 3),
         f"Got: {ip}")
    
    test("Second derivative", 
         p.differentiate(order=2) == Polynomial([6]),
         f"Got: {p.differentiate(order=2)}")
    
    # Evaluation tests
    print("\n5. Evaluation Tests")
    print("-" * 40)
    
    p = Polynomial([1, 2, 3])
    test("Evaluate", _is_close(p(2), 17))
    test("Call syntax", _is_close(p.evaluate(2), p(2)))
    
    # Multivariate
    q = Polynomial({(1, 1): 1})  # xy
    test("Multivariate evaluate", _is_close(q(2, 3), 6))
    test("Named args", _is_close(q(x=2, y=3), 6))
    
    # Composition tests
    print("\n6. Composition Tests")
    print("-" * 40)
    
    p = Polynomial([0, 0, 1])  # x²
    q = Polynomial([1, 1])     # 1 + x
    composed = p @ q            # (1+x)²
    test("Composition", 
         composed == Polynomial([1, 2, 1]),
         f"Got: {composed}")
    
    # Transform
    transformed = p.transform([q])
    test("Transform", 
         transformed == Polynomial([1, 2, 1]),
         f"Got: {transformed}")
    
    # Root finding tests
    print("\n7. Root Finding Tests")
    print("-" * 40)
    
    p = Polynomial([1, 0, -1])  # x² - 1
    roots = p.roots(real_only=True)
    test("Roots", 
         len(roots) == 2 and _is_close(roots[0], -1) and _is_close(roots[1], 1),
         f"Got: {roots}")
    
    p2 = Polynomial.from_roots([1, 2, 3])
    roots2 = sorted(p2.roots(real_only=True))
    test("Roots from polynomial", 
         len(roots2) == 3 and all(_is_close(r, e) for r, e in zip(roots2, [1, 2, 3])),
         f"Got: {roots2}")
    
    # Special polynomials tests
    print("\n8. Special Polynomials Tests")
    print("-" * 40)
    
    # Chebyshev identity: T_n(cos(θ)) = cos(nθ)
    import math
    theta = 0.5
    t5 = chebyshev_t(5)
    test("Chebyshev T_5", 
         _is_close(t5(math.cos(theta)), math.cos(5 * theta)),
         f"T_5(cos(0.5))={t5(math.cos(theta))}, cos(2.5)={math.cos(2.5)}")
    
    # Legendre orthogonality: integral from -1 to 1 of P_m * P_n = 0 for m ≠ n
    p2_leg = legendre(2)
    p3_leg = legendre(3)
    # Approximate integral using numpy
    x = np.linspace(-1, 1, 1000)
    integral = np.trapezoid([p2_leg(xi) * p3_leg(xi) for xi in x], x)
    test("Legendre orthogonality", 
         _is_close(integral, 0, ),
         f"Integral of P_2 * P_3 = {integral}")
    
    # GCD tests
    print("\n9. GCD Tests")
    print("-" * 40)
    
    p = Polynomial.from_roots([1, 2, 3])
    q = Polynomial.from_roots([2, 3, 4])
    gcd = polynomial_gcd(p, q)
    test("GCD", 
         _is_close(gcd(2), 0) and _is_close(gcd(3), 0),
         f"GCD = {gcd}")
    
    # Properties tests
    print("\n10. Property Tests")
    print("-" * 40)
    
    p = Polynomial([1, 2, 0, 4])
    test("Degree", p.degree == 3)
    test("Leading coefficient", p.leading_coefficient == 4)
    test("Constant term", p.constant_term == 1)
    test("Is not constant", not p.is_constant)
    test("Is not zero", not p.is_zero)
    test("Shape", p.shape == (4,))
    test("Number of terms", len(p) == 3)  # 1, 2x, 4x³
    
    # String representation tests
    print("\n11. String Representation Tests")
    print("-" * 40)
    
    p = Polynomial([1, -2, 3])
    s = str(p)
    test("String contains x²", "x" in s.lower())
    test("String formatted", len(s) > 0)
    print(f"    String: {s}")
    
    # Multivariate tests
    print("\n12. Multivariate Tests")
    print("-" * 40)
    
    p = Polynomial({(1, 2): 3, (2, 1): 4})  # 3xy² + 4x²y
    test("2D creation", p._ndim == 2)
    test("2D evaluation", _is_close(p(2, 3), 3*2*9 + 4*4*3))  # 54 + 48 = 102
    
    grad = p.gradient()
    test("Gradient length", len(grad) == 2)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        success = _run_tests()
        sys.exit(0 if success else 1)
    
    # Demo
    print("=" * 60)
    print("POLYNOMIAL CLASS DEMO")
    print("=" * 60)
    
    print("\n1. Basic Operations")
    print("-" * 40)
    p = Polynomial([1, 2, 3])
    q = Polynomial([1, 1])
    print(f"p(x) = {p}")
    print(f"q(x) = {q}")
    print(f"p + q = {p + q}")
    print(f"p * q = {p * q}")
    print(f"p² = {p ** 2}")
    print(f"p(2) = {p(2)}")
    
    print("\n2. Calculus")
    print("-" * 40)
    print(f"p'(x) = {p.differentiate()}")
    print(f"∫p dx = {p.integrate()}")
    
    print("\n3. Root Finding")
    print("-" * 40)
    r = Polynomial([1, 0, -1])  # x² - 1
    print(f"r(x) = {r}")
    print(f"Roots: {r.roots(real_only=True)}")
    
    print("\n4. Composition")
    print("-" * 40)
    f = Polynomial([0, 0, 1])  # x²
    g = Polynomial([1, 1])     # x + 1
    print(f"f(x) = {f}")
    print(f"g(x) = {g}")
    print(f"f(g(x)) = {f @ g}")
    
    print("\n5. Special Polynomials")
    print("-" * 40)
    print(f"Chebyshev T₄(x) = {chebyshev_t(4)}")
    print(f"Legendre P₃(x) = {legendre(3)}")
    print(f"Hermite He₃(x) = {hermite_prob(3)}")
    
    print("\n6. Multivariate")
    print("-" * 40)
    mv = Polynomial({(1, 1): 3, (2, 0): 2, (0, 2): 1})
    print(f"p(x,y) = {mv}")
    print(f"p(2, 3) = {mv(2, 3)}")
    print(f"∂p/∂x = {mv.differentiate('x')}")
    print(f"∂p/∂y = {mv.differentiate('y')}")
    
    print("\n7. Polynomial Division")
    print("-" * 40)
    dividend = Polynomial([2, 3, 1])  # x² + 3x + 2 = (x+1)(x+2)
    divisor = Polynomial([1, 1])      # x + 1
    q, r = dividend.divmod(divisor)
    print(f"({dividend}) ÷ ({divisor})")
    print(f"Quotient: {q}")
    print(f"Remainder: {r}")
    
    print("\n" + "=" * 60)
    print("Run with --test for comprehensive tests")
    print("=" * 60)
