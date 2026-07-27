"""
Calculator MCP Server
======================
A Model Context Protocol (MCP) server built with FastMCP that exposes
basic and advanced calculator operations as tools.

Setup:
    pip install fastmcp

Run (stdio transport - default, for use with Claude Desktop / MCP clients):
    python calculator_mcp_server.py

Run (HTTP transport, e.g. for testing with curl or web clients):
    python calculator_mcp_server.py --http
"""

import math
import sys

from fastmcp import FastMCP

# Create the MCP server instance
mcp = FastMCP("Calculator")


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers together and return the sum."""
    return a + b


@mcp.tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first and return the difference."""
    return a - b


@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return the product."""
    return a * b


@mcp.tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second and return the quotient.

    Raises a ValueError if b is zero.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@mcp.tool
def modulo(a: float, b: float) -> float:
    """Return the remainder of a divided by b (a % b).

    Raises a ValueError if b is zero.
    """
    if b == 0:
        raise ValueError("Modulo by zero is not allowed.")
    return a % b


# ---------------------------------------------------------------------------
# Powers, roots, and logarithms
# ---------------------------------------------------------------------------

@mcp.tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent (base ** exponent)."""
    return base ** exponent


@mcp.tool
def square_root(a: float) -> float:
    """Return the square root of a.

    Raises a ValueError if a is negative.
    """
    if a < 0:
        raise ValueError("Cannot take the square root of a negative number.")
    return math.sqrt(a)


@mcp.tool
def nth_root(a: float, n: float) -> float:
    """Return the nth root of a (e.g. n=3 gives the cube root).

    Raises a ValueError if a is negative and n is even, or if n is zero.
    """
    if n == 0:
        raise ValueError("Root degree n cannot be zero.")
    if a < 0 and n % 2 == 0:
        raise ValueError("Cannot take an even root of a negative number.")
    if a < 0:
        return -((-a) ** (1 / n))
    return a ** (1 / n)


@mcp.tool
def log(a: float, base: float = math.e) -> float:
    """Return the logarithm of a with the given base (defaults to natural log, base e).

    Raises a ValueError if a <= 0 or base <= 0 or base == 1.
    """
    if a <= 0:
        raise ValueError("Logarithm is undefined for values <= 0.")
    if base <= 0 or base == 1:
        raise ValueError("Logarithm base must be positive and not equal to 1.")
    return math.log(a, base)


# ---------------------------------------------------------------------------
# Trigonometry (angles in degrees for convenience)
# ---------------------------------------------------------------------------

@mcp.tool
def sin(degrees: float) -> float:
    """Return the sine of an angle given in degrees."""
    return math.sin(math.radians(degrees))


@mcp.tool
def cos(degrees: float) -> float:
    """Return the cosine of an angle given in degrees."""
    return math.cos(math.radians(degrees))


@mcp.tool
def tan(degrees: float) -> float:
    """Return the tangent of an angle given in degrees."""
    return math.tan(math.radians(degrees))


# ---------------------------------------------------------------------------
# Other common operations
# ---------------------------------------------------------------------------

@mcp.tool
def factorial(n: int) -> int:
    """Return the factorial of a non-negative integer n.

    Raises a ValueError if n is negative or not an integer.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    return math.factorial(n)


@mcp.tool
def absolute_value(a: float) -> float:
    """Return the absolute value of a."""
    return abs(a)


@mcp.tool
def average(numbers: list[float]) -> float:
    """Return the arithmetic mean of a list of numbers.

    Raises a ValueError if the list is empty.
    """
    if not numbers:
        raise ValueError("Cannot compute the average of an empty list.")
    return sum(numbers) / len(numbers)


@mcp.tool
def percentage(part: float, whole: float) -> float:
    """Return what percentage 'part' is of 'whole'.

    Raises a ValueError if whole is zero.
    """
    if whole == 0:
        raise ValueError("Whole cannot be zero when computing a percentage.")
    return (part / whole) * 100


@mcp.tool
def evaluate_expression(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression string, e.g. '3 + 4 * (2 - 1)'.

    Only numbers and the operators + - * / % ** and parentheses are allowed.
    Raises a ValueError if the expression contains disallowed characters
    or cannot be evaluated.
    """
    allowed_chars = set("0123456789.+-*/()% \t")
    if not set(expression) <= allowed_chars:
        raise ValueError("Expression contains disallowed characters.")
    try:
        # eval is safe here because we've whitelisted characters above
        # (no names, no attribute access, no builtins reachable).
        result = eval(expression, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ValueError("Division by zero in expression.")
    except Exception as exc:
        raise ValueError(f"Could not evaluate expression: {exc}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--http" in sys.argv:
        # Run as an HTTP server (default host 127.0.0.1, port 8000)
        mcp.run(transport="http", host="127.0.0.1", port=8000)
    else:
        # Default: stdio transport, for MCP clients like Claude Desktop
        mcp.run()