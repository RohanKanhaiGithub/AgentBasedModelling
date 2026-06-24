"""Compatibility wrapper for code that imports plots.py.

Input:
    Any import that expects a module named plots.

Output:
    The plotting functions from plot.py.
"""

from plot import *  # noqa: F401,F403
