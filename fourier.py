# Imports
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
import math

# Integrals
from scipy.integrate import quad
import sympy as smp
import warnings

warnings.filterwarnings("ignore")


# Fourier Coefficients
def fourier_a0(f, T):
    func = lambda t: f(t)
    a0, _ = quad(func, 0, T)
    a0 = (1 / T) * a0

    return a0


def symbolic_a0(array, T):
    t = smp.symbols("t", real=True)
    r = lambda t, low, high: (t > low) & (t < high)
    piecewise_expr = []

    # Build the Piecewise function from the array
    for segment in array:
        equation_lambda_numpy = lambda t: eval(
            segment["equation"].get()
        )  # Lambda function with numpy expressions
        equation_sympy = smp.sympify(equation_lambda_numpy(t))
        lower_bound = float(segment["lower_bound"].get())
        upper_bound = float(segment["upper_bound"].get())
        piecewise_expr.append((equation_sympy, r(t, lower_bound, upper_bound)))

    piecewise_expr.append((0, True))

    return (
        1
        / T
        * smp.integrate(
            smp.Piecewise(*piecewise_expr), (t, 0, T), conds="none"
        ).simplify()
    )


def fourier_an(f, T, n):
    w = (2 * np.pi) / T
    func = lambda t: f(t) * np.cos(n * w * t)
    an, _ = quad(func, 0, T)
    an = (2 / T) * an

    return an


def symbolic_an(array, T):
    t = smp.symbols("t", real=True)
    n = smp.symbols("n", real=True)
    w = (2 * smp.pi) / T
    f = smp.cos(n * w * t)
    r = lambda t, low, high: (t > low) & (t < high)
    piecewise_expr = []

    # Build the Piecewise function from the array
    for segment in array:
        equation_lambda_numpy = lambda t: eval(
            segment["equation"].get()
        )  # Lambda function with numpy expressions
        equation_sympy = smp.sympify(equation_lambda_numpy(t))
        lower_bound = float(segment["lower_bound"].get())
        upper_bound = float(segment["upper_bound"].get())
        piecewise_expr.append((equation_sympy, r(t, lower_bound, upper_bound)))

    piecewise_expr.append((0, True))

    return (
        2
        / T
        * smp.integrate(
            smp.Piecewise(*piecewise_expr) * f, (t, 0, T), conds="none"
        ).simplify()
    )

def symbolic_an_n(array, T, n):
    t = smp.symbols("t", real=True)
    w = (2 * smp.pi) / T
    f = smp.cos(n * w * t)
    r = lambda t, low, high: (t > low) & (t < high)
    piecewise_expr = []

    # Build the Piecewise function from the array
    for segment in array:
        equation_lambda_numpy = lambda t: eval(
            segment["equation"].get()
        )  # Lambda function with numpy expressions
        equation_sympy = smp.sympify(equation_lambda_numpy(t))
        lower_bound = float(segment["lower_bound"].get())
        upper_bound = float(segment["upper_bound"].get())
        piecewise_expr.append((equation_sympy, r(t, lower_bound, upper_bound)))

    piecewise_expr.append((0, True))

    return (
        2
        / T
        * smp.integrate(
            smp.Piecewise(*piecewise_expr) * f, (t, 0, T), conds="none"
        ).simplify()
    )

def fourier_bn(f, T, n):
    w = (2 * np.pi) / T
    func = lambda t: f(t) * np.sin(n * w * t)
    bn, _ = quad(func, 0, T)
    bn = (2 / T) * bn

    return bn


def symbolic_bn(array, T):
    t = smp.symbols("t", real=True)
    n = smp.symbols("n", real=True)
    w = (2 * smp.pi) / T
    f = smp.sin(n * w * t)
    r = lambda t, low, high: (t > low) & (t < high)
    piecewise_expr = []

    # Build the Piecewise function from the array
    for segment in array:
        equation_lambda_numpy = lambda t: eval(
            segment["equation"].get()
        )  # Lambda function with numpy expressions
        equation_sympy = smp.sympify(equation_lambda_numpy(t))
        lower_bound = float(segment["lower_bound"].get())
        upper_bound = float(segment["upper_bound"].get())
        piecewise_expr.append((equation_sympy, r(t, lower_bound, upper_bound)))

    piecewise_expr.append((0, True))

    return (
        2
        / T
        * smp.integrate(
            smp.Piecewise(*piecewise_expr) * f, (t, 0, T), conds="none"
        ).simplify()
    )

def symbolic_bn_n(array, T, n):
    t = smp.symbols("t", real=True)
    w = (2 * smp.pi) / T
    f = smp.sin(n * w * t)
    r = lambda t, low, high: (t > low) & (t < high)
    piecewise_expr = []

    # Build the Piecewise function from the array
    for segment in array:
        equation_lambda_numpy = lambda t: eval(
            segment["equation"].get()
        )  # Lambda function with numpy expressions
        equation_sympy = smp.sympify(equation_lambda_numpy(t))
        lower_bound = float(segment["lower_bound"].get())
        upper_bound = float(segment["upper_bound"].get())
        piecewise_expr.append((equation_sympy, r(t, lower_bound, upper_bound)))

    piecewise_expr.append((0, True))

    return (
        2
        / T
        * smp.integrate(
            smp.Piecewise(*piecewise_expr) * f, (t, 0, T), conds="none"
        ).simplify()
    )


# Signal energy
def energy_f(array, T):
    t = smp.symbols("t", real=True)
    r = lambda t, low, high: (t > low) & (t < high)
    piecewise_expr = []

    # Build the Piecewise function from the array
    for segment in array:
      equation_lambda_numpy = lambda t: eval(segment["equation"].get())
      equation_sympy = smp.sympify(equation_lambda_numpy(t))
      lower_bound = float(segment["lower_bound"].get())
      upper_bound = float(segment["upper_bound"].get())
      piecewise_expr.append((equation_sympy, r(t, lower_bound, upper_bound)))

    piecewise_expr.append((0, True))

    return smp.integrate(smp.Piecewise(*piecewise_expr)**2, (t, 0, T), conds="none").simplify()


# Integral cuadrada del error (ICE)
def ice(array, T):
    """
    Determines the value of N.

    Parameters
    ----------
    f : lambda function
    T : int
        period of f

    Returns
    ------
    int
      The value of N that truncates the Fourier Series
    """
    # E_f
    energy = energy_f(array, T)
    # a_0
    a_0 = symbolic_a0(array, T)
    # Return value
    N = 0
    # Condition for ending the while loop
    conditional = 0.02 * energy
    condition = 5.0

    # Finds the value of N
    while math.isclose(conditional, condition) or conditional<=condition:
        # Calculate the sumation
        result = 0
        for n in range(1, N + 1):
            an = symbolic_an_n(array, T, n)
            bn = symbolic_bn_n(array, T, n)
            result += an**2 + bn**2

        condition = energy - ((a_0**2 * T) + (T / 2) * result)
        N += 1

    return N


# Serie de fourier
def fourier_suma_parcial(f, T, N):
    # funcion a retornar
    def s(t):
        a0 = fourier_a0(f, T)
        # resultado de la suma
        sum = 0
        for i in range(1, N + 1):
            a_i = fourier_an(f, T, i)
            b_i = fourier_bn(f, T, i)
            w_i = (2 * i * np.pi) / T

            def part_1(t):
                return a_i * np.cos(w_i * t)

            def part_2(t):
                return b_i * np.sin(w_i * t)

            sum += part_1(t) + part_2(t)

        res = a0 + sum
        return res

    return s
