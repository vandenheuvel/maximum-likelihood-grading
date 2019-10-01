from functools import partial

import numpy as np
from numpy import cos, cosh, sin, sinh, pi


number_of_series_terms_to_evaluate = 10


def eval_on_lower_triangle_and_flip(s, q, f):
    ss, qs = np.meshgrid(s, q, sparse=False, indexing='ij')

    flip = ss + qs > 1
    p = np.empty_like(ss)
    p[~flip] = sum_terms(f, ss[~flip], qs[~flip])
    p[flip] = 1 - sum_terms(f, 1 - qs[flip], 1 - ss[flip])

    return p


def eval_on_lower_triangle(s, q, f):
    ss, qs = np.meshgrid(s, q, sparse=False, indexing='ij')

    flip = ss + qs > 1
    p = np.empty_like(ss)
    p[~flip] = sum_terms(f, ss[~flip], qs[~flip])
    p[flip] = sum_terms(f, 1 - qs[flip], 1 - ss[flip])

    return p


def sum_terms(f, s, q):
    n = np.arange(0, number_of_series_terms_to_evaluate) * 2 + 1
    ns = np.reshape(n, (-1, 1))
    ss = np.reshape(s, (1, -1))
    qs = np.reshape(q, (1, -1))

    return np.sum(f(ns, ss, qs), axis=0)


def u(n, s, q):
    """
    nth term of the series

    :param n: should be odd, this function is zero for n even
    :param s:
    :param q:
    :return:
    """
    first = sin(n * pi * s) * sinh(n * pi * q)
    second = sinh(n * pi * s) * sin(n * pi * q)

    factor = 4 / (n * pi * sinh(n * pi))

    return factor * (first + second)


def duds(n, s, q):
    """
    nth term of the series

    :param n: should be odd, this function is zero for n even
    :param s:
    :param q:
    :return:
    """
    first = cos(n * pi * s) * sinh(n * pi * q)
    second = cosh(n * pi * s) * sin(n * pi * q)

    factor = 4 / sinh(n * pi)

    return factor * (first + second)


def dudq(n, s, q):
    """
    nth term of the series

    :param n: should be odd, this function is zero for n even
    :param s:
    :param q:
    :return:
    """
    first = sin(n * pi * s) * cosh(n * pi * q)
    second = sinh(n * pi * s) * cos(n * pi * q)

    factor = 4 / sinh(n * pi)

    return factor * (first + second)


p = partial(eval_on_lower_triangle_and_flip, f=u)
dpds = partial(eval_on_lower_triangle, f=duds)
dpdq = partial(eval_on_lower_triangle, f=dudq)
