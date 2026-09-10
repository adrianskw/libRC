# -*- coding: utf-8 -*-
"""Explicit time-stepping integrators used by diffRC."""


def RK2(r, y, f):
    k1 = f(r, y)
    k2 = f(r + k1 / 2, y)
    return r + k2


def RK4(r, y, f):
    k1 = f(r, y)
    k2 = f(r + k1 / 2, y)
    k3 = f(r + k2 / 2, y)
    k4 = f(r + k3, y)
    return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
