#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 00:12:41 2017

@author: ratnadeepb
"""

# Gaussian
# This is from my Hackerrank implementation

import math

# Standard Normal
def stdNorm(x):
    return math.exp((-x)**2/2)/(math.pi**0.5)

# Normal using stdNorm
def normal(x, mu, sigma):
    return (1/sigma) * stdNorm((x-mu)/sigma)

# Cumulative Gaussian with erf
def erf(x, mu, sigma):
    return (1.0 + math.erf( (x - mu) / (sigma * (2.0 ** 0.5)) ) ) / 2
