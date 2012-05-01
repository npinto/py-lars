# Copyright (c) 2012 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''A Python/numpy implementation of sparse dictionary learning using LARS.'''

import numpy


def lars(y, X, max_sum_coeffs=None, min_coeff=0, sigma=None):
    '''Given an observation vector y and a column basis X, compute LARS.

    This computes an encoding \beta that minimizes 1/2 ||y - X\beta||^2, subject
    to the constraint that |\beta_i| sum to less than some fixed limit. Beta is
    generated iteratively given the residual r by choosing the maximally
    responding column of X and moving r in that direction until the correlation
    between r and the maximal X equals the correlation between r and some "newly
    active" column in X.

    y is a vector of "response" variables that we are trying to encode. The mean
    of y must be 0.

    X is a matrix of covariates (one per column ; each column must be
    zero-centered and of unit length) to use for the regression.

    If max_sum_coeffs is a positive real, then the algorithm will halt once the
    sum-of-coefficients constraint above is violated.

    If min_coeff is a positive real, the algorithm will halt once the maximally
    responding column(s) in X have less than this response magnitude.

    If both max_sum_coeffs and min_coeff are None, then the algorithm will halt
    when the risk induced by the regression coefficients starts to grow. In this
    case, sigma should be an estimate of the standard deviation of the elements
    of y, if they had been drawn IID from N(0, sigma^2). If it is not given,
    y.std() will be used.

    This implementation is based on Efron, Hastie, Johnstone, and Tibshirani's
    2004 paper, "Least Angle Regression," especially the equations on pages 413
    and 414.
    '''
    assert numpy.allclose([x.mean() for x in X.T], 0)
    assert numpy.allclose([numpy.linalg.norm(x) for x in X.T], 1)
    assert numpy.allclose(y.mean(), 0)

    n = len(y)
    sigma = sigma or y.std()
    mu = numpy.zeros_like(y)
    beta = numpy.zeros((len(X.T), ), float)

    risk = None
    if max_sum_coeffs is None:
        max_sum_coeffs = numpy.inf
        risk = numpy.inf

    # helper functions.
    def near(a, b):
        return abs(a - b) < 1e-12 + 1e-10 * b

    def finite_positive(num, den):
        x = num / numpy.where(den != 0, den, 1e-100)
        return numpy.where(x <= 0, x.max(), x)

    L = len(X.T)
    for k in range(L):
        # eq 2.8
        c = numpy.dot(X.T, y - mu)

        # eq 2.9
        cabs = abs(c)
        C = cabs.max()
        active = near(cabs, C).squeeze()

        if C == 0 or C < min_coeff:
            break

        # quick shortcut -- if there are more active vectors than our current
        # step number, there's been some numerical instability, so just exit
        # early from the encoding loop.
        if active.sum() > k + 1 or numpy.alltrue(active):
            break

        # eq 2.10
        s = numpy.where(c[active] > 0, 1, -1)

        # eq 2.4
        X_active = s * X[:, active]

        # eq 2.5
        G = numpy.dot(X_active.T, X_active)
        Ginv = numpy.linalg.inv(G)
        Ginv1 = Ginv.sum(axis=1)  # G^-1 only ever appears multiplied by 1_A

        A = numpy.sqrt(Ginv1.sum())

        # eq 2.6
        w = A * Ginv1
        u = numpy.dot(X_active, w)

        # eq 2.11
        a = numpy.dot(X.T, u)

        # eq 2.13
        complement = numpy.invert(active)
        cc = c[complement]
        ac = a[complement]
        gamma = min(finite_positive(C - cc, A - ac).min(),
                    finite_positive(C + cc, A + ac).min())

        # eq 2.12
        mu += gamma * u

        if risk is not None:
            # eq 4.10 (page 424) -- risk-based stop criterion
            risk_ = (numpy.linalg.norm(y - mu) / sigma) ** 2 - n + 2 * k
            if risk < risk_:
                break
            risk = risk_

        beta[active] += s * gamma

        if sum(abs(beta)) > max_sum_coeffs:
            break

        yield mu, beta
