#!/usr/bin/env python

import numpy
import numpy.random as rng
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import lmj.lars

import scikits.learn.datasets
diabetes = scikits.learn.datasets.load_diabetes()


def sample_and_encode(X, scale=1.):
    #X = diabetes.data

    #truth = numpy.zeros_like(X[0])
    #y = diabetes.target
    truth = rng.laplace(0, scale, size=X.shape[1])
    y = numpy.dot(X, truth)

    yc = y - y.mean()

    print
    print ' T |', '%6.3f |' % abs(truth).sum(),
    print' '.join('%6.3f' % b for b in truth)

    guess = numpy.zeros_like(yc)
    for i, (guess, beta) in enumerate(lmj.lars.lars(yc, X)):
        print '%2d |' % i, '%6.3f |' % abs(beta).sum(),
        print ' '.join('%6.3f' % b for b in beta),
        print '|  yerr %.5f' % numpy.linalg.norm(yc - guess),
        print '  xerr %.5f' % numpy.linalg.norm(truth - beta)
    return yc, guess


def test(X, scale=1.):
    e = numpy.zeros(10.)
    for i in range(len(e)):
        truth, guess = sample_and_encode(X, scale)
        e[i] = numpy.linalg.norm(truth - guess)
    print
    print 'min %.3f avg %.3f max %.3f' % (e.min(), e.mean(), e.max())


def create(dimensions, vectors):
    X = rng.randn(dimensions, vectors)
    for i, x in enumerate(X.T):
        x -= x.mean()
        x /= numpy.linalg.norm(x)
        #print 'codebook vector', i, x
    return X


if __name__ == '__main__':
    test(create(8, 10), 0.1)
