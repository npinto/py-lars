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

import scikits.learn.linear_model.least_angle as lars


def infer_basis(px,
                num_codebooks,
                sparsity=0.1,
                batch=100,
                learning_iterations=100, learning_threshold=0.01,
                resample_every=None, resample_percentile=None):
    '''Given a distribution of data, compute an efficient basis.

    Arguments
    ---------

    px: A callable that takes no arguments and returns a 1-dimensional array of
    observed data. The dimensionality of this array must be constant. The
    training process will halt whenever px() returns None.

    num_codebooks: Use this many elements (rows) in the basis.

    sparsity: We encode each input signal returned by p(x) using least angle
    regression (LARS). This parameter, a value in [0, 1], gives the fraction of
    codebook vectors to use during encoding, and thus determines the number of
    LARS iterations to perform.

    batch: A positive integer indicating the "mini-batch" size -- in practice,
    we sample more than one value from p(x) between our attempts to maximize the
    efficiency of the dictionary.

    learning_iterations: Maximize the efficiency of the dictionary for at most
    this many iterations at each step in the algorithm.

    learning_threshold: Stop maximizing the efficiency of the dictionary when
    the norm of the improvement falls below this scalar value.

    resample_every: If this is a positive integer, resample some of the codebook
    elements every this-many steps through the algorithm. This helps remove
    codebook vectors that are used very infrequently during encoding.

    resample_percentile: If resample_every is not None and this is some value in
    (0, 1), then resample this proportion of the codebook elements. Codebook
    elements are replaced with samples drawn from p(x) in increasing order of
    the absolute sum of the coefficients that have been used with that
    dictionary element.

    Result
    ------

    This function generates a sequence of basis dictionaries, each more
    efficient than the last for encoding elements drawn from p(x). Each
    dictionary has num_codebooks rows and the same number of columns as the
    length of elements returned by p(x).

    Attribution
    -----------

    This algorithm is implemented from Mairal, Bach, Ponce, and Sapiro (2009),
    "Online Dictionary Learning for Sparse Coding."
    '''
    # initialize the codebook with elements from our sample space.
    D = numpy.array([px() for _ in xrange(num_codebooks)])

    # normalize codebook vectors.
    for x in D:
        x /= numpy.linalg.norm(x)

    # set up storage matrices for the learning algorithm.
    A = numpy.zeros((num_codebooks, num_codebooks), float)
    B = numpy.zeros_like(D)
    D_ = numpy.zeros_like(D)

    # keep track of codebook usage.
    usage = numpy.zeros((num_codebooks, ), float)

    # limiting the number of codebook elements that are used for the encoding.
    max_features = int(num_codebooks * sparsity)

    # if resampling is enabled, calculate the number of elements to resample.
    num_resample = 0
    if 0 < resample_percentile < 1:
        num_resample = int(num_codebooks * resample_percentile)

    t = 0
    while True:
        t += 1

        # resample the lowest-use elements of D with samples from our space.
        if resample_every and not t % resample_every:
            for i in usage.argsort()[:num_resample]:
                D[i] = px()
            usage[:] = 0

        # eq 11 -- mini-batch extension
        theta = t * batch
        if t >= batch:
            theta = batch ** 2 + t - batch
        beta = (theta + 1. - batch) / (theta + 1.)
        A *= beta
        B *= beta

        for _ in range(batch):
            x = px()
            if x is None:
                return

            try:
                _, _, coeffs = lars.lars_path(D.T, x, max_features=max_features)
            except:
                continue

            alpha = coeffs[:, -1].reshape((len(coeffs[:, -1]), 1))
            A += numpy.dot(alpha, alpha.T).T
            B += numpy.dot(x.reshape((len(x), 1)), alpha.T).T
            usage += abs(alpha).flatten()

        # algorithm 2 -- repeatedly increment basis vectors.
        for _ in xrange(learning_iterations):
            D_[:] = D
            for j, (d, a, b) in enumerate(zip(D, A, B)):
                # eq 10
                u = d + (b - numpy.dot(D.T, a)) / (a[j] or 1)
                d[:] = u / max(numpy.linalg.norm(u), 1)

            if numpy.linalg.norm(D - D_) < learning_threshold:
                break

        yield D
