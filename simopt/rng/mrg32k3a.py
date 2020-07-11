#!/usr/bin/env python
"""
Summary
-------
Provide a subclass of random.Random using mrg32k3a as the generator
with stream/substream/subsubstream support.

Listing
-------
MRG323k3a
get_next_prnstream
jump_substream
"""

# Code largely adopted from PyMOSO repository (https://github.com/pymoso/PyMOSO).

import random
from math import log
import functools

## constants used in mrg32k3a and in substream generation
## all from:
 # P. L'Ecuyer, ``Good Parameter Sets for Combined Multiple Recursive Random Number Generators'',
 # Operations Research, 47, 1 (1999), 159--164.
 #
 # P. L'Ecuyer, R. Simard, E. J. Chen, and W. D. Kelton,
 # ``An Objected-Oriented Random-Number Package with Many Long Streams and Substreams'',
 # Operations Research, 50, 6 (2002), 1073--1075

mrgnorm = 2.328306549295727688e-10
mrgm1 = 4294967087
mrgm2 = 4294944443
mrga12 = 1403580
mrga13n = 810728
mrga21 = 527612
mrga23n = 1370589

A1p0 =  [[0, 1, 0],
    [0, 0, 1],
    [-mrga13n, mrga12, 0]
]

A2p0 =  [[0, 1, 0],
    [0, 0, 1],
    [-mrga23n, 0, mrga21]
]

# A1p47 = matrix_power_mod(A1p0, 2**47, mrgm1)
A1p47 = [[1362557480, 3230022138, 4278720212],
    [3427386258, 3848976950, 3230022138],
    [2109817045, 2441486578, 3848976950]
]

# A2p47 = matrix_power_mod(A2p0, 2**47, mrgm2)
A2p47 = [[2920112852, 1965329198, 1177141043],
    [2135250851, 2920112852, 969184056],
    [296035385, 2135250851, 4267827987]
]

# A1p94 = matrix_power_mod(A1p0, 2**94, mrgm1)
A1p94 = [[2873769531, 2081104178, 596284397],
    [4153800443, 1261269623, 2081104178],
    [3967600061, 1830023157, 1261269623]
]

# A2p94 = matrix_power_mod(A2p0, 2**94, mrgm2)
A2p94 = [[1347291439, 2050427676, 736113023],
    [4102191254, 1347291439, 878627148],
    [1293500383, 4102191254, 745646810]
]

# A1p141 = matrix_power_mod(A1p0, 2**141, mrgm1)
A1p141 = [[3230096243, 2131723358, 3262178024],
    [2882890127, 4088518247, 2131723358],
    [3991553306, 1282224087, 4088518247]
]

# A2p141 = matrix_power_mod(A2p0, 2**141, mrgm2)
A2p141 = [[2196438580, 805386227, 4266375092],
    [4124675351, 2196438580, 2527961345],
    [94452540, 4124675351, 2825656399]
]

# #constants used for approximating the inverse standard normal cdf
# ## Beasly-Springer-Moro
# bsma = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
# bsmb = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
# bsmc = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863, 0.0038405729373609,0.0003951896411919, 0.0000321767881768, 0.0000002888167364, 0.0000003960315187]

# this is adapted to pure Python from the P. L'Ecuyer code referenced above
def mrg32k3a(seed):
    """
    Generate a random number between 0 and 1 from a seed.

    Parameters
    ----------
    seed : tuple of int
        Length must be 6.

    Returns
    -------
    newseed : tuple of int
    u : float
    """
    # Component 1
    p1 = mrga12*seed[1] - mrga13n*seed[0]
    k1 = int(p1/mrgm1)
    p1 -= k1*mrgm1
    if p1 < 0.0:
        p1 += mrgm1

    # Component 2
    p2 = mrga21*seed[5] - mrga23n*seed[3]
    k2 = int(p2/mrgm2)
    p2 -= k2*mrgm2
    if p2 < 0.0:
        p2 += mrgm2

    # Combination
    if p1 <= p2:
        u = (p1 - p2 + mrgm1)*mrgnorm
    else:
        u = (p1 - p2)*mrgnorm
    newseed = (seed[1], seed[2], int(p1), seed[4], seed[5], int(p2))
    return newseed, u


# # as in beasly-springer-moro
# def bsm(u):
#     """
#     Approximate the quantiles of the standard normal distribution.

#     Parameters
#     ----------
#     u : float
#         Desired quantile between 0 and 1

#     Returns
#     -------
#     z : float
#     """
#     y = u - 0.5
#     if abs(y) < 0.42:
#         ## approximate from the center (Beasly Springer 1973)
#         r = pow(y, 2)
#         r2 = pow(r, 2)
#         r3 = pow(r, 3)
#         r4 = pow(r, 4)
#         asum = sum([bsma[0], bsma[1]*r, bsma[2]*r2, bsma[3]*r3])
#         bsum = sum([1, bsmb[0]*r, bsmb[1]*r2, bsmb[2]*r3, bsmb[3]*r4])
#         z = y*(asum/bsum)
#     else:
#         ## approximate from the tails (Moro 1995)
#         if y < 0.0:
#             signum = -1
#             r = u
#         else:
#             signum = 1
#             r = 1 - u
#         s = log(-log(r))
#         s0 = pow(s, 2)
#         s1 = pow(s, 3)
#         s2 = pow(s, 4)
#         s3 = pow(s, 5)
#         s4 = pow(s, 6)
#         s5 = pow(s, 7)
#         s6 = pow(s, 8)
#         clst = [bsmc[0], bsmc[1]*s, bsmc[2]*s0, bsmc[3]*s1, bsmc[4]*s2, bsmc[5]*s3, bsmc[6]*s4, bsmc[7]*s5, bsmc[8]*s6]
#         t = sum(clst)
#         z = signum*t
#     return z


class MRG32k3a(random.Random):
    """
    Implements mrg32k3a as the generator for a random.Random object

    Attributes
    ----------
    _current_seed : tuple of int
        6 integer mrg32k3a seed

    Parameters
    ----------
    x : tuple of int, optional
        Seed from which to start the generator

    See also
    --------
    random.Random
    """

    def __init__(self, x=None):
        if not x:
            x = (12345, 12345, 12345, 12345, 12345, 12345)
        assert(len(x) == 6)
        self.version = 2
        self.generate = mrg32k3a
        self.ref_seed = x
        start_fixed_s_ss_sss(self, 0, 0, 0)
        # self.s_ss_sss_index = [0, 0, 0]
        # self.stream_start = x
        # self.substream_start = x
        # self.subsubstream_start = x
        # self.bsm = bsm
        super().__init__(x)

    # def __init__(self, x=None):
    #     if not x:
    #         x = (12345, 12345, 12345, 12345, 12345, 12345)
    #     assert(len(x) == 6)
    #     self.version = 2
    #     self.generate = mrg32k3a
    #     # self.bsm = bsm
    #     super().__init__(x)

    # def set_class_cache(self, cache_flag):
    #     """
    #     Sets whether to use an LRU cache for both the random function and the
    #     bsm function.

    #     Parameters
    #     ----------
    #     cache_flag : bool

    #     See also
    #     --------
    #     functools.lru_cache
    #     """
    #     if not cache_flag:
    #         self.generate = mrg32k3a
    #         self.bsm = bsm
    #     else:
    #         self.generate = functools.lru_cache(maxsize=None)(mrg32k3a)
    #         self.bsm = functools.lru_cache(maxsize=None)(bsm)

    def seed(self, a):
        """
        Set the seed of mrg32k3a and update the generator state.

        Parameters
        ----------
        a : tuple of int
        """
        assert(len(a) == 6)
        self._current_state = a
        super().seed(a)

    def random(self):
        """
        Generate a standard uniform variate and advance the generator
        state.

        Returns
        -------
        u : float
        """
        seed = self._current_state
        newseed, u = self.generate(seed)
        self.seed(newseed)
        return u

    def get_current_state(self):
        """
        Return the current mrg32k3a seed.

        Returns
        -------
        tuple of int
            The current mrg32k3a seed
        """
        return self._current_state

    def getstate(self):
        """
        Return the state of the generator.

        Returns
        -------
        tuple of int
            The current seed
        tuple
            Random.getstate output

        See also
        --------
        random.Random
        """
        return self.get_current_state(), super().getstate()

    def setstate(self, state):
        """
        Set the internal state of the generator.

        Parameters
        ----------
        state : tuple
            tuple[0] is mrg32k3a seed, [1] is random.Random.getstate

        See also
        --------
        random.Random
        """
        self.seed(state[0])
        super().setstate(state[1])

    # def normalvariate(self, mu=0, sigma=1):
    #     """
    #     Generate a normal random variate.

    #     Parameters
    #     ----------
    #     mu : float
    #         Expected value of the normal distribution from which to
    #         generate. Default is 0.
    #     sigma : float
    #         Standard deviatoin of the normal distribution from which to
    #         generate. Default is 1.

    #     Returns
    #     -------
    #     float
    #         A normal variate from the specified distribution

    #     """
    #     u = self.random()
    #     z = self.bsm(u)
    #     return sigma*z + mu

# def get_next_prnstream(seed, use_cache):
#     """
#     Instantiate a generator seeded 2^127 steps from the input seed.

#     Parameters
#     ----------
#     seed : tuple of int
#     crn : bool

#     Returns
#     -------
#     prn : MRG32k3a object
#     """
#     assert(len(seed) == 6)
#     # split the seed into 2 components of length 3
#     s1 = seed[0:3]
#     s2 = seed[3:6]
#     # A*s % m for both seed parts
#     ns1m = mat333mult(a1p127, s1)
#     ns2m = mat333mult(a2p127, s2)
#     ns1 = mat311mod(ns1m, mrgm1)
#     ns2 = mat311mod(ns2m, mrgm2)
#     # random.Random objects need a hashable seed e.g. a tuple
#     sseed = tuple(ns1 + ns2)
#     prn = MRG32k3a(sseed)
#     # prn.set_class_cache(use_cache)
#     return prn

# def jump_substream(prn):
#     """
#     Advance the rng to the next substream 2^76 steps.

#     Parameters
#     ----------
#     prn : MRG32k3a object
#     """
#     seed = prn.get_current_state()
#     # split the seed into 2 components of length 3
#     s1 = seed[0:3]
#     s2 = seed[3:6]
#     # A*s % m for both seed parts
#     ns1m = mat333mult(a1p76, s1)
#     ns2m = mat333mult(a2p76, s2)
#     ns1 = mat311mod(ns1m, mrgm1)
#     ns2 = mat311mod(ns2m, mrgm2)
#     # random.Random objects need a hashable seed e.g. a tuple
#     sseed = tuple(ns1 + ns2)
#     prn.seed(sseed)

def advance_stream(rng):
    """
    Advance the rng to the start of the next stream.
    Streams are of length 2^141.

    Parameters
    ----------
    rng : MRG32k3a object
    """
    state = rng.stream_start
    # split the state into 2 components of length 3
    st1 = state[0:3]
    st2 = state[3:6]
    # A*s % m for both state parts
    nst1m = mat333mult(A1p141, st1)
    nst2m = mat333mult(A2p141, st2)
    nst1 = mat311mod(nst1m, mrgm1)
    nst2 = mat311mod(nst2m, mrgm2)
    # random.Random objects need a hashable seed e.g. a tuple
    nstate = tuple(nst1 + nst2)
    rng.seed(nstate)
    # increment the stream index
    rng.s_ss_sss_index[0] += 1
    # update state referencing
    rng.stream_start = nstate
    rng.substream_start = nstate
    rng.subsubstream_start = nstate

def advance_substream(rng):
    """
    Advance the rng to the start of the next substream.
    Substreams are of length 2^94.

    Parameters
    ----------
    rng : MRG32k3a object
    """
    state = rng.substream_start
    # split the state into 2 components of length 3
    st1 = state[0:3]
    st2 = state[3:6]
    # A*s % m for both state parts
    nst1m = mat333mult(A1p94, st1)
    nst2m = mat333mult(A2p94, st2)
    nst1 = mat311mod(nst1m, mrgm1)
    nst2 = mat311mod(nst2m, mrgm2)
    # random.Random objects need a hashable seed e.g. a tuple
    nstate = tuple(nst1 + nst2)
    rng.seed(nstate)
    # increment the substream index
    rng.s_ss_sss_index[1] += 1
    # update state referencing
    rng.substream_start = nstate
    rng.subsubstream_start = nstate

def advance_subsubstream(rng):
    """
    Advance the rng to the start of the next subsubstream.
    Subsubstreams are of length 2^47.

    Parameters
    ----------
    rng : MRG32k3a object
    """
    state = rng.subsubstream_start
    # split the state into 2 components of length 3
    st1 = state[0:3]
    st2 = state[3:6]
    # A*s % m for both state parts
    nst1m = mat333mult(A1p47, st1)
    nst2m = mat333mult(A2p47, st2)
    nst1 = mat311mod(nst1m, mrgm1)
    nst2 = mat311mod(nst2m, mrgm2)
    # random.Random objects need a hashable seed e.g. a tuple
    nstate = tuple(nst1 + nst2)
    rng.seed(nstate)
    # increment the subsubstream index
    rng.s_ss_sss_index[3] += 1
    # update state referencing
    rng.subsubstream_start = nstate

def reset_stream(rng):
    """
    Reset the rng to the start of the current stream.

    Parameters
    ----------
    rng : MRG32k3a object
    """
    nstate = rng.stream_start
    rng.seed(nstate)
    # update state referencing
    rng.substream_start = nstate
    rng.subsubstream_start = nstate
    # reset index for substream and subsubstream
    rng.s_ss_sss_index[1] = 0
    rng.s_ss_sss_index[2] = 0

def reset_substream(rng):
    """
    Reset the rng to the start of the current substream.

    Parameters
    ----------
    rng : MRG32k3a object
    """
    nstate = rng.substream_start
    rng.seed(nstate)
    # update state referencing
    rng.subsubstream_start = nstate
    # reset index for substream and subsubstream
    rng.s_ss_sss_index[2] = 0

def reset_subusbstream(rng):
    """
    Reset the rng to the start of the current subsubstream.

    Parameters
    ----------
    rng : MRG32k3a object
    """
    nstate = rng.subsubstream_start
    rng.seed(nstate)

def start_fixed_s_ss_sss(rng, stream, substream, subsubstream):
    """
    Set the rng to the start of a specified (stream, substream, subsubstream) triplet.

    Parameters
    ----------
    rng : MRG32k3a object
    """
    state = rng.ref_seed
    # split the reference seed into 2 components of length 3
    st1 = state[0:3]
    st2 = state[3:6]
    # advance to start of specified stream
    for _ in range(stream):
        # A*s % m for both state parts
        nst1m = mat333mult(A1p141, st1)
        nst2m = mat333mult(A2p141, st2)
        st1 = mat311mod(nst1m, mrgm1)
        st2 = mat311mod(nst2m, mrgm2)
    rng.stream_start = tuple(st1 + st2)
    # advance to start of specified substream
    for _ in range(substream):
        # A*s % m for both state parts
        nst1m = mat333mult(A1p94, st1)
        nst2m = mat333mult(A2p94, st2)
        st1 = mat311mod(nst1m, mrgm1)
        st2 = mat311mod(nst2m, mrgm2)
    rng.substream_start = tuple(st1 + st2)
    # advance to start of specified subsubstream
    for _ in range(subsubstream):
        # A*s % m for both state parts
        nst1m = mat333mult(A1p94, st1)
        nst2m = mat333mult(A2p94, st2)
        st1 = mat311mod(nst1m, mrgm1)
        st2 = mat311mod(nst2m, mrgm2)
    rng.subsubstream_start = tuple(st1 + st2)
    # random.Random objects need a hashable seed e.g. a tuple
    nstate = tuple(st1 + st2)
    rng.seed(nstate)
    # update index referencing
    rng.s_ss_sss_index = [stream, substream, subsubstream]

def mat333mult(A, b):
    """
    Multiply a 3x3 matrix with a 3x1 matrix.

    Parameters
    ----------
    A : tuple of tuple of float
        3x3 matrix
    b : tuple of tuple of float
        3x1 matrix

    Returns
    -------
    res : list of float
        3x1 matrix
    """
    res = [0, 0, 0]
    r3 = range(3)
    for i in r3:
        res[i] = sum([A[i][j]*b[j] for j in r3])
    return res


def mat311mod(b, c):
    """
    Compute moduli of a 3x1 matrix.

    Parameters
    ----------
    b : tuple of float
        3x1 matrix
    c : float
        modulus

    Returns
    -------
    res : tuple of float
        3x1 matrix
    """
    res = [0, 0, 0]
    r3 = range(3)
    for i in r3:
        res[i] = int(b[i] - int(b[i]/c)*c)
        if res[i] < 0:
            res[i] += c
    return res


def matrix_power_mod(A, j, m):
    """
    Compute moduli of a matrix power.
    Use divide-and-conquer algorithm described in L'Ecuyer (1990).

    Parameters
    ----------
    A : tuple of tuple of float
        3x3 matrix
    j : int
        exponent
    m : float
        modulus

    Returns
    -------
    res : tuple of tuple of float
        3x3 matrix
    """
    W = A
    B = [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    while j > 0:
        if (j % 2 == 1):
            B = matrix_matrix_mod(W, B, m)
        W = matrix_matrix_mod(W, W, m)
        j = int(j/2)
    res = B
    return res

def matrix_matrix_mod(A, B, m):
    """
    Computes moduli of matrix product.

    Parameters
    ----------
    A : tuple of tuple of float
        3x3 matrix
    B : tuple of tuple of float
        3x3 matrix
    m : float
        modulus

    Returns
    -------
    res : tuple of tuple of float
        3x3 matrix
    """
    C = mat33prod(A, B)
    res = mat33mod(C, m)
    return res

def mat33prod(A, B):
    """
    Compute product of two 3x3 matrices.

    Parameters
    ----------
    A : tuple of float
        3x3 matrix
    B : tuple of tuple of float
        3x3 matrix

    Returns
    -------
    res : tuple of float
        3x3 matrix
    """
    res = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    r3 = range(3)
    for i in r3:
        for j in r3:
            for k in r3:
                res[i][j] += A[i][k]*B[k][j]
    return res

def mat33mod(A, m):
    """
    Compute moduli of a 3x3 matrix.

    Parameters
    ----------
    A : tuple of float
        3x3 matrix
    m : float
        modulus

    Returns
    -------
    res : tuple of float
        3x3 matrix
    """
    res = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    r3 = range(3)
    for i in r3:
        for j in r3:
            res[i][j] = int(A[i][j] - int(A[i][j]/m)*m)
            if res[i][j] < 0:
                res[i][j] += m
    return res
    