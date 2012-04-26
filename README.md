# py-lars

A literal implementation of the LARS algorithm described by Efron, Hastie,
Johnstone, and Tibshirani (2004). This implementation is much less complex than
the one in `scikits.learn`, which I think might be useful for pedagogical
purposes. For doing "real" regression problems, though, I'd recommend using the
`scikits.learn` implementation.

## Installation

Install with `pip`

    pip install lmj.lars

Or directly from the source

    git clone http://github.com/lmjohns3/py-lars
    cd py-lars
    python setup.py install
