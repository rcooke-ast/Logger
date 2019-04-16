import timeit
from scipy.special import wofz
import numpy as np
import pdb
from matplotlib import pyplot as plt
import voigt_spline as vs


def voigt_wofz(u, a):
    return wofz(u + 1j * a).real


def benchmark(test='wofz', nmbr=10000):
    # Perform the benchmark tests
    if test == 'wofz':
        setup = '''from __main__ import voigt_wofz; import numpy as np; a, alluvoigt = 0.0005, np.linspace(0., 5., 1000)'''
        tst_code = '''voigt_wofz(alluvoigt, a)'''
        times = timeit.repeat(setup=setup, stmt=tst_code, repeat=3, number=nmbr)
    elif test == 'spline':
        setup = '''from __main__ import voigt_wofz; import voigt_spline as vs; import numpy as np; a, alluvoigt = 0.0005, np.linspace(0., 5., 1000); voigt_spline = vs.generate_spline()'''
        tst_code = '''voigt_spline(alluvoigt, a)'''
        times = timeit.repeat(setup=setup, stmt=tst_code, repeat=3, number=nmbr)
    print(test, 1.0E6 * min(times) / nmbr, "us")


def accuracy(test='wofz', plotdiff=False):
    a = 0.000005
    alluvoigt = np.linspace(0., 5., 1000)
    # First get the true values
    voigt_tru = np.zeros(alluvoigt.size)
    voigt_tst = np.zeros(alluvoigt.size)
    for uu, uvoigt in enumerate(alluvoigt):
        voigt_tru[uu] = vs.voigt_true(uvoigt, a)
    if test == 'wofz':
        voigt_tst = voigt_wofz(alluvoigt, a)
    elif test == 'spline':
        voigt_spline = vs.generate_spline()
        voigt_tst = voigt_spline(alluvoigt, a)
    # Calculate the difference
    diff = voigt_tst-voigt_tru
    rdiff = diff/voigt_tru
    print(test, np.max(np.abs(diff)), np.max(np.abs(rdiff)), np.std(diff), np.std(rdiff))
    # Plot the deviations from the true value
    if plotdiff:
        plt.subplot(211)
        plt.plot(alluvoigt, diff, 'k-')
        plt.subplot(212)
        plt.plot(alluvoigt, rdiff, 'r-')
        plt.show()


if __name__ == "__main__":
    # Some timings
    benchmark(test='wofz')
    benchmark(test='spline')
    # Test for accuracy
    accuracy(test='wofz')
    accuracy(test='spline')
