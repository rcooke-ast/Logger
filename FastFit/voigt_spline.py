# Generate a spline representation of the voigt function
from scipy import interpolate
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import wofz


def voigt_wofz(u, a):
    return wofz(u + 1j * a).real


def voigt_true(u, a):
    """ Calculate the voigt function to very high accuracy.
    Uses numerical integration, so is slow.  Answer is correct to 20
    significant figures.
    Note this needs `mpmath` or `sympy` to be installed.
    """
    try:
        import mpmath as mp
    except ImportError:
        from sympy import mpmath as mp
    with mp.workdps(20):
        z = mp.mpc(u, a)
        result = mp.exp(-z*z) * mp.erfc(-1j*z)
    return result.real


def generate_points():
    umin, umax, unum = -6.0, 1.0, 5000
    amin, amax, anum = 0.0, 0.01, 10
    uarr = 10.0 ** np.linspace(umin, umax, unum)
    uval = np.append(-uarr[::-1], np.append(0.0, uarr))
    aval = np.linspace(amin, amax, anum)
    return uval, aval


def generate_spline():
    uval, aval = generate_points()
    uu, aa = np.meshgrid(uval, aval)
    fval = np.zeros_like(uu)
    for ii in range(uu.shape[0]):
        for jj in range(uu.shape[1]):
            fval[ii, jj] = voigt_true(uu[ii, jj], aa[ii, jj])
            fval[ii, jj] = voigt_wofz(uu[ii, jj], aa[ii, jj])
    fspl = interpolate.interp2d(uval, aval, fval, kind='cubic', fill_value=0.0, bounds_error=False)
    return fspl


def accuracy(plotdiff=True):
    a = 0.0004
    alluvoigt = np.linspace(0., 10., 100)
    # First get the true values
    print("Generating True values")
    voigt_tru = np.zeros(alluvoigt.size)
    voigt_tst = np.zeros(alluvoigt.size)
    for uu, uvoigt in enumerate(alluvoigt):
        voigt_tru[uu] = voigt_true(uvoigt, a)
    print("Generating spline")
    voigt_spl = generate_spline()
    print("Evaluating spline")
    voigt_tst = voigt_spl(alluvoigt, a)
    print("Performing tests")
    # Calculate the difference
    diff = voigt_tst-voigt_tru
    rdiff = diff/voigt_tru
    print(np.max(np.abs(diff)), np.max(np.abs(rdiff)), np.std(diff), np.std(rdiff))
    # Plot the deviations from the true value
    if plotdiff:
        plt.subplot(211)
        plt.plot(alluvoigt, diff, 'k-')
        plt.subplot(212)
        plt.plot(alluvoigt, rdiff, 'r-')
        plt.show()


if __name__ == "__main__":
    # Some timings
    accuracy()
