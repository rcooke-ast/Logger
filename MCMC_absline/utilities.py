from scipy.special import wofz
import numpy as np


def get_binsize(wave, bintype="km/s", maxonly=False):
    binsize  = np.zeros((2, wave.size))
    binsizet = wave[1:] - wave[:-1]
    if bintype == "km/s":
        binsizet *= 2.99792458E5/wave[:-1]
    elif bintype == "A":
        pass
    elif bintype == "Hz":
        pass
    maxbin = np.max(binsizet)
    binsize[0, :-1], binsize[1, 1:] = binsizet, binsizet
    binsize[0, -1], binsize[1, 0] = maxbin, maxbin
    binsize = binsize.min(0)
    if maxonly:
        return np.max(binsize)
    else:
        return binsize


def get_subpixels(wave, nsubpix=10):
    binsize = get_binsize(wave)
    binlen = 1.0 / np.float64(nsubpix)
    interpwav = (1.0 + ((np.arange(nsubpix) - (0.5 * (nsubpix - 1.0)))[np.newaxis, :] * binlen * binsize[:, np.newaxis] / 2.99792458E5))
    subwave = (wave.reshape(wave.size, 1) * interpwav).flatten(0)
    return subwave


def rebin_subpix(flux, nsubpix=10):
    model = flux.reshape(flux.size//nsubpix, nsubpix).sum(axis=1) / np.float64(nsubpix)
    return model


def generate_wave(wavemin=3200.0, wavemax=5000.0, velstep=2.5, nsubpix=10):
    npix = np.log10(wavemax/wavemin) / np.log10(1.0+velstep/299792.458)
    npix = np.int(npix)
    wave = wavemin*(1.0+velstep/299792.458)**np.arange(npix)
    # Now generate a subpixellated wavelength grid
    subwave = get_subpixels(wave, nsubpix=nsubpix)
    return wave, subwave


def voigt_tau(wave, p0, p1, p2, lam, fvl, gam):
    cold = 10.0**p0
    zp1 = p1+1.0
    wv = lam*1.0e-8
    bl = p2*wv/2.99792458E5
    a = gam*wv*wv/(3.76730313461770655E11*bl)
    cns = wv*wv*fvl/(bl*2.002134602291006E12)
    cne = cold*cns
    ww = (wave*1.0e-8)/zp1
    v = wv*ww*((1.0/ww)-(1.0/wv))/bl
    tau = cne*wofz(v + 1j * a).real
    return tau


def voigt(wave, p0, p1, p2, lam, fvl, gam):
    tau = voigt_tau(wave, p0, p1, p2, lam, fvl, gam)
    return np.exp(-1.0*tau)


def convolve_spec(wave, flux, vfwhm):
    """
    Define the functional form of the model
    --------------------------------------------------------
    wave  : array of wavelengths
    flux  : model flux array
    vfwhm  : array of parameters for this model
    --------------------------------------------------------
    """
    sigd = vfwhm / (2.99792458E5 * (2.0*np.sqrt(2.0*np.log(2.0))))
    ysize = flux.size
    fsigd = 6.0*sigd
    dwav = 0.5 * (wave[2:] - wave[:-2]) / wave[1:-1]
    dwav = np.append(np.append(dwav[0], dwav), dwav[-1])
    df = int(np.min([np.int(np.ceil(fsigd/dwav).max()), ysize/2 - 1]))
    yval = np.zeros(2*df+1)
    yval[df:2*df+1] = (wave[df:2 * df + 1] / wave[df] - 1.0) / sigd
    yval[:df] = (wave[:df] / wave[df] - 1.0) / sigd
    gaus = np.exp(-0.5*yval*yval)
    size = ysize + gaus.size - 1
    fsize = 2 ** np.int(np.ceil(np.log2(size)))  # Use this size for a more efficient computation
    conv = np.fft.fft(flux, fsize)
    conv *= np.fft.fft(gaus/gaus.sum(), fsize)
    ret = np.fft.ifft(conv).real.copy()
    del conv
    return ret[df:df+ysize]


