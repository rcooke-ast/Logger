import pdb
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt
from pyigm.fN.fnmodel import FNModel
from pyigm.fN.mockforest import mk_mock
from pyigm.continuum import quasar as pycq
from linetools.spectra.xspectrum1d import XSpectrum1D
from scipy import interpolate
from astropy.io.votable import parse_single_table

# Set some constants
const = (2.99792458E5 * (2.0 * np.sqrt(2.0 * np.log(2.0))))


def load_atomic(return_HIwav=True):
    """
    Load the atomic transitions data
    """
#    dir = "/Users/rcooke/Software/ALIS/alis/data/"
    atmname = "atomic.xml"
    print("Loading atomic data")
    # If the user specifies the atomic data file, make sure that it exists
    try:
        dir = "/home/rcooke/Software/ALIS/alis/data/"
        table = parse_single_table(dir+atmname)
    except:
        dir = "/cosma/home/durham/rcooke/Software/ALIS/alis/data/"
        table = parse_single_table(dir+atmname)
    isotope = table.array['MassNumber'].astype("|S3").astype(np.object)+table.array['Element']
    atmdata = dict({})
    atmdata['Ion'] = np.array(isotope+b"_"+table.array['Ion']).astype(np.str)
    atmdata['Wavelength'] = np.array(table.array['RestWave'])
    atmdata['fvalue'] = np.array(table.array['fval'])
    atmdata['gamma'] = np.array(table.array['Gamma'])
    if return_HIwav:
        ww = np.where(atmdata["Ion"] == "1H_I")
        wavs = atmdata["Wavelength"][ww][3:]
        return wavs*u.AA
    else:
        return atmdata


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


def convolve(y, x, vfwhm):
    vsigd = vfwhm / const
    ysize = y.shape[0]
    fsigd = 6.0 * vsigd
    dwav = np.gradient(x) / x
    df = int(np.min([np.int(np.ceil(fsigd / dwav).max()), ysize // 2 - 1]))
    yval = np.zeros(2 * df + 1)
    yval[df:2 * df + 1] = (x[df:2 * df + 1] / x[df] - 1.0) / vsigd
    yval[:df] = (x[:df] / x[df] - 1.0) / vsigd
    gaus = np.exp(-0.5 * yval * yval)
    size = ysize + gaus.size - 1
    fsize = 2 ** np.int(np.ceil(np.log2(size)))  # Use this size for a more efficient computation
    conv = np.fft.fft(y, fsize, axis=0)
    if y.ndim == 1:
        conv *= np.fft.fft(gaus / gaus.sum(), fsize)
    else:
        conv *= np.fft.fft(gaus / gaus.sum(), fsize).reshape((fsize, 1))
    ret = np.fft.ifft(conv, axis=0).real.copy()
    del conv
    if y.ndim == 1:
        return ret[df:df + ysize]
    else:
        return ret[df:df + ysize, :]


def convolve1d(y, x, vfwhm):
    vsigd = vfwhm / const
    ysize = y.size
    fsigd = 6.0 * vsigd
    dwav = np.gradient(x) / x
    df = int(np.min([np.int(np.ceil(fsigd / dwav).max()), ysize // 2 - 1]))
    yval = np.zeros(2 * df + 1)
    yval[df:2 * df + 1] = (x[df:2 * df + 1] / x[df] - 1.0) / vsigd
    yval[:df] = (x[:df] / x[df] - 1.0) / vsigd
    gaus = np.exp(-0.5 * yval * yval)
    size = ysize + gaus.size - 1
    fsize = 2 ** np.int(np.ceil(np.log2(size)))  # Use this size for a more efficient computation
    conv = np.fft.fft(y, fsize)
    conv *= np.fft.fft(gaus / gaus.sum(), fsize)
    ret = np.fft.ifft(conv).real.copy()
    del conv
    return ret[df:df + ysize]


def generate_wave(wavemin=3200.0, wavemax=5000.0, velstep=2.5, nsubpix=10):
    npix = np.log10(wavemax/wavemin) / np.log10(1.0+velstep/299792.458)
    npix = np.int(npix)
    wave = wavemin*(1.0+velstep/299792.458)**np.arange(npix)
    # Now generate a subpixellated wavelength grid
    subwave = get_subpixels(wave, nsubpix=nsubpix)
    return wave, subwave


def rebin_subpix(flux, nsubpix=10):
    model = flux.reshape(flux.size//nsubpix, nsubpix).sum(axis=1) / np.float64(nsubpix)
    return model


def generate_fakespectra(zqso, wave=None, subwave=None, nsubpix=10, snr=30.0, plot_spec=False, seed=1234, vfwhm=7.0):
    add_noise = True
    if snr <= 0.0:
        add_noise = False
        snr = 1.0
    # Get a random state so that the noise and components can be reproduced
    rstate = np.random.RandomState(seed)
    # Define the wavelength coverage
    if wave is None or subwave is None:
        wave, subwave = generate_wave(wavemax=1240.0*(1.0+zqso), nsubpix=nsubpix)
        wave *= u.AA
        subwave *= u.AA
    # Get the CDDF
    NHI = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    sply = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    params = dict(sply=sply)
    fN_model = FNModel('Hspline', pivots=NHI, param=params, zmnx=(2., 5.))
    # Generate a fake spectrum
    print("Note :: I made some changes in pyigm.fn.mockforest.py to avoid convolution and noise")
    print("Note :: I made some changes in pyigm.fn.mockforest.py to perform my own subpixellation")
    _, HI_comps, mock_subspec = mk_mock(wave, zqso, fN_model, fwhm=0.0, s2n=0.0, subwave=subwave)
    # Generate a quasar continuum
    print("Note :: I made some changes in pyigm.continuum.quasar.py to return the raw WFC3 spectra")
    conti, wfc3_idx = pycq.wfc3_continuum(zqso=zqso, get_orig=True, rstate=rstate)
    convcont = convolve(conti.flux, conti.wavelength, 5000.0)  # Need to smooth out the noisy WFC3 spectra
    cspl = interpolate.interp1d(conti.wavelength, convcont, kind='cubic', bounds_error=False, fill_value="extrapolate")
    cflux = cspl(subwave)
    # Create the final subpixellated model
    model_flux_sub = mock_subspec[0].flux * cflux
    # Convolve the spectrum
    mock_conv = convolve(model_flux_sub, subwave, vfwhm)
    # Rebin the spectrum and store as XSpectrum1D
    final_spec = rebin_subpix(mock_conv, nsubpix=nsubpix)
    final_serr = rebin_subpix(cflux*np.ones(mock_conv.size)/snr, nsubpix=nsubpix)
    mock_spec = XSpectrum1D.from_tuple((wave, final_spec, final_serr))
    # Add reproducible noise
    noisy_spec = mock_spec.add_noise(rstate=rstate)
    # Plot the spectrum, if requested
    if plot_spec:
        plt.plot(subwave, mock_conv, 'b-')   # Subpixel spectrum
        plt.plot(wave, final_spec, 'r--')     # Rebinned spectrum
        plt.plot(mock_spec.wavelength, noisy_spec.flux, 'k-')   # Noisy spectrum
        plt.plot(subwave, cflux, 'g-')   # Continuum
        plt.plot(conti.wavelength, conti.flux, 'm-')
        plt.plot(conti.wavelength, convcont, 'c-')
        plt.show()
    if add_noise:
        return noisy_spec, HI_comps
    else:
        return mock_spec, HI_comps


if __name__ == "__main__":
    zem = 3.0
    mock_spec, HI_comps = generate_fakespectra(zem, plot_spec=True)
    pdb.set_trace()
    plt.plot(HI_comps['lgNHI'].data, HI_comps['bval'].data, 'bx')
    plt.show()
    #print(HI_comps['lgNHI'])
    #print(HI_comps['z'])
    #print(HI_comps['bval'])
