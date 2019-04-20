import pdb
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt
from pyigm.fN.fnmodel import FNModel
from pyigm.fN.mockforest import mk_mock


def generate_wave(wavemin=3200.0, wavemax=5000.0, velstep=2.5):
    npix = np.log10(wavemax/wavemin) / np.log10(1.0+velstep/299792.458)
    npix = np.int(npix)
    wave = wavemin*(1.0+velstep/299792.458)**np.arange(npix)
    return wave


def generate_fakespectra(zqso, wave=None, snr=30.0, plot_spec=False, seed=1234):
    # Get a random state so that the noise and components can be reproduced
    rstate = np.random.RandomState(seed)
    # Define the wavelength coverage
    if wave is None:
        wave = generate_wave(wavemax=1240.0*(1.0+zqso)) * u.AA
    # Get the CDDF
    NHI = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    sply = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    params = dict(sply=sply)
    fN_model = FNModel('Hspline', pivots=NHI, param=params, zmnx=(2., 5.))
    # Generate a fake spectrum
    print("Note :: I made some changes in pyigm.fn.mockforest.py to avoid convolution and noise")
    mock_spec, HI_comps, _ = mk_mock(wave, zqso, fN_model, fwhm=0.0, s2n=0.0)
    # Convolve the spectrum

    # Add reproducible noise
    mock_spec.add_noise(s2n=snr, rstate=rstate)
    # Plot the spectrum, if requested
    if plot_spec:
        plt.plot(mock_spec.wave, mock_spec.flux, 'k-')
        plt.show()
    return mock_spec, HI_comps


zem = 3.0
mock_spec, HI_comps = generate_fakespectra(zem)
print(HI_comps['lgNHI'])
print(HI_comps['z'])
print(HI_comps['bval'])
