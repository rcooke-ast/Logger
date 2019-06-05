import utilities as util
from matplotlib import pyplot as plt
import numpy as np



cold, zabs, bval = 14.0, 0.0, 15.0   # Abs line parameters
lam, fvl, gam = 1215.6701, 0.416400, 6.265E8
nsubpix = 10  # Number of subpixels to evaluate the spectrum
vsize = 2.5  # Pixel size in km/s
vfwhm = 7.0  # velocity FWHM instrument resolution

# Generate some fake data
wave, wavesub = util.generate_wave(wavemin=1212.0, wavemax=1218.0, velstep=vsize, nsubpix=nsubpix)

fluxsub = util.voigt(wavesub, cold, zabs, bval, lam, fvl, gam)
# Convolution
convsub = util.convolve_spec(wavesub, fluxsub, vfwhm)
# Rebin
conv = util.rebin_subpix(convsub, nsubpix=nsubpix)

# Convert to velocity
velo = 299792.458*(wave - lam)/lam
velosub = 299792.458*(wavesub - lam)/lam
plt.plot(velo, conv, 'k-', drawstyle='steps')
plt.plot(velosub-vsize/2.0, convsub, 'r-')
plt.show()
