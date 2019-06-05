import pdb
import sys
import time
import utilities as util
import numpy as np
from matplotlib import pyplot as plt
import emcee
import corner

# Generate some fake data
cold, zabs, bval = 14.0, 0.0, 15.0   # Abs line parameters
lam, fvl, gam = 1215.6701, 0.416400, 6.265E8
nsubpix = 10  # Number of subpixels to evaluate the spectrum
vsize = 2.5  # Pixel size in km/s
vfwhm = 7.0  # velocity FWHM instrument resolution
snr = 20.0   # S/N of data
wmin, wmax = 1212.0, 1218.0

walk_cold, walk_bval = 12.0, vfwhm/2.0

# Generate some fake data
wave, wavesub = util.generate_wave(wavemin=1212.0, wavemax=1218.0, velstep=vsize, nsubpix=nsubpix)

fluxsub = util.voigt(wavesub, cold, zabs, bval, lam, fvl, gam)
# Convolution
convsub = util.convolve_spec(wavesub, fluxsub, vfwhm)
# Rebin
fluxerr = np.ones(wave.size)/snr
flux = util.rebin_subpix(convsub, nsubpix=nsubpix)
flux_nse = np.random.normal(flux, fluxerr)

# Convert to velocity
velo = 299792.458*(wave - lam)/lam
velosub = 299792.458*(wavesub - lam)/lam


def get_model(wav, theta):
    tau = np.zeros(wav.size)
    for th in theta:
        tau += util.voigt_tau(wav, walk_cold, th, walk_bval, lam, fvl, gam)
    modflux = np.exp(-tau)
    convflux = util.convolve_spec(wav, modflux, vfwhm)
    return util.rebin_subpix(convflux, nsubpix=nsubpix)


def lnlike(theta, x, y, yerr):
    model = get_model(x, theta)
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2 * inv_sigma2))


def lnprior(theta):
    priors = -np.inf*np.ones(theta.size)
    priors[np.where((lam*(1+theta) > wmin) & (lam*(1+theta) < wmax))] = 0.0
    return priors


def lnprob(theta, x, y, yerr):
    thflat = theta.flatten()
    lp = lnprior(thflat)
    retval = lp + lnlike(thflat, x, y, yerr)
    # May need to set retval to -np.inf if lp is -np.inf
    return retval.reshape(theta.shape)


ndim, nwalkers, nsteps = 1, 1000, 500
pr_min, pr_max = wmin/lam - 1.0, wmax/lam-1.0
pos = np.array([[np.random.uniform(pr_min, pr_max) for i in range(nwalkers)]]).T

# Create the sampler, and set vectorize to be true to get all the samples
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wavesub, flux, fluxerr), vectorize=True)

width = 30
t0 = time.time()
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    n = int((width + 1) * float(i) / nsteps)
    sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
sys.stdout.write("\n")
t1 = time.time()
print("It took", t1 - t0, "seconds")

pdb.set_trace()

params = sampler.chain[:, -1, 0]
outmodel = get_model(wavesub, params)
strmodel = get_model(wavesub, pos.flatten())

plt.plot(velo, flux, 'k-', drawstyle='steps')
plt.plot(velo-vsize/2.0, outmodel, 'r-')
plt.plot(velo-vsize/2.0, strmodel, 'b--')
plt.show()
