# This script reads the spectra generated with
# cnn_create_training_set and generates some ID_labels
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
from scipy import interpolate
from utilities import load_atomic, convolve
from pyigm.continuum import quasar as pycq
from scipy.special import wofz, erf
import pdb

nHIwav = 4
atmdata = load_atomic(return_HIwav=False)
ww = np.where(atmdata["Ion"] == "1H_I")
HIwav = atmdata["Wavelength"][ww][3:3+nHIwav]
HIfvl = atmdata["fvalue"][ww][3:3+nHIwav]
HIgam = atmdata["gamma"][ww][3:3+nHIwav]


def voigt_tau(par, wavein, logn=True):
    """ Obtain the optical depth for the Lyman series absorption lines
    """
    # Column density
    if logn:
        cold = 10.0 ** par[0]
    else:
        cold = par[0]
    # Redshift
    zp1 = par[1] + 1.0
    tau = np.zeros((wavein.size, HIwav.size))
    for ii in range(HIwav.size):
        wv = HIwav[ii] * 1.0e-8  # Convert to cm
        # Doppler parameter
        bl = par[2] * wv / 2.99792458E5
        a = HIgam[ii] * wv * wv / (3.76730313461770655E11 * bl)
        cns = wv * wv * HIfvl[ii] / (bl * 2.002134602291006E12)
        cne = cold * cns
        ww = (wavein * 1.0e-8) / zp1
        v = wv * ww * ((1.0 / ww) - (1.0 / wv)) / bl
        tau[:, ii] = cne * wofz(v + 1j * a).real
    return tau


def load_dataset(zem=3.0, snr=0, numspec=20, normalise=False):
    zstr = "zem{0:.2f}".format(zem)
    sstr = "snr{0:d}".format(int(snr))
    extstr = "{0:s}_{1:s}_nspec{2:d}".format(zstr, sstr, numspec)
    fname = "train_data/cnn_qsospec_fluxspec_{0:s}.npy".format(extstr)
    wfdata_all = np.load(fname)
    wdata = wfdata_all[0, :]
    fdata_all = wfdata_all[1:, :]
    zdata_all = np.load("train_data/cnn_qsospec_zvals_{0:s}.npy".format(extstr))
    Ndata_all = np.load("train_data/cnn_qsospec_Nvals_{0:s}.npy".format(extstr))
    bdata_all = np.load("train_data/cnn_qsospec_bvals_{0:s}.npy".format(extstr))
    if normalise:
        for ispec in range(fdata_all.shape[0]):
            fdata_all[ispec, :] /= generate_continuum(ispec, wdata)
    return fname, fdata_all, wdata, zdata_all, Ndata_all, bdata_all


def generate_continuum(seed, wave, zqso=3.0):
    rstate = np.random.RandomState(seed)
    conti, wfc3_idx = pycq.wfc3_continuum(zqso=zqso, get_orig=True, rstate=rstate)
    convcont = convolve(conti.flux, conti.wavelength, 5000.0)  # Need to smooth out the noisy WFC3 spectra
    cspl = interpolate.interp1d(conti.wavelength, convcont, kind='cubic', bounds_error=False, fill_value="extrapolate")
    return cspl(wave)


def generate_labels(ispec, wdata, fdata, zdata_all, Ndata_all, bdata_all, zqso=3.0, snr=0, snr_thresh=1.0, Nstore=2):
    """
    Nstore : maximum number of absorption profiles that contribute to a given pixel
    """
    if snr == 0:
        # For perfec data, assume S/N is very high
        snr = 200.0
    # Calculate the bluest Lyman series line wavelength of the highest redshift absorber
    wlim = (1.0+np.max(zdata_all[ispec, :]))*atmdata["Wavelength"][ww][3+nHIwav]
    # Prepare some label arrays
    ID_labels = np.zeros((wdata.shape[0], Nstore), dtype=np.int)
    N_labels = np.zeros((wdata.shape[0], Nstore), dtype=np.float)
    b_labels = np.zeros((wdata.shape[0], Nstore), dtype=np.float)
    z_labels = np.zeros((wdata.shape[0], Nstore), dtype=np.float)
    maxodv = np.zeros((wdata.shape[0], Nstore), dtype=np.float)
    maxcld = np.zeros((wdata.shape[0], Nstore), dtype=np.float)
    widarr = np.arange(wdata.shape[0])
    # Set some constants
    fact = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fc = erf(fact/2.0)  # Fraction of the profile containing the FWHM (i.e. the probability of being within the FWHM)
    dd = ispec
    # Sort by decreasing column density
    srted = np.argsort(Ndata_all[ispec, :])[::-1]
    N_sort = Ndata_all[ispec, :][srted]
    z_sort = zdata_all[ispec, :][srted]
    b_sort = bdata_all[ispec, :][srted]
    wlc = np.where(N_sort > 17.0)
    zmax = np.max(z_sort[wlc])
    print("Preparing ID_labels for spectrum {0:d}/{1:d}".format(dd+1, nspec))
    for zz in range(z_sort.size):
        if z_sort[zz] == -1:
            # No more lines
            break
        if (1.0+z_sort[zz])*HIwav[0] < wlim:
            # Lya line is lower than the highest Lyman series line of the highest redshift absorber
            continue
#        if (1.0+z_sort[zz])*HIwav[-1] < (1.0+zmax)*912.0:
            # HIwav[-1] line of this system is lower wavelength than the Lyman limit of the highest redshift LLS
#            continue
        par = [N_sort[zz], z_sort[zz], b_sort[zz]]
        odtmp = voigt_tau(par, wdata, logn=True)
        for vv in range(HIwav.size):
            if (1.0+z_sort[zz])*HIwav[vv] < (1.0+zmax)*912.0:
                # This H I line of this absorber is at a lower wavelength than the LL of the highest redshift LLS
                continue
            amx = np.argmax(odtmp[:, vv])
            pixdiff = widarr - amx

            # First deal with saturated pixels
            limsat = 10.0/snr
            if np.exp(-odtmp[amx, vv]) <= limsat:
                tst = np.where((np.exp(-odtmp[:, vv]) <= limsat) &
                               (odtmp[:, vv] > maxodv[:, 0]) &
                               (maxodv[:, 0] < -np.log(limsat)))[0]
                if tst.size == 0:
                    # This pixel is saturated, but a nearby absorption feature is stronger
                    pass
                else:
                    # Update the values for the maximum optical depth
                    maxodv[tst, 0] = odtmp[tst, vv]
                    ID_labels[tst, 0] = vv + 1
                    N_labels[tst, 0] = N_sort[zz]
                    b_labels[tst, 0] = b_sort[zz]
                    z_labels[tst, 0] = pixdiff[tst]
            # Now deal with unsaturated pixels
            tst = np.where((odtmp[:, vv] > maxodv[:, 0]) &
                           (odtmp[:, vv] > snr_thresh / snr) &
                           (np.abs(pixdiff) < 1000))[0]
            if fdata[amx] > limsat*0.5:
                if tst.size == 0:
                    # Does not contribute maximum optical depth - try next
                    tst = np.where((odtmp[:, vv] > maxodv[:, 1]) &
                                   (odtmp[:, vv] > snr_thresh / snr) &
                                   (np.abs(pixdiff) < 1000))[0]
                    if tst.size == 0:
                        # This line doesn't contribute significantly to the optical depth
                        continue
                    else:
                        # Update the labels at the corresponding locations
                        maxodv[tst, 1] = odtmp[tst, vv]
                        ID_labels[tst, 1] = vv+1
                        N_labels[tst, 1] = N_sort[zz]
                        b_labels[tst, 1] = b_sort[zz]
                        z_labels[tst, 1] = pixdiff[tst]
                else:
                    # Shift down the maximum optical depth and labels
                    maxodv[tst, 1] = maxodv[tst, 0].copy()
                    ID_labels[tst, 1] = ID_labels[tst, 0].copy()
                    N_labels[tst, 1] = N_labels[tst, 0].copy()
                    b_labels[tst, 1] = b_labels[tst, 0].copy()
                    z_labels[tst, 1] = z_labels[tst, 0].copy()
                    # Update the values for the maximum optical depth
                    maxodv[tst, 0] = odtmp[tst, vv]
                    ID_labels[tst, 0] = vv + 1
                    N_labels[tst, 0] = N_sort[zz]
                    b_labels[tst, 0] = b_sort[zz]
                    z_labels[tst, 0] = pixdiff[tst]

                # Given S/N of spectrum, estimate significance
                # EW = 10.0**Ndata_all[ispec, zz] * HIflm[vv] * HIwlm[vv]**2 / 1.13E20
                # nsig = snr_thresh  # Record all line positions for perfect data
                # if snr > 0:
                #     # Estimate the significance of every feature
                #     bFWHM = (fact/np.sqrt(2.0)) * bdata_all[ispec, zz]
                #     dellam = wvpass[vv] * bFWHM / 299792.458  # Must be in Angstroms
                #     nsig = fc * EW * snr / dellam
                # if nsig >= snr_thresh:
                #     maxodv[amin] = odtmp[vv]
                #     ID_labels[amin] = vv+1
                #     N_labels[amin] = Ndata_all[ispec, zz]
                #     b_labels[amin] = bdata_all[ispec, zz]
    return ID_labels, N_labels, b_labels, z_labels


# Load the data
plotcont = False
plotlabl = False
snr = 0
print("Loading dataset...")
fname, fdata_all, wdata, zdata_all, Ndata_all, bdata_all = load_dataset(3.0, snr, normalise=True)
print("Complete")
nspec = fdata_all.shape[0]
ID_labels = np.zeros((nspec, wdata.shape[0], 2))
N_labels = np.zeros((nspec, wdata.shape[0], 2))
b_labels = np.zeros((nspec, wdata.shape[0], 2))
z_labels = np.zeros((nspec, wdata.shape[0], 2))
for ispec in range(nspec):
    if ispec > 1: continue
    ID_labels[ispec, :, :], \
    N_labels[ispec, :, :], \
    b_labels[ispec, :, :], \
    z_labels[ispec, :, :] = generate_labels(ispec, wdata, fdata_all[ispec, :], zdata_all, Ndata_all, bdata_all, snr=snr)

# Plot the ID_labels to ensure this has been done correctly
if plotlabl:
    specplot = 1
    cont = generate_continuum(specplot, wdata)
    ymin, ymax = 0.0, np.max(fdata_all[specplot, :])
    plt.plot(wdata, fdata_all[specplot, :], 'k-', drawstyle='steps')
    # Plot Lya
    ww = np.where(ID_labels[specplot, :] == 1)
    plt.vlines(HIwav[0]*(1.0+zdata_all[specplot, :].flatten()), ymin, ymax, 'r', '-')
    plt.plot(wdata[ww], cont[ww], 'ro', drawstyle='steps')
    # Plot Lyb
    ww = np.where(ID_labels[specplot, :] == 2)
    plt.vlines(HIwav[1]*(1.0+zdata_all[specplot, :].flatten()), ymin, ymax, 'g', '--')
    plt.plot(wdata[ww], cont[ww], 'go', drawstyle='steps')
    # Plot Lyg
    ww = np.where(ID_labels[specplot, :] == 3)
    plt.vlines(HIwav[2]*(1.0+zdata_all[specplot, :].flatten()), ymin, ymax, 'b', ':')
    plt.plot(wdata[ww], cont[ww], 'bo', drawstyle='steps')
    plt.show()

print("WARNING :: By continuing, you will save/overwrite the previously stored label data...")
pdb.set_trace()
# Save the ID_labels and the data
print("Saving the ID_labels: Check the following sizes are the same")
print(ID_labels.shape, fdata_all.shape)
np.save(fname.replace(".npy", "_fluxonly_{0:d}.npy".format(nspec)), fdata_all)
np.save(fname.replace(".npy", "_IDlabelonly_{0:d}.npy".format(nspec)), ID_labels)
np.save(fname.replace(".npy", "_Nlabelonly_{0:d}.npy".format(nspec)), N_labels)
np.save(fname.replace(".npy", "_blabelonly_{0:d}.npy".format(nspec)), b_labels)
np.save(fname.replace(".npy", "_zlabelonly_{0:d}.npy".format(nspec)), z_labels)

if True:
    plt.plot(fdata_all[0, :] * 30)
    plt.plot(z_labels[0, :, 0])
    plt.show()
    #plt.plot(z_labels[0, :, 0])
    #plt.plot(N_labels[0, :, 0])
    plt.plot(fdata_all[0, :], 'k-', drawstyle='steps')
    tlocs = np.where(z_labels[0, :, 0] == 1)[0] - 1
    plt.vlines(tlocs, 0, 1, 'r', '-')
    plt.show()

if plotcont:
    for ispec in range(10):
        cont = generate_continuum(ispec, wdata)
        plt.plot(wdata, fdata_all[ispec, :], 'k-')
        plt.plot(wdata, cont, 'r-')
        plt.show()
        plt.clf()
