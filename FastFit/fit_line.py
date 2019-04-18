import pdb
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import wofz

# Get a spline representation of the Voigt function
import voigt_spline as vs
vfunc = vs.generate_spline()

# H I Lya params
wave0 = 1215.6701 * 1.0e-8  # convert A to cm
fval = 0.4164
gamma = 6.265E8
c = 299792.458
beta = 3.76730313461770655E11


def voigt(par, wavein, logn=True):
    # Column density
    if logn:
        cold = 10.0 ** par[0]
    else:
        cold = par[0]
    # Redshift
    zp1 = par[1] + 1.0
    wv = par[3]
    # Doppler parameter
    bl = par[2] * wv / 2.99792458E5
    a = par[5] * wv * wv / (3.76730313461770655E11 * bl)
    cns = wv * wv * par[4] / (bl * 2.002134602291006E12)
    cne = cold * cns
    ww = (wavein * 1.0e-8) / zp1
    v = wv * ww * ((1.0 / ww) - (1.0 / wv)) / bl
    tau = cne * wofz(v + 1j * a).real
    # tau = cne*voigtking(v, a)
    return np.exp(-1.0 * tau)


def voigt_spl(par, wavein, logn=True):
    # Column density
    if logn:
        cold = 10.0 ** par[0]
    else:
        cold = par[0]
    # Redshift
    zp1 = par[1] + 1.0
    wv = par[3]
    # Doppler parameter
    bl = par[2] * wv / 2.99792458E5
    a = par[5] * wv * wv / (3.76730313461770655E11 * bl)
    cns = wv * wv * par[4] / (bl * 2.002134602291006E12)
    cne = cold * cns
    ww = (wavein * 1.0e-8) / zp1
    v = wv * ww * ((1.0 / ww) - (1.0 / wv)) / bl
    v *= -1
    sig_j = wofz(v + 1j * a).real#vfunc(v, a)
    dsig_dv = vfunc(v, a, dx=1)
    dsig_da = vfunc(v, a, dy=1)
    tau = cne * sig_j
    # tau = cne*voigtking(v, a)
    return np.exp(-1.0 * tau), sig_j, dsig_dv, dsig_da


def chisqmin(indict, rate=1.0E-10):
    # NOTE: The rate needs to be increased/decreased depending on whether the new chi-sq is better or worse (i.e. as in LM)
    # Enter loop to continue
    converged = False
    params = indict['p0']
    csq_old = 1.0E99
    itnum = 1
    while not converged:
        # Calculate the model
        par = np.append(params, [wave0, fval, gamma])
        model, sig_j, dsig_dv, dsig_da = voigt_spl(par, indict['wave'])
        # Calculate the chi-squared
        chi = (model-indict['flux'])/indict['error']
        csq_new = np.sum(chi**2)
        print("Iteration {0:d} :: chi-squared =".format(itnum), csq_new)
        # Calculate the parameter derivates
        coldens = 10.0 ** params[0]
        Nloglin = coldens * np.log(10)
        zp1 = 1.0 + params[1]
        bval = params[2]
        k_j = fval * wave0 * 299792.458 / (2.002134602291006E12 * bval)
        c_b = (299792.458 / bval)
        wvscl = 1.0 - indict['wave'] * 1.0E-8 / (zp1 * wave0)
        fact = gamma * wave0 / 3.76730313461770655E11
        dM_dN = -model * k_j * sig_j * Nloglin
        dM_dz = model * k_j * coldens * (indict['wave'] * 1.0E-8 / wave0) * c_b * dsig_dv / zp1 ** 2
        dM_db = -model * k_j * (coldens / bval) * (sig_j + c_b * (-dsig_dv * wvscl + dsig_da * fact))
        # Test if we have converged
        if abs(csq_old-csq_new) < 0.001:
            converged = True
        else:
            if csq_new > csq_old:
                # We moved to a bad place, increase the rate, and go back to the old parameters
                rate /= 10.0
            else:
                # Reduce the rate, we are converging
                rate *= 2.0
                csq_old = csq_new
                # TODO :: convolve
                #pdb.set_trace()
                print(params)
                print(rate * np.sum(dM_dN * 2.0 * chi / indict['error']))
                print(rate * np.sum(dM_dz * 2.0 * chi / indict['error']))
                print(rate * np.sum(dM_db * 2.0 * chi / indict['error']))
                params[0] -= rate * np.sum(dM_dN * 2.0 * chi / indict['error'])
                params[1] -= rate * np.sum(dM_dz * 2.0 * chi / indict['error'])
                params[2] -= rate * np.sum(dM_db * 2.0 * chi / indict['error'])
        if converged or itnum > 10:
            break
        itnum += 1

    results = dict(params=params)
    return results

# Generate some fake data
snr = 30.0
cold = 14.7
zabs = 3.0
bval = 10.0
par = [cold, zabs, bval, wave0, fval, gamma]
p0 = np.array([14.5, 3.0, 20.0])

wv_arr = np.linspace(4860.0, 4865.0, 125)
md_arr = voigt(par, wv_arr)
fe_arr = np.ones(wv_arr.size)/snr
fx_arr = np.random.normal(md_arr, fe_arr)

plotit = False
if plotit:
    param = np.append(p0, [wave0, fval, gamma])
    model, sig_j, dsig_dv, dsig_da = voigt_spl(param, wv_arr)
    md_tst,_,_,_ = voigt_spl(par, wv_arr)
    plt.plot(wv_arr, fx_arr, 'k-', drawstyle='steps')
    plt.plot(wv_arr, md_arr, 'r-')
    plt.plot(wv_arr, md_tst, 'b--')
    plt.plot(wv_arr, model, 'm--')
    plt.show()
    plt.clf()
    plt.plot(wv_arr, md_arr-md_tst, 'k-')
    plt.show()

indict = dict(p0=p0, wave=wv_arr, flux=fx_arr, error=fe_arr)
outpar = chisqmin(indict)
