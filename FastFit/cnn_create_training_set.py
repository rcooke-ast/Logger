import os
import pdb
import time
import numpy as np
import astropy.units as u
from utilities import generate_fakespectra, load_atomic
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
from linetools.spectra.xspectrum1d import XSpectrum1D
atomic = load_atomic()


def collect_data(zem, snr, numwav, nruns, rmdata=False):
    print("Collecting the data into a single file")
    rmfiles = ["fluxspec", "wavespec", "zvals", "Nvals", "bvals"]
    for numseg in numwav:
        # Load the data
        for nrun in range(nruns):
            zstr = "zem{0:.2f}".format(zem)
            sstr = "snr{0:d}".format(int(snr))
            nstr = "npx{0:d}".format(int(numseg))
            extstr = "{0:s}_{1:s}_{2:s}_nrun{3:d}".format(zstr, sstr, nstr, nrun)
            fdata = np.load("train_data/cnn_img_fluxspec_{0:s}.npy".format(extstr))
            wdata = np.load("train_data/cnn_img_wavespec_{0:s}.npy".format(extstr))
            zdata = np.load("train_data/cnn_img_zvals_{0:s}.npy".format(extstr))
            Ndata = np.load("train_data/cnn_img_Nvals_{0:s}.npy".format(extstr))
            bdata = np.load("train_data/cnn_img_bvals_{0:s}.npy".format(extstr))
            # Check the dimensions match up
            if nrun == 0:
                fdata_all = fdata.copy()
                wdata_all = wdata.copy()
                zdata_all = zdata.copy()
                Ndata_all = Ndata.copy()
                bdata_all = bdata.copy()
                numNzb_all = zdata_all.shape[1]
            else:
                fdata_all = np.append(fdata_all, fdata.copy(), axis=0)
                wdata_all = np.append(wdata_all, wdata.copy(), axis=0)
                # Pad arrays as needed
                zshp = zdata.shape[1]
                zout = zdata.copy()
                Nout = Ndata.copy()
                bout = bdata.copy()
                if numNzb_all < zshp:
                    zdata_all = np.pad(zdata_all, ((0, 0), (0, zshp - numNzb_all)), 'constant', constant_values=-1)
                    Ndata_all = np.pad(Ndata_all, ((0, 0), (0, zshp - numNzb_all)), 'constant', constant_values=-1)
                    bdata_all = np.pad(bdata_all, ((0, 0), (0, zshp - numNzb_all)), 'constant', constant_values=-1)
                    # Update the max array size
                    numNzb_all = zshp
                elif numNzb_all > zshp:
                    zout = np.pad(zout, ((0, 0), (0, numNzb_all - zshp)), 'constant', constant_values=-1)
                    Nout = np.pad(Nout, ((0, 0), (0, numNzb_all - zshp)), 'constant', constant_values=-1)
                    bout = np.pad(bout, ((0, 0), (0, numNzb_all - zshp)), 'constant', constant_values=-1)
                zdata_all = np.append(zdata_all, zout, axis=0)
                Ndata_all = np.append(Ndata_all, Nout, axis=0)
                bdata_all = np.append(bdata_all, bout, axis=0)
        # Save the data into a single file
        zstr = "zem{0:.2f}".format(zem)
        sstr = "snr{0:d}".format(int(snr))
        nstr = "npx{0:d}".format(int(numseg))
        extstr = "{0:s}_{1:s}_{2:s}".format(zstr, sstr, nstr)
        np.save("train_data/cnn_img_fluxspec_{0:s}".format(extstr), fdata_all)
        np.save("train_data/cnn_img_wavespec_{0:s}".format(extstr), wdata_all)
        np.save("train_data/cnn_img_zvals_{0:s}".format(extstr), zdata_all)
        np.save("train_data/cnn_img_Nvals_{0:s}".format(extstr), Ndata_all)
        np.save("train_data/cnn_img_bvals_{0:s}".format(extstr), bdata_all)
        # Delete the individual files that are no longer needed
        if rmdata:
            print("Removing individual files...")
            for nrun in range(nruns):
                zstr = "zem{0:.2f}".format(zem)
                sstr = "snr{0:d}".format(int(snr))
                nstr = "npx{0:d}".format(int(numseg))
                extstr = "{0:s}_{1:s}_{2:s}_nrun{3:d}".format(zstr, sstr, nstr, nrun)
                for rmstr in rmfiles:
                    fname = "train_data/cnn_img_{0:s}_{1:s}.npy".format(rmstr, extstr)
                    os.remove(fname)
                    print("  Removed: {0:s}".format(fname))
    return


def extraction(zem, numseg, numlym, mock_spec, HI_comps, numNzb_all):
    ww = np.where(mock_spec.wavelength < atomic[0] * (1.0 + zem))[0]
    npix = ww.size
    zarr = HI_comps['z'].data
    Narr = HI_comps['lgNHI'].data
    barr = HI_comps['bval'].value
    numsamp = npix // numseg  # Total number of samples to generate
    numNzb = max(numNzb_all, zarr.size // numsamp)  # The maximum number of lines in any segment is guessed to be the average
    numsamp *= 2  # Do half window shifts too
    fdata = -1 * np.ones((numsamp, numlym, numseg), dtype=np.float)  # Flux Segments
    wdata = -1 * np.ones((numsamp, numlym, numseg), dtype=np.float)  # Wavelength Segments
    zdata = -1 * np.ones((numsamp, numNzb), dtype=np.float)  # Labels (redshift)
    Ndata = -1 * np.ones((numsamp, numNzb), dtype=np.float)  # Labels (column density)
    bdata = -1 * np.ones((numsamp, numNzb), dtype=np.float)  # Labels (Doppler parameter)
    remdr = npix % numseg
    ww = ww[remdr - 1:]
    for ll in range(numsamp):
        idxmin = ww[(ll + 0) * numseg//2]  # Divide by 2 is to consider half window shifts
        zmin = mock_spec.wavelength[idxmin] / atomic[0] - 1.0
        zmax = mock_spec.wavelength[idxmin+numseg] / atomic[0] - 1.0
        wz = np.where((zarr > zmin) & (zarr <= zmax))
        wzsz = wz[0].size
        for ss in range(numlym):
            wmin = (1.0 + zmin) * atomic[ss]
            wmax = (1.0 + zmax) * atomic[ss]
            if wmin < mock_spec.wavelength[0]:
                # No data, so this will be masked
                continue
            ixmin = np.argmin(np.abs(mock_spec.wavelength-wmin))
            ixmax = np.argmin(np.abs(mock_spec.wavelength-wmax))
            fdata[ll, ss, :] = mock_spec.flux[ixmin:ixmax] / np.max(mock_spec.flux[ixmin:ixmax])
            wdata[ll, ss, :] = mock_spec.wavelength[ixmin:ixmax]
        # Save the details of all (Lya) absorption lines - ignore blends
        if wzsz > numNzb:
            # First pad the arrays so the new values can fit
            zdata = np.pad(zdata, ((0, 0), (0, wzsz - numNzb)), 'constant', constant_values=-1)
            Ndata = np.pad(Ndata, ((0, 0), (0, wzsz - numNzb)), 'constant', constant_values=-1)
            bdata = np.pad(bdata, ((0, 0), (0, wzsz - numNzb)), 'constant', constant_values=-1)
            # Reset the size of the array
            numNzb = wzsz
        # Set the array values
        zdata[ll, :wzsz] = zarr[wz]
        Ndata[ll, :wzsz] = Narr[wz]
        bdata[ll, :wzsz] = barr[wz]
    return fdata, wdata, zdata, Ndata, bdata


def cnn_lyman_images(zem=3.0, numwav=None, numlym=10, numspec=1, seed=None, snr=30, nrun=0):
    """
    zem = qso redshift
    numseg = number of pixels in a segment (i.e. the number of pixels in a single "sample" in Keras-speak)
    numspec = number of spectra to generate
    seed = seed for pseudo random number
    snr = S/N ratio of the spectrum
    """
    # Check that numwav has been set
    if numwav is None:
        numwav = [4, 8, 16]

    # Generate numspec spectra
    spec = []
    comp = []
    for dd in range(numspec):
        seedval = dd
        if type(seed) is int:
            seedval = seed
        elif type(seed) is np.ndarray:
            seedval = seed[dd]
        print("Using seed={0:d} for spectrum {1:d}/{2:d}".format(seedval, dd+1, numspec))
        mock_spec, HI_comps = generate_fakespectra(zem, plot_spec=False, snr=snr, seed=seedval)
        spec.append(mock_spec)
        comp.append(HI_comps)

    # Extract all of the information from the spectra
    numNzb_all = [0 for all in numwav]
    for nn, numseg in enumerate(numwav):
        print("Building data for numseg = ", numseg)
        for dd in range(numspec):
            print("  Analysing spectrum {0:d}/{1:d}".format(dd+1, numspec))
            mock_spec, HI_comps = spec[dd], comp[dd]
            fdata, wdata, zdata, Ndata, bdata = extraction(zem, numseg, numlym, mock_spec, HI_comps, numNzb_all[nn])
            # Store the data in the master arrays
            if dd == 0:
                fdata_all = fdata.copy()
                wdata_all = wdata.copy()
                zdata_all = zdata.copy()
                Ndata_all = Ndata.copy()
                bdata_all = bdata.copy()
                numNzb_all[nn] = zdata.shape[1]
            else:
                fdata_all = np.append(fdata_all, fdata, axis=0)
                wdata_all = np.append(wdata_all, wdata, axis=0)
                # Pad arrays as needed
                zshp = zdata.shape[1]
                if numNzb_all[nn] < zshp:
                    zdata_all = np.pad(zdata_all, ((0, 0), (0, zshp - numNzb_all[nn])), 'constant', constant_values=-1)
                    Ndata_all = np.pad(Ndata_all, ((0, 0), (0, zshp - numNzb_all[nn])), 'constant', constant_values=-1)
                    bdata_all = np.pad(bdata_all, ((0, 0), (0, zshp - numNzb_all[nn])), 'constant', constant_values=-1)
                    # Update the max array size
                    numNzb_all[nn] = zshp
                zdata_all = np.append(zdata_all, zdata, axis=0)
                Ndata_all = np.append(Ndata_all, Ndata, axis=0)
                bdata_all = np.append(bdata_all, bdata, axis=0)
        # Save the data for this numseg
        print("Generated {0:d} input segments of length {1:d} for training".format(fdata_all.shape[0], numseg))
        print("This requires {0:f} MB of memory.".format(fdata_all.nbytes / 1.0E6))
        # Save the data into a single file
        zstr = "zem{0:.2f}".format(zem)
        sstr = "snr{0:d}".format(int(snr))
        nstr = "npx{0:d}".format(int(numseg))
        extstr = "{0:s}_{1:s}_{2:s}_nrun{3:d}".format(zstr, sstr, nstr, nrun)
        np.save("train_data/cnn_img_fluxspec_{0:s}".format(extstr), fdata_all)
        np.save("train_data/cnn_img_wavespec_{0:s}".format(extstr), wdata_all)
        np.save("train_data/cnn_img_zvals_{0:s}".format(extstr), zdata_all)
        np.save("train_data/cnn_img_Nvals_{0:s}".format(extstr), Ndata_all)
        np.save("train_data/cnn_img_bvals_{0:s}".format(extstr), bdata_all)
    return


def cnn_numabs(zem=3.0, numseg=512, numspec=1, seed=None, snr=30, plotsegs=False):
    """
    zem = qso redshift
    numseg = number of pixels in a segment (i.e. the number of pixels in a single "sample" in Keras-speak)
    numspec = number of spectra to generate
    seed = seed for pseudo random number
    snr = S/N ratio of the spectrum
    """
    # First generate N spectra
    numNzb_all = 0
    for dd in range(numspec):
        seedval = dd
        if type(seed) is int:
            seedval = seed
        elif type(seed) is np.ndarray:
            seedval = seed[dd]
        print("Using seed={0:d} for spectrum {1:d}/{2:d}".format(seedval, dd+1, numspec))
        mock_spec, HI_comps = generate_fakespectra(zem, plot_spec=False, snr=snr, seed=seedval)
        # Determine number of pixels between the QSO Lya and Lyb lines
        ww = np.where((mock_spec.wavelength > atomic[1]*(1.0+zem)) &
                      (mock_spec.wavelength < atomic[0]*(1.0+zem)))[0]
        # print(HI_comps['lgNHI'])
        # print(HI_comps['z'])
        # print(HI_comps['bval'])
        zarr = HI_comps['z'].data
        Narr = HI_comps['lgNHI'].data
        barr = HI_comps['bval'].value
        wz = np.where((zarr > atomic[1]*(1.0+zem)/atomic[0] - 1.0) * (zarr < zem))
        npix = ww.size
        numsamp = npix//numseg  # Total number of samples to generate
        numNzb = max(numNzb_all, wz[0].size//numsamp)  # The maximum number of lines in any segment is guessed to be the average
        fdata = np.zeros((numsamp, numseg), dtype=np.float)  # Flux Segments
        wdata = np.zeros((numsamp, numseg), dtype=np.float)  # Wavelength Segments
        ldata = np.zeros(numsamp, dtype=np.int)   # Labels (number of absorption lines)
        zdata = -1*np.ones((numsamp, numNzb), dtype=np.float)   # Labels (redshift)
        Ndata = -1*np.ones((numsamp, numNzb), dtype=np.float)   # Labels (column density)
        bdata = -1*np.ones((numsamp, numNzb), dtype=np.float)   # Labels (Doppler parameter)
        remdr = npix % numseg
        ww = ww[remdr-1:]
        zmin = mock_spec.wavelength[ww[0]] / atomic[0] - 1.0
        for ll in range(numsamp):
            zmax = mock_spec.wavelength[ww[(ll+1) * numseg]] / atomic[0] - 1.0
            wz = np.where((zarr > zmin) & (zarr <= zmax))
            wzsz = wz[0].size
            fdata[ll, :] = mock_spec.flux[ww[ll * numseg]:ww[(ll+1) * numseg]]
            wdata[ll, :] = mock_spec.wavelength[ww[ll * numseg]:ww[(ll+1) * numseg]]
            ldata[ll] = wzsz
            # Save the details of all absorption lines
            if wzsz > numNzb:
                # First pad the arrays so the new values can fit
                zdata = np.pad(zdata, ((0, 0), (0, wzsz-numNzb)), 'constant', constant_values=-1)
                Ndata = np.pad(Ndata, ((0, 0), (0, wzsz-numNzb)), 'constant', constant_values=-1)
                bdata = np.pad(bdata, ((0, 0), (0, wzsz-numNzb)), 'constant', constant_values=-1)
                # Reset the size of the array
                numNzb = wzsz
            # Set the array values
            zdata[ll, :wzsz] = zarr[wz]
            Ndata[ll, :wzsz] = Narr[wz]
            bdata[ll, :wzsz] = barr[wz]
            # Reset the minimum redshift
            zmin = zmax
            if plotsegs:
                plt.clf()
                plt.plot(mock_spec.wavelength[ww[ll * numseg]:ww[(ll+1) * numseg]],
                         mock_spec.flux[ww[ll * numseg]:ww[(ll+1) * numseg]], 'k-', drawstyle='steps')
                plt.show()
        # Store the data in the master arrays
        if dd == 0:
            fdata_all = fdata.copy()
            wdata_all = wdata.copy()
            ldata_all = ldata.copy()
            zdata_all = zdata.copy()
            Ndata_all = Ndata.copy()
            bdata_all = bdata.copy()
            numNzb_all = numNzb
        else:
            fdata_all = np.append(fdata_all, fdata, axis=0)
            wdata_all = np.append(wdata_all, wdata, axis=0)
            ldata_all = np.append(ldata_all, ldata, axis=0)
            # Pad arrays as needed
            zshp = zdata.shape[1]
            if numNzb_all < zshp:
                zdata_all = np.pad(zdata_all, ((0, 0), (0, zshp-numNzb_all)), 'constant', constant_values=-1)
                Ndata_all = np.pad(Ndata_all, ((0, 0), (0, zshp-numNzb_all)), 'constant', constant_values=-1)
                bdata_all = np.pad(bdata_all, ((0, 0), (0, zshp-numNzb_all)), 'constant', constant_values=-1)
                # Update the max array size
                numNzb_all = zshp
            zdata_all = np.append(zdata_all, zdata, axis=0)
            Ndata_all = np.append(Ndata_all, Ndata, axis=0)
            bdata_all = np.append(bdata_all, bdata, axis=0)

    return fdata_all, wdata_all, ldata_all, zdata_all, Ndata_all, bdata_all


def cnn_qsospec(zem=3.0, numspec=1, seed=None, snrs=[30], plotsegs=False):
    """
    zem = qso redshift
    numspec = number of spectra to generate
    seed = seed for pseudo random number
    snr = S/N ratio of the spectrum
    """
    # First generate N spectra
    wavebuffer = 5.0*u.AA  # Number of angstroms beyond QSO Lya emission to include in the saved spectrum
    numNzb_all = len(snrs)*[0]
    for dd in range(numspec):
        seedval = dd
        if type(seed) is int:
            seedval = seed
        elif type(seed) is np.ndarray:
            seedval = seed[dd]
        print("Using seed={0:d} for spectrum {1:d}/{2:d}".format(seedval, dd+1, numspec))
        mock_spec, HI_comps = generate_fakespectra(zem, plot_spec=False, snr=-1, seed=seedval)
        ww = np.where(mock_spec.wavelength-wavebuffer < atomic[0]*(1.0+zem))
        npix = ww[0].size
        zarr = HI_comps['z'].data
        Narr = HI_comps['lgNHI'].data
        barr = HI_comps['bval'].value
        rstate = np.random.RandomState(seed)
        mock_spec.add_noise(rstate=rstate)
        for ss, snr in enumerate(snrs):
            numNzb = max(numNzb_all[ss], zarr.size)  # The maximum number of lines in this spectrum
            newerr = mock_spec.sig/snr
            genspec = XSpectrum1D.from_tuple((mock_spec.wavelength, mock_spec.flux, newerr))
            # Add reproducible noise
            noisy_spec = genspec.add_noise(rstate=rstate)
            if dd == 0:
                # Initialize
                fdata_all = len(snrs)*[np.zeros((numspec+1, npix))]
                zdata_all = len(snrs)*[-1*np.ones((numspec, numNzb), dtype=np.float)]   # Labels (redshift)
                Ndata_all = len(snrs)*[-1*np.ones((numspec, numNzb), dtype=np.float)]   # Labels (column density)
                bdata_all = len(snrs)*[-1*np.ones((numspec, numNzb), dtype=np.float)]   # Labels (Doppler parameter)
                # Fill
                fdata_all[ss][0, :] = noisy_spec.wavelength[ww]
                fdata_all[ss][1, :] = noisy_spec.flux[ww]
                numNzb_all[ss] = numNzb
            else:
                fdata_all[ss][dd+1, :] = noisy_spec.flux[ww]
                # Pad arrays as needed
                zshp = zarr.size
                if numNzb_all[ss] < zshp:
                    zdata_all[ss] = np.pad(zdata_all[ss], ((0, 0), (0, zshp-numNzb_all[ss])), 'constant', constant_values=-1)
                    Ndata_all[ss] = np.pad(Ndata_all[ss], ((0, 0), (0, zshp-numNzb_all[ss])), 'constant', constant_values=-1)
                    bdata_all[ss] = np.pad(bdata_all[ss], ((0, 0), (0, zshp-numNzb_all[ss])), 'constant', constant_values=-1)
                    # Update the max array size
                    numNzb_all[ss] = zshp
            zdata_all[ss][dd, :] = zarr.copy()
            Ndata_all[ss][dd, :] = Narr.copy()
            bdata_all[ss][dd, :] = barr.copy()
    for ss, snr in enumerate(snrs):
        # Save the data
        print("Generated {0:d} spectra of length {1:d} for training".format(numspec, npix))
        print("This requires {0:f} MB of memory.".format(fdata_all.nbytes / 1.0E6))
        # Save the data into a single file
        zstr = "zem{0:.2f}".format(zem)
        sstr = "snr{0:d}".format(int(snr))
        extstr = "{0:s}_{1:s}_nspec{2:d}".format(zstr, sstr, numspec)
        np.save("train_data/cnn_qsospec_fluxspec_{0:s}".format(extstr), fdata_all[ss])
        np.save("train_data/cnn_qsospec_zvals_{0:s}".format(extstr), zdata_all[ss])
        np.save("train_data/cnn_qsospec_Nvals_{0:s}".format(extstr), Ndata_all[ss])
        np.save("train_data/cnn_qsospec_bvals_{0:s}".format(extstr), bdata_all[ss])
    return


if __name__ == "__main__":
    # Set some starting variables
    testdata = 2
    multip = True  # Multiprocess?
    starttime = time.time()

    # Now generate the test data
    if testdata == 0:
        # Create small strips around Lyman-alpha lines
        zem = 3.0
        snr = 30
        numseg = 512
        numspec = 200
        if multip:
            nruns = 4
            pool = Pool(processes=cpu_count())
            async_results = []
            for jj in range(nruns):
                seed = np.arange(numspec) + jj * numspec
                async_results.append(pool.apply_async(cnn_numabs, (zem, numseg, numspec, seed, snr)))
            pool.close()
            pool.join()
            map(ApplyResult.wait, async_results)
            # Collect the returned data
            for jj in range(nruns):
                getVal = async_results[jj].get()
                # Store the data in the master arrays
                if jj == 0:
                    fdata_all = getVal[0].copy()
                    wdata_all = getVal[1].copy()
                    ldata_all = getVal[2].copy()
                    zdata_all = getVal[3].copy()
                    Ndata_all = getVal[4].copy()
                    bdata_all = getVal[5].copy()
                    numNzb_all = zdata_all.shape[1]
                else:
                    fdata_all = np.append(fdata_all, getVal[0].copy(), axis=0)
                    wdata_all = np.append(wdata_all, getVal[1].copy(), axis=0)
                    ldata_all = np.append(ldata_all, getVal[2].copy(), axis=0)
                    # Pad arrays as needed
                    zshp = getVal[3].shape[1]
                    zout = getVal[3].copy()
                    Nout = getVal[4].copy()
                    bout = getVal[5].copy()
                    if numNzb_all < zshp:
                        zdata_all = np.pad(zdata_all, ((0, 0), (0, zshp - numNzb_all)), 'constant', constant_values=-1)
                        Ndata_all = np.pad(Ndata_all, ((0, 0), (0, zshp - numNzb_all)), 'constant', constant_values=-1)
                        bdata_all = np.pad(bdata_all, ((0, 0), (0, zshp - numNzb_all)), 'constant', constant_values=-1)
                        # Update the max array size
                        numNzb_all = zshp
                    elif numNzb_all > zshp:
                        zout = np.pad(zout, ((0, 0), (0, numNzb_all - zshp)), 'constant', constant_values=-1)
                        Nout = np.pad(Nout, ((0, 0), (0, numNzb_all - zshp)), 'constant', constant_values=-1)
                        bout = np.pad(bout, ((0, 0), (0, numNzb_all - zshp)), 'constant', constant_values=-1)
                    zdata_all = np.append(zdata_all, zout, axis=0)
                    Ndata_all = np.append(Ndata_all, Nout, axis=0)
                    bdata_all = np.append(bdata_all, bout, axis=0)
        else:
            fdata_all, wdata_all, ldata_all, zdata_all, Ndata_all, bdata_all =\
                cnn_numabs(zem=zem, numseg=numseg, numspec=numspec, snr=snr)
        print("Generated {0:d} input segments of length {1:d} for training".format(fdata_all.shape[0], numseg))
        print("This requires {0:f} MB of memory.".format(fdata_all.nbytes / 1.0E6))
        # Save the data into a single file
        zstr = "zem{0:.2f}".format(zem)
        sstr = "snr{0:d}".format(int(snr))
        extstr = "{0:s}_{1:s}".format(zstr, sstr)
        np.save("train_data/cnn_fluxspec_{0:s}".format(extstr), fdata_all)
        np.save("train_data/cnn_wavespec_{0:s}".format(extstr), wdata_all)
        np.save("train_data/cnn_numbrabs_{0:s}".format(extstr), ldata_all)
        np.save("train_data/cnn_zvals_{0:s}".format(extstr), zdata_all)
        np.save("train_data/cnn_Nvals_{0:s}".format(extstr), Ndata_all)
        np.save("train_data/cnn_bvals_{0:s}".format(extstr), bdata_all)
    elif testdata == 1:
        # Create small images of the Lyman series
        zem = 3.0
        #snrs = [0, 20, 50, 100, 200]
        snrs = [20, 50, 100, 200]
        numwav = [4, 8, 16, 32, 64]   # Number of pixels in the wavelength direction
        numlym = 8   # Number of Lyman series lines to use
        numspec = 12
        if multip:
            for snr in snrs:
                nruns = 4
                pool = Pool(processes=cpu_count())
                async_results = []
                for jj in range(nruns):
                    seed = np.arange(numspec) + jj * numspec
                    async_results.append(pool.apply_async(cnn_lyman_images, (zem, numwav, numlym, numspec, seed, snr, jj)))
                pool.close()
                pool.join()
                map(ApplyResult.wait, async_results)
                # Collect all data into a single file
                collect_data(zem, snr, numwav, nruns, rmdata=True)
        else:
            cnn_lyman_images(zem, numwav, numlym, numspec, snrs[0], 0)
    elif testdata == 2:
        # Create a series of QSO spectra
        zem = 3.0
        numspec = 1000
        snrs = [0, 20, 50, 100, 200]
        seed = np.arange(numspec)
        cnn_qsospec(zem, numspec, seed, snrs=snrs)
    else:
        pass
    tottime = time.time() - starttime
    print("Total execution time = {0:f} minutes".format(tottime / 60.0))
