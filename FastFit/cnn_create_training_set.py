import pdb
import time
import numpy as np
import astropy.units as u
from utilities import generate_fakespectra
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
Lya = 1215.6701 * u.AA
Lyb = 1025.7223 * u.AA


def cnn_numabs(zem=3.0, numseg=512, numspec=1, seed=None, snr=30, plotsegs=False):
    """
    zem = qso redshift
    numseg = number of pixels in a segment (i.e. the number of pixels in a single "sample" in Keras-speak)
    numspec = number of spectra to generate
    seed = seed for pseudo random number
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
        ww = np.where((mock_spec.wavelength > Lyb*(1.0+zem)) &
                      (mock_spec.wavelength < Lya*(1.0+zem)))[0]
        # print(HI_comps['lgNHI'])
        # print(HI_comps['z'])
        # print(HI_comps['bval'])
        zarr = HI_comps['z'].data
        Narr = HI_comps['lgNHI'].data
        barr = HI_comps['bval'].value
        wz = np.where((zarr > Lyb*(1.0+zem)/Lya - 1.0) * (zarr < zem))
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
        zmin = mock_spec.wavelength[ww[0]] / Lya - 1.0
        for ll in range(numsamp):
            zmax = mock_spec.wavelength[ww[(ll+1) * numseg]] / Lya - 1.0
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


if __name__ == "__main__":
    # Set some starting variables
    testdata = 0
    multip = True  # Multiprocess?
    starttime = time.time()

    # Now generate the test data
    if testdata == 0:
        # Generate data for a CNN that detects the number of absorption features within a given window
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
    else:
        pass
    tottime = time.time() - starttime
    print("Total execution time = {0:f} minutes".format(tottime / 60.0))
