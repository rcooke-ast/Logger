import pdb
import numpy as np
import astropy.units as u
from utilities import generate_fakespectra
from matplotlib import pyplot as plt
Lya = 1215.6701 * u.AA
Lyb = 1025.7223 * u.AA


def cnn_numabs(zem=3.0, numseg=512, numspec=1, seed=None, plotsegs=False):
    """
    zem = qso redshift
    numseg = number of pixels in a segment (i.e. the number of pixels in a single "sample" in Keras-speak)
    numspec = number of spectra to generate
    """
    # First generate N spectra
    for dd in range(numspec):
        seedval = dd
        if type(seed) is int:
            seedval = seed
        elif type(seed) is np.ndarray:
            seedval = seed[dd]
        print("Using seed={0:d} for specrum {1:d}/{2:d}".format(seedval, dd+1, numspec))
        mock_spec, HI_comps = generate_fakespectra(zem, plot_spec=False, seed=seedval)
        # Determine number of pixels between the QSO Lya and Lyb lines
        ww = np.where((mock_spec.wavelength > Lyb*(1.0+zem)) &
                      (mock_spec.wavelength < Lya*(1.0+zem)))[0]
        npix = ww.size
        numsamp = npix//numseg  # Total number of samples to generate
        xdata = np.zeros((numsamp, numseg), dtype=np.float)  # Segments
        ydata = np.zeros(numsamp, dtype=np.int)   # Labels
        remdr = npix % numseg
        ww = ww[remdr-1:]
        zmin = mock_spec.wavelength[ww[0]] / Lya - 1.0
        zarr = HI_comps['z'].data
        for ll in range(numsamp):
            zmax = mock_spec.wavelength[ww[(ll+1) * numseg]] / Lya - 1.0
            wz = np.where((zarr > zmin) & (zarr <= zmax))[0]
            xdata[ll, :] = mock_spec.flux[ww[ll * numseg]:ww[(ll+1) * numseg]]
            ydata[ll] = wz.size
            zmin = zmax
            if plotsegs:
                plt.clf()
                plt.plot(mock_spec.wavelength[ww[ll * numseg]:ww[(ll+1) * numseg]],
                         mock_spec.flux[ww[ll * numseg]:ww[(ll+1) * numseg]], 'k-', drawstyle='steps')
                plt.show()
        # Store the data in the master arrays
        if dd == 0:
            xdata_all = xdata.copy()
            ydata_all = ydata.copy()
        else:
            xdata_all = np.append(xdata_all, xdata, axis=0)
            ydata_all = np.append(ydata_all, ydata, axis=0)
        # TODO :: Need to append final output data together if more than one spectra are generated
    print("Generated {0:d} input segments of length {1:d} for training".format(xdata_all.shape[0], numseg))
    print("This requires {0:f} MB of memory.".format(xdata_all.nbytes/1.0E6))
    # Save the data
    np.save("train_data/cnn_numabs_spects", xdata_all)
    np.save("train_data/cnn_numabs_labels", ydata_all)


if __name__ == "__main__":
    test = 0
    if test == 0:
        # Generate data for a CNN that detects the number of absorption features within a given window
        cnn_numabs(numspec=10)

