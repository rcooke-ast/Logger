import pdb
import numpy as np
from utilities import generate_fakespectra

Lya = 1215.6701
Lyb = 1025.7223


def cnn_numabs(zem=3.0, numseg=64, numspec=1):
    """
    zem = qso redshift
    numseg = number of pixels in a segment (i.e. the number of pixels in a single "sample" in Keras-speak)
    numspec = number of spectra to generate
    """
    # First generate N spectra
    for dd in range(numspec):
        mock_spec, HI_comps = generate_fakespectra(zem, plot_spec=False)
        # Determine number of pixels between the QSO Lya and Lyb lines
        ww = np.where((mock_spec.wavelength > Lyb*(1.0+zem)) &
                      (mock_spec.wavelength < Lya*(1.0+zem)))[0]
        npix = ww.size
        numsamp = npix//numseg  # Total number of samples to generate
        xdata = np.zeros((numsamp, numseg), dtype=np.float)  # Segments
        ydata = np.zeros(numsamp, dtype=np.int)   # Labels
        remdr = npix % numseg
        ww = ww[remdr:]
        zmin = mock_spec.wavelength[ww[0]] / Lya - 1.0
        zarr = HI_comps['z'].data
        for ll in range(numsamp):
            zmax = mock_spec.wavelength[ww[(ll+1) * numseg]] / Lya - 1.0
            wz = np.where((zarr > zmin) & (zarr <= zmax))[0]
            xdata[ll, :] = mock_spec.flux[ww[ll * numseg]:ww[(ll+1) * numseg]]
            ydata[ll] = wz.size
            zmin = zmax
        # TODO :: Need to append final output data together if more than one spectra are generated
    # Save the data
    np.save("train_data/cnn_numabs_spects")
    np.save("train_data/cnn_numabs_labels")


if __name__ == "__main__":
    test = 0
    if test == 0:
        # Generate data for a CNN that detects the number of absorption features within a given window
        cnn_numabs()

