import pdb
import os
import sys
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from quasars import QSO
matplotlib.use('Qt5Agg')
import fit_spectrum

Hlines = np.array([1215.6701, 1025.7223, 972.5368, 949.7431, 937.8035, 930.7483, 926.2257, 923.1504, 920.9631, 919.3514, 918.1294])

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # How many lines of the Lyman series should be plotted, and over what (half) velocity interval
    nLyseries = 8  # Must be even
    velwin = 1500.0
    rcen, rwid = 1.0, 0.05

    prop = fit_spectrum.Props(QSO("HS1700p6416"))

    # Ignore lines outside of wavelength range
    wmin, wmax = np.min(prop._wave)/(1.0+prop._zem), np.max(prop._wave)/(1.0+prop._zem)

    # Load atomic data
    atom = fit_spectrum.Atomic(wmin=wmin, wmax=wmax)

    fig, axs = plt.subplots(nrows=nLyseries//2, ncols=2, figsize=(16, 9), facecolor="white", sharex=True, sharey=True)
    axs = axs.T.flatten()

    # First draw 1/2 sigma residuals bars
    for i in range(nLyseries):
        lam = Hlines[i]*(1.0+prop._zem)
        velo = 299792.458*(prop._wave-lam)/lam
        axs[i].fill_between(velo, rcen-2*rwid, rcen+2*rwid, facecolors='lightgrey')  # 2 sigma bar
        axs[i].fill_between(velo, rcen-1*rwid, rcen+1*rwid, facecolors='darkgrey')  # 1 sigma bar

    # Draw spectra and residuals
    specs = []
    for i in range(nLyseries):
        lam = Hlines[i]*(1.0+prop._zem)
        velo = 299792.458*(prop._wave-lam)/lam
        specs.append(Line2D(velo, prop._flux, linewidth=1, linestyle='solid', color='k', drawstyle='steps', animated=True))
        axs[i].add_line(specs[-1])

    # Add some information GUI axis
    for ax in axs: ax.set_ylim((-0.1, 1.1))
    axi = fig.add_axes([0.15, .9, .7, 0.08])
    axi.get_xaxis().set_visible(False)
    axi.get_yaxis().set_visible(False)
    axi.text(0.5, 0.5, "Press '?' to list the available options", transform=axi.transAxes,
             horizontalalignment='center', verticalalignment='center')
    axi.set_xlim((0, 1))
    axi.set_ylim((0, 1))

    reg = fit_spectrum.SelectRegions(fig.canvas, axs, axi, specs, prop, atom, vel=velwin, lines=Hlines[:nLyseries], resid=[rcen, rwid])

    plt.show()
