import pdb
import os
import sys
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from quasars import QSO
matplotlib.use('Qt5Agg')
import fit_spectrum

Hlines = np.array([1215.6701, 1025.7223, 972.5368, 949.7431, 937.8035])

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	# How many lines of the Lyman series should be plotted, and over what (half) velocity interval
	nLyseries = 5
	velwin = 3000.0

	prop = fit_spectrum.Props(QSO("HS1700p6416"))

	# Ignore lines outside of wavelength range
	wmin, wmax = np.min(prop._wave)/(1.0+prop._zem), np.max(prop._wave)/(1.0+prop._zem)

	# Load atomic data
	atom = fit_spectrum.Atomic(wmin=wmin, wmax=wmax)

	fig, axs = plt.subplots(nrows=nLyseries, figsize=(16,9), facecolor="white", sharex=True, sharey=True)
	specs = []
	for i in range(nLyseries):
		lam = Hlines[i]*(1.0+prop._zem)
		velo = 299792.458*(prop._wave-lam)/lam
		specs.append(Line2D(velo, prop._flux, linewidth=1, linestyle='solid', color='k', drawstyle='steps', animated=True))
		axs[i].add_line(specs[-1])

	reg = fit_spectrum.SelectRegions(fig.canvas, axs, specs, prop, atom, vel=velwin, lines=Hlines[:nLyseries])

	axs[0].set_title("Press '?' to list the available options")
	#ax.set_xlim((prop._wave.min(), prop._wave.max()))
	for ax in axs: ax.set_ylim((-0.1, 1.1))
	plt.show()
