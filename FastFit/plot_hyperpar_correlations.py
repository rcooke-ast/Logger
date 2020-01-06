import pdb
import glob
import copy
import pickle
import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt


def load_obj(dirname):
    with open(dirname + '.pkl', 'rb') as f:
        return pickle.load(f)


filedir = 'fit_data/multiscr_blend/'
files = glob.glob(filedir+'model_*.log')
cntr = 0
objs = []
for file in files:
    tab_m = Table.read(file, format='ascii.csv')

    if np.log10(tab_m['loss'])[-1] < 1.75 or True:
        par = load_obj(file.replace('.log', ''))
        pk = par.keys()
        if cntr == 0:
            xplot = [[] for all in pk]
            yplot = [[] for all in pk]
            name = []
            print(len(xplot))
        for ii, key in enumerate(pk):
            xplot[ii].append(par[key])
            yplot[ii].append(np.log10(tab_m['output_z_loss'][-1]))
            # loss  output_N_loss
            if cntr == 0:
                name.append(key)
        cntr += 1
nax = len(xplot)
naxx = int(np.ceil(np.sqrt(nax)))
for pp in range(nax):
    plt.subplot(naxx, naxx, pp+1)
    plt.plot(xplot[pp], yplot[pp], 'bx')
    plt.title(name[pp])
plt.show()
