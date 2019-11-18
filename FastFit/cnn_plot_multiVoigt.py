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

filedir = 'fit_data/multiscr/'
files = glob.glob(filedir+'model_????.log')
cntr = 0
objs = []
for file in files:
    tab_m = Table.read(file, format='ascii.csv')
    #tab_m = Table.read(file.replace('/simple/', '/multiscr/'), format='ascii.csv')
    if np.log10(tab_m['loss'])[-1] < 1.75:
        par = load_obj(file.replace('.log', ''))
        if cntr == 0:
            pk = par.keys()
            val, name = [], []
            for key in pk:
                val.append(par[key])
                name.append(key)
            t = Table(np.array(copy.deepcopy(val)), names=tuple(name))
            print(file, tab_m['loss'][-1])
        else:
            print(file, tab_m['loss'][-1])
            val = []
            for key in name:
                val.append(par[key])
            t.add_row(np.array(copy.deepcopy(val)))
        cntr += 1
    plt.subplot(221)
    plt.plot(tab_m['epoch'], np.log10(tab_m['loss']), 'r-')
    plt.subplot(222)
    plt.plot(tab_m['epoch'], np.log10(tab_m['output_N_loss']), 'r-')
    plt.subplot(223)
    plt.plot(tab_m['epoch'], np.log10(tab_m['output_z_loss']), 'r-')
    plt.subplot(224)
    plt.plot(tab_m['epoch'], np.log10(tab_m['output_b_loss']), 'r-')
    #plt.plot(tab_m['epoch'], np.log10(tab_m['val_loss']), 'r-')
    #plt.subplot(212)
    #plt.plot(tab_m['epoch'], np.log10(tab_s['loss'])-np.log10(tab_m['loss']), 'r--')
    #plt.plot(tab_m['epoch'], np.log10(tab_s['val_loss'])-np.log10(tab_m['val_loss']), 'r-')
print('Number of models satisfying threshold = ', cntr)
plt.show()

t.write('best_pars.txt', format='ascii.fixed_width', overwrite=True)
