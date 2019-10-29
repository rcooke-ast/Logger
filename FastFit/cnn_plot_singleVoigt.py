import glob
import copy
import pickle
import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt


def load_obj(dirname):
    with open(dirname + '.pkl', 'rb') as f:
        return pickle.load(f)

filedir = 'fit_data/simple/'
files = glob.glob(filedir+'model_????.log')
cntr = 0
objs = []
for file in files:
    tab = Table.read(file, format='ascii.csv')
    if np.log10(tab['loss'])[-1] < -2.0:
        par = load_obj(file.replace('.log', ''))
        if cntr == 0:
            pk = par.keys()
            val, name = [], []
            for key in pk:
                val.append(par[key])
                name.append(key)
            t = Table(copy.deepcopy(val), names=tuple(name))
        else:
            val = []
            for key in name:
                val.append(par[key])
            t.add_row(copy.deepcopy(val))
        plt.plot(tab['epoch'], np.log10(tab['loss']), 'r-')
        plt.plot(tab['epoch'], np.log10(tab['val_loss']), 'r--')
        cntr += 1
print('Number of models satisfying threshold = ', cntr)
plt.show()

print(t)