import glob
import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt

filedir = 'fit_data/simple/'
files = glob.glob(filedir+'model_????.log')
cntr = 0
for file in files:
    tab = Table.read(file, format='ascii.csv')
    if np.log10(tab['loss'])[-1] < -2.0:
        print(file)
        plt.plot(tab['epoch'], np.log10(tab['loss']), 'r-')
        plt.plot(tab['epoch'], np.log10(tab['val_loss']), 'r--')
        cntr += 1
print('Number of models satisfying threshold = ', cntr)
plt.show()
