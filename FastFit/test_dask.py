import os
import sys
import pdb
import numpy as np
from dask.array import Array
from itertools import product

nHIwav = 4


def make_filelist(param, ival=None, zem=3.0, snr=0, numspec=25000):
    if ival is None:
        ival = [0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]
    sval = [0, 5000, 10000, 15000, 20000]
    filelist = []
    for ii in ival:
        zstr = "zem{0:.2f}".format(zem)
        sstr = "snr{0:d}".format(int(snr))
        extxt = "{0:s}_{1:s}_nspec{2:d}_i{3:d}".format(zstr, sstr, numspec, ii)
        if param == "flux":
            nchunk = 25000
            filelist.append("cnn_qsospec_fluxspec_{0:s}_normalised_fluxonly".format(extxt))
        else:
            nchunk = 5000
            for ss in sval:
                sstxt = "vs{0:d}-ve{1:d}".format(ss, ss+5000)
                if param == "ID":
                    filelist.append("cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_IDlabelonly_{2:s}".format(extxt, nHIwav, sstxt))
                elif param == "N":
                    filelist.append("cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_Nlabelonly_{2:s}".format(extxt, nHIwav, sstxt))
                elif param == "z":
                    filelist.append("cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_zlabelonly_{2:s}".format(extxt, nHIwav, sstxt))
                elif param == "b":
                    filelist.append("cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_blabelonly_{2:s}".format(extxt, nHIwav, sstxt))
                else:
                    print("Unrecognised parameter")
    return filelist, nchunk


filelist, nchunk = make_filelist("ID")

chunks = (len(filelist)*(nchunk,), (50302,), (2,))
axis = 0
dtype = np.dtype(np.float64)
mmap_mode = None    # Must be 'r' or None
dirname = "label_data/"

name = 'from-npy-stack-%s' % dirname
keys = list(product([name], *[range(len(c)) for c in chunks]))
values = [(np.load, os.path.join(dirname, '%s.npy' % filelist[i]), mmap_mode)
          for i in range(len(chunks[axis]))]
dsk = dict(zip(keys, values))

res = Array(dsk, name, chunks, dtype)

print(res)