import numpy as np
import fastforest as ff
from matplotlib import pyplot as plt

factor = 0.05
nN, nB = 100, 20
NHI = np.linspace(12.0, 18.0, nN)
zabs = 0.0
bHI = np.linspace(10.0, 30.0, nB)
wave = np.linspace(1210.0, 1220.0, 1000)
dwave = wave[1]-wave[0]
par = [NHI[-1], zabs, bHI[-1]]
mod = ff.voigt(par, wave)
#plt.plot(wave, mod)
#plt.show()


depthwidth = np.zeros((nN, nB))
for n in range(nN):
    for b in range(nB):
        par = [NHI[n], zabs, bHI[b]]
        model = ff.voigt(par, wave)
        mnval = np.min(model)
        if mnval > factor:
            depthwidth[n, b] = mnval
        else:
            val = dwave * np.where(model < factor)[0].size
            depthwidth[n, b] = val

plt.imshow(depthwidth)
plt.show()