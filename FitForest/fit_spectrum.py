import pdb
import os
import sys
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
from quasars import QSO
from scipy.optimize import curve_fit
from scipy.special import wofz
matplotlib.use('Qt5Agg')

def voigt(wave, p0, p1, p2, lam, fvl, gam):
    cold = 10.0**p0
    zp1=p1+1.0
    wv=lam*1.0e-8
    bl=p2*wv/2.99792458E5
    a=gam*wv*wv/(3.76730313461770655E11*bl)
    cns=wv*wv*fvl/(bl*2.002134602291006E12)
    cne=cold*cns
    ww=(wave*1.0e-8)/zp1
    v=wv*ww*((1.0/ww)-(1.0/wv))/bl
    tau = cne*wofz(v + 1j * a).real
    return np.exp(-1.0*tau)

def modwrite(vals, wminmax):
    outfile = open("outfile.mod", 'w')
    outfile.write("# Change the default settings\n")
    outfile.write("run ncpus -1\nrun nsubpix 5\nrun blind False\nrun convergence False\nrun convcriteria 0.2\n")
    outfile.write("chisq atol 0.01\nchisq xtol 0.0\nchisq ftol 0.0\nchisq gtol 0.0\nchisq  miniter  10\nchisq  maxiter  3000\n")
    outfile.write("out model True\nout fits True\nout verbose 1\nout overwrite True\nout covar datafile.mod.out.covar\n")
    outfile.write("plot dims 3x3\nplot ticklabels True\nplot labels True\nplot fitregions True\nplot fits True\n\n")
    outfile.write("# Read in the data\ndata read\n")
    outfile.write("  datafile.dat  specid=0  fitrange=[{0:f},{1:f}]  resolution=vfwhm(6.28vh)  columns=[wave:0,flux:1,error:2]  plotone=True\ndata end\n\n".format(wminmax[0],wminmax[1]))
    outfile.write("# Read in the model\nmodel read\n")
    outfile.write(" fix vfwhm value True\n lim voigt bturb [0.2,None]\n lim voigt ColDens [8.0,22.0]\n")
    outfile.write("emission\n # Specify the continuum\n constant 1.0  specid=0\nabsorption\n # Specify the absorption\n")
    for i in range(vals.shape[0]):
        outfile.write(" voigt ion=1H_I {0:f} {1:f} {2:f} 0.0ZEROT specid=0\n".format(vals[i,0],vals[i,1],vals[i,2]))
    outfile.write("model end\n")
    outfile.close()


class SelectRegions(object):
    """
    Generate a model and regions to be fit with ALIS
    """

    def __init__(self, canvas, axs, specs, prop, atom, vel=500.0, lines=None):
        """
        vel : float
          Default +/- plotting window in km/s
        """
        self.axs = axs
        self.naxis = len(axs)
        self.specs = specs
        self.prop = prop
        self.atom = atom
        self.veld = vel
        self.curreg = [None for i in range(self.naxis)]
        self.fitreg = [None for i in range(self.naxis)]
        self.backgrounds = [None for i in range(self.naxis)]
        self.lineidx = 0
        self.axisidx = 0
        self._addsub = 0  # Adding a region (1) or removing (0)
        self._changes = False
        self.annlines = []
        self.anntexts = []

        if lines is None:
            lines = 1215.6701*np.ones(self.naxis)
        self.lines = lines

        # Create the model lines variables
        self.modelpars = None
        self.modelvals = None
        self.modelcrvs = [None for ii in range(self.naxis)]
        self.actors = [np.zeros(self.prop._wave.size) for ii in range(self.naxis)]

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas = canvas

        # Sort the lines by decreasing probability of detection
        self.sort_lines()
        self.get_current_line()
        
        # Draw the first line
        self.next_spectrum()

    def draw_lines(self):
        """
        Draw labels on the absorption lines for the input redshift
        """
        if self.modelpars is None:
            # There are no models at the moment
            return
        for i in self.annlines: i.remove()
        for i in self.anntexts: i.remove()
        self.annlines = []
        self.anntexts = []
        for i in range(self.naxis):
            for j in range(self.modelpars.shape[0]):
                lam = self.lines[i]*(1.0+self.prop._zem)
                velo = 299792.458*(self.lines[i]*(1.0+self.modelpars[j,1])-lam)/lam
                self.annlines.append(self.axs[i].axvline(velo, color='r'))
        return

        #annotations = [child for child in self.ax.get_children() if isinstance(child, matplotlib.text.Annotation)]
        for i in self.annlines: i.remove()
        for i in self.anntexts: i.remove()
        self.annlines = []
        self.anntexts = []
        for ax in self.axs:
            # Plot the lines
            xmn, xmx = ax.get_xlim()
            ymn, ymx = ax.get_ylim()
            xmn /= (1.0+self.prop._zem)
            xmx /= (1.0+self.prop._zem)
            w = np.where((self.atom._atom_wvl > xmn) & (self.atom._atom_wvl < xmx))[0]
            for i in range(w.size):
                dif = i%5
                self.annlines.append(ax.axvline(self.atom._atom_wvl[w[i]]*(1.0+self.prop._zem), color='b'))
                txt = "{0:s} {1:s} {2:.1f}".format(self.atom._atom_atm[w[i]],self.atom._atom_ion[w[i]],self.atom._atom_wvl[w[i]])
                ylbl = ymn + (ymx-ymn)*(dif+1.5)/8.0
                self.anntexts.append(ax.annotate(txt, (self.atom._atom_wvl[w[i]]*(1.0+self.prop._zem), ylbl), rotation=90.0, color='b', ha='center', va='bottom'))
        return

    def draw_model(self):
        if self.modelpars is None:
            # There are no models at the moment
            return
        for i in self.modelcrvs:
            if i is not None:
                i.pop(0).remove()
        # Generate the model curve
        model = np.ones(self.prop._wave.size)
        for i in range(self.modelpars.shape[0]):
            for j in range(self.naxis):
                p0, p1, p2 = self.modelpars[i,0], self.modelpars[i,1], self.modelpars[i,2]
                atidx = np.argmin(np.abs(self.lines[j]-self.atom._atom_wvl))
                wv = self.lines[j]
                fv = self.atom._atom_fvl[atidx]
                gm = self.atom._atom_gam[atidx]
                model *= voigt(self.prop._wave, p0, p1, p2, wv, fv, gm)
        # Plot the model
        for i in range(self.naxis):
            lam = self.lines[i]*(1.0+self.prop._zem)
            velo = 299792.458*(self.prop._wave-lam)/lam
            self.modelcrvs[i] = self.axs[i].plot(velo, model, 'r-')
        return

    def draw_callback(self, event):
        for i in range(self.naxis):
            trans = mtransforms.blended_transform_factory(self.axs[i].transData, self.axs[i].transAxes)
            self.backgrounds[i] = self.canvas.copy_from_bbox(self.axs[i].bbox)
            lam = self.lines[i]*(1.0+self.prop._zem)
            velo = 299792.458*(self.prop._wave-lam)/lam
            if self.fitreg[i] is not None:
                self.fitreg[i].remove()
            self.fitreg[i] = self.axs[i].fill_between(velo, 0, 1, where=self.prop._regions==1, facecolor='green', alpha=0.5, transform=trans)
            if self.curreg[i] is not None:
                self.curreg[i].remove()
                self.curreg[i] = self.axs[i].fill_between(velo, 0, 1, where=self.actor==1, facecolor='red', alpha=0.5, transform=trans)
            self.axs[i].set_yscale("linear")
            self.axs[i].draw_artist(self.specs[i])
        for i in range(self.naxis):
            self.axs[i].set_yscale("linear")
        self.draw_lines()
        self.draw_model()

    def get_ind_under_point(self, event):
        """
        Get the index of the spectrum closest to the cursor
        """
        for i in range(self.naxis):
            if event.inaxes != self.axs[i]:
                continue
            lam = self.lines[i]*(1.0+self.prop._zem)
            velo = 299792.458*(self.prop._wave-lam)/lam
        ind = np.argmin(np.abs(velo-event.xdata))
        return ind

    def sort_lines(self, method="ion"):
        """
        Sort lines by decreasing probability of detection
        """
        if method == "sig":
            coldens = 10.0**(self.atom.solar-12.0)
            ew = coldens * (self.atom._atom_wvl**2 * self.atom._atom_fvl)
            #snr = 1.0/self.prop._flue
            #indices = np.abs(np.subtract.outer(self.prop._wave, self.atom._atom_wvl*(1.0+self.prop._zem))).argmin(0)
            sigdet = ew#*snr[indices]
            self.sortlines = np.argsort(sigdet)[::-1]
        elif method == "ion":
            self.sortlines = np.arange(self.atom._atom_wvl.size)
        elif method == "wave":
            self.sortlines = np.argsort(self.atom._atom_wvl)
        return

    def get_current_line(self):
        if self.lineidx < 0:
            self.lineidx += self.sortlines.size
        if self.lineidx >= self.sortlines.size:
            self.lineidx -= self.sortlines.size
        self.linecur = self.sortlines[self.lineidx]

    def button_press_callback(self, event):
        """
        whenever a mouse button is pressed
        """
        if event.inaxes is None:
            return
        if self.canvas.toolbar.mode != "":
            return
        if event.button == 1:
            self._addsub = 1
        elif event.button == 3:
            self._addsub = 0
        self._start = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        """
        whenever a mouse button is released
        """
        if event.inaxes is None:
            return
        if self.canvas.toolbar.mode != "":
            return
        self._end = self.get_ind_under_point(event)
        if self._end != self._start:
            if self._start > self._end:
                tmp = self._start
                self._start = self._end
                self._end = tmp
            self.actor *= 0
            self.actor[self._start:self._end] = 1
        # Add this to the master fitted regions
        self.prop.add_region(self.actor)
        for i in range(self.naxis):
            if self.curreg[i] is not None:
                self.curreg[i].remove()
                self.curreg[i] = None
            if event.inaxes != self.axs[i]:
                continue
            # Set the axis being used
            self.axisidx = i
            # Plot the selected region
            trans = mtransforms.blended_transform_factory(self.axs[i].transData, self.axs[i].transAxes)
            self.canvas.restore_region(self.backgrounds[i])
            lam = self.lines[i]*(1.0+self.prop._zem)
            velo = 299792.458*(self.prop._wave-lam)/lam
            #self.fb = self.ax.fill_between(velo, 0, 1, where=self.prop._regions==1, facecolor='green', alpha=0.5, transform=trans)
            self.curreg[i] = self.axs[i].fill_between(velo, 0, 1, where=self.actor==1, facecolor='red', alpha=0.5, transform=trans)
        self.canvas.draw()

    def key_press_callback(self, event):
        """
        whenever a key is pressed
        """
        if not event.inaxes:
            return
        # Used keys include:  acdfklmprquw?[]<>
        if event.key == '?':
            print("============================================================")
            print("       MAIN OPERATIONS")
            print("f       : toggle fullscreen")
            print("p       : toggle pan/zoom with the cursor")
            print("w       : write the model and a spectrum with the associated fitting regions")
            print("q       : exit")
            print("------------------------------------------------------------")
            print("       FITTING COMMANDS")
            print("k       : Add a line at the specified location (line ID will depend on panel)")
            print("l       : Add a Lya line at the specified location (line ID will always be Lya - regardless of panel)")
            print("m       : Add a metal line to the cursor")
            print("d       : Delete the nearest line to the cursor")
            print("a       : Fit the current regions in all panels with ALIS")
            print("c       : Clear current fitting (start over)")
            print("u       : Update (merge) current fitting into master model and regions")
            print("------------------------------------------------------------")
            print("       INTERACTION COMMANDS")
            print("[ / ]   : pan left and right")
            print("< / >   : zoom in and out")
            print("r       : toggle residuals plotting (i.e. remove master model from data)")
            print("------------------------------------------------------------")
#            print("       ATOMIC DATA OF THE CURRENT LINE")
#            print("{0:s} {1:s}  {2:f}".format(self.atom._atom_atm[self.linecur].strip(),self.atom._atom_ion[self.linecur].strip(),self.atom._atom_wvl[self.linecur]))
#            print("Observed wavelength = {0:f}".format(self.atom._atom_wvl[self.linecur]*(1.0+self.prop._zem)))
#            print("f-value = {0:f}".format(self.atom._atom_fvl[self.linecur]))
#            print("------------------------------------------------------------")
        elif event.key == 'a':
            pass
        elif event.key == 'c':
            pass
        elif event.key == 'd':
            self.delete_line()
        # Don't need to explicitly put f in there
        elif event.key == 'k':
            pass
        elif event.key == 'l':
            self.fit_line()
        elif event.key == 'm':
            pass
        # Don't need to explicitly put p in there
        elif event.key == 'q':
            if self._changes:
                print("WARNING: There are unsaved changes!!")
                print("Press q again to exit")
                self._changes = False
            else:
                sys.exit()
        elif event.key == 'r':
            pass
        elif event.key == 'u':
            self.update_master()
        elif event.key == 'w':
                self.write_data()
#        elif event.key == ']':
#            self.next_element(1)
#            self.next_spectrum()
#        elif event.key == '[':
#            self.next_element(-1)
#            self.next_spectrum()
        self.canvas.draw()

    def key_release_callback(self, event):
        """
        whenever a key is released
        """
        if not event.inaxes:
            return

    def delete_line(self):
        if self.modelpars is None:
            return
        zclose = self.prop._wave[self._end]/self.lines[self.axisidx] - 1.0
        zarg = np.argmin(np.abs(self.modelpars[:,1]-zclose))
        self.modelpars = np.delete(self.modelpars, (zarg), axis=0)
        if self.modelpars.shape[0] == 0:
            self.modelpars = None
        self.draw_lines()
        self.draw_model()

    def fit_line(self):
        w = np.where(self.actor == 1)
        if w[0].size <= 3:
            print("WARNING : not enough pixels to fit a single line - at least 3 pixels are needed")
            return
        # Pick some starting parameters
        coldens0 = 14.0
        zabs0 = np.mean(self.prop._wave[w])/self.lines[self.axisidx] - 1.0
        bval0 = 10.0
        atidx = np.argmin(np.abs(self.lines[self.axisidx]-self.atom._atom_wvl))
        wv = self.lines[self.axisidx]
        fv = self.atom._atom_fvl[atidx]
        gm = self.atom._atom_gam[atidx]
        # Perform the fit
        popt, pcov = curve_fit(lambda x, ng, zg, bg : voigt(x, ng, zg, bg, wv, fv, gm), self.prop._wave[w], self.prop._flux[w], sigma=self.prop._flue[w], method='lm', p0=[coldens0,zabs0,bval0])
        #popt, pcov = curve_fit(lambda x, ng, bg : voigt(x, ng, zg, bg), wavfit[lnww[i]-nnpf:lnww[i]-nnpf+npixfit+1], fxpix, sigma=fepix, method='lm', p0=[ng,bg])
        # Check if any of the parameters have gone "out of bounds"
        addarr = np.array([popt[0], popt[1], popt[2]]).reshape((1, 3))
        if self.modelpars is None:
            self.modelpars = addarr.copy()
        else:
            self.modelpars = np.append(self.modelpars, addarr, axis=0)

    def next_element(self, pm, ion=False):
        if ion == True:
            arrsrt = np.core.defchararray.add(self.atom._atom_atm, self.atom._atom_ion)
        else:
            arrsrt = self.atom._atom_atm
        unq, idx = np.unique(arrsrt, return_index=True)
        unq = unq[idx.argsort()]
        nxt = np.where(unq == arrsrt[self.linecur])[0][0]+pm
        if nxt >= unq.size:
            nxt = 0
        ww = np.where(arrsrt == unq[nxt])[0]
        self.lineidx = ww[0]
        return

    def next_spectrum(self):
        self.get_current_line()
        # Update the wavelength range of the spectrum being plot
        self.update_waverange()
        # See if any regions need to be loaded
        self.prop._regions[:] = 0
        idtxt = "{0:s}_{1:s}_{2:.1f}".format(self.atom._atom_atm[self.linecur].strip(),self.atom._atom_ion[self.linecur].strip(),self.atom._atom_wvl[self.linecur])
        tstnm = self.prop._outp + "_" + idtxt + "_reg.dat"
        if os.path.exists(tstnm):
            wv, reg = np.loadtxt(tstnm, unpack=True, usecols=(0,3))
            mtch = np.in1d(self.prop._wave, wv, assume_unique=True)
            self.prop._regions[np.where(mtch)] = reg.copy()
            self.ax.set_xlim([np.min(wv), np.max(wv)])
        # Other stuff
        self.canvas.draw()
        return

    def update_model(self):
        return

    def update_regions(self):
        self.prop._regions[self._start:self._end] = self._addsub

    def update_waverange(self):
        for i in range(len(self.lines)):
            #wcen = self.lines[i]*(1.0+self.prop._zem)
            #xmn = wcen * (1.0 - self.veld/299792.458)
            #xmx = wcen * (1.0 + self.veld/299792.458)
            xmn, xmx = -self.veld, self.veld
            self.axs[i].set_xlim([xmn, xmx])
            self.axs[i].set_ylim([-0.1, 1.1])
        self.canvas.draw()

    def write_data(self):
        for i in range(self.naxis):
            # Plot the lines
            lam = self.lines[i]*(1.0+self.prop._zem)
            velo = 299792.458*(self.prop._wave-lam)/lam
            xmn, xmx = self.axs[i].get_xlim()
            wsv = np.where((velo>xmn) & (velo<xmx))
            idtxt = "H_I_{0:.1f}".format(self.lines[i])
            outnm = self.prop._outp + "_" + idtxt + "_reg.dat"
            np.savetxt(outnm, np.transpose((self.prop._wave[wsv], self.prop._flux[wsv]*self.prop._cont[wsv], self.prop._flue[wsv]*self.prop._cont[wsv], self.prop._regions[wsv])))
            print("Saved file:")
            print(outnm)
        # Save the ALIS model file
        modwrite(self.modelpars)
        return

class props:
    def __init__(self, qso):
        # Load the data
        ifil = qso._path + qso._filename
        outf = qso._path + qso._filename.replace(".dat", "_reg.dat")
        try:
            wave, flux, flue, regions = np.loadtxt(outf, unpack=True, usecols=(0,1,2,3))
            cont = np.ones(wave.size)
            print("Loaded file:")
            print(outf)
        except:
            try:
                wave, flux, flue, cont = np.loadtxt(ifil, unpack=True, usecols=(0,1,2,4))
                regions = np.zeros(wave.size)
                print("Loaded file:")
                print(ifil)
            except:
                wave, flux, flue = np.loadtxt(ifil, unpack=True, usecols=(0,1,2))
                cont = np.ones(wave.size)
                regions = np.zeros(wave.size)
        self._wave = wave
        self._flux = flux
        self._flue = flue
        self._cont = cont
        self._file = ifil
        self._regions = regions
        self._outp = qso._path + qso._filename.replace(".dat", "")
        self._outf = outf
        self._zem = qso._zem

    def set_regions(self, arr):
        self._regions = arr.copy()
        return

    def add_region(self, arr):
        self._regions += arr.copy()
        np.clip(self._regions, 0, 1)
        return

class atomic:
    def __init__(self, filename="/Users/rcooke/Software/ALIS_dataprep/atomic.dat", wmin=None, wmax=None):
        self._wmin = wmin
        self._wmax = wmax
        self._atom_atm=[]
        self._atom_ion=[]
        self._atom_lbl=[]
        self._atom_wvl=[]
        self._atom_fvl=[]
        self._atom_gam=[]
        self._molecule_atm=[]
        self._molecule_ion=[]
        self._molecule_lbl=[]
        self._molecule_wvl=[]
        self._molecule_fvl=[]
        self._molecule_gam=[]
        self.load_lines(filename)

    def solar(self):
        elem = np.array(['H ', 'He','Li','Be','B ', 'C ', 'N ', 'O ', 'F ', 'Ne','Na','Mg','Al','Si','P ', 'S ', 'Cl','Ar','K ', 'Ca','Sc','Ti','V ', 'Cr','Mn','Fe','Co','Ni','Cu','Zn'])
        mass = np.array([1.0, 4.0, 7.0, 8.0, 11.0, 12.0,14.0,16.0,19.0,20.0,23.0,24.0,27.0,28.0,31.0,32.0,35.0,36.0,39.0,40.0,45.0,48.0,51.0,52.0,55.0,56.0,59.0,60.0,63.0,64.0])
        solr = np.array([12.0,10.93,3.26,1.30,2.79,8.43,7.83,8.69,4.42,7.93,6.26,7.56,6.44,7.51,5.42,7.14,5.23,6.40,5.06,6.29,3.05,4.91,3.95,5.64,5.48,7.47,4.93,6.21,4.25,4.63])
        solar = np.zeros(self._atom_atm.size)
        for i in range(elem.size):
            w = np.where(self._atom_atm==elem[i])
            solar[w] = solr[i]
        self.solar = solar
        return

    def load_lines(self, filename):
        # Load the lines file
        print("Loading a list of atomic transitions...")
        try:
            infile = open(filename, "r")
        except IOError:
            print("The atomic data file:\n" +filename+"\ndoes not exist!")
            sys.exit()
        atom_list = infile.readlines()
        leninfile = len(atom_list)
        infile.close()
        infile = open(filename, "r")
        for i in range(0, leninfile):
            nam = infile.read(2)
            if nam.strip() == "#":
                null = infile.readline()
                continue
            self._atom_atm.append(nam)
            self._atom_ion.append(infile.read(4))
            self._atom_lbl.append((self._atom_atm[-1]+self._atom_ion[-1]).strip())
#            self._atom_lbl[i].strip()
            line2 = infile.readline()
            wfg = line2.split()
            self._atom_wvl.append(eval(wfg[0]))
            self._atom_fvl.append(eval(wfg[1]))
            self._atom_gam.append(eval(wfg[2]))

        # Convert to numpy array
        self._atom_atm = np.array(self._atom_atm)
        self._atom_ion = np.array(self._atom_ion)
        self._atom_lbl = np.array(self._atom_lbl)
        self._atom_wvl = np.array(self._atom_wvl)
        self._atom_fvl = np.array(self._atom_fvl)
        self._atom_gam = np.array(self._atom_gam)

        # Ignore lines outside of the specified wavelength range
        if self._wmin is not None:
            ww = np.where(self._atom_wvl > self._wmin)
            self._atom_atm = self._atom_atm[ww]
            self._atom_ion = self._atom_ion[ww]
            self._atom_lbl = self._atom_lbl[ww]
            self._atom_wvl = self._atom_wvl[ww]
            self._atom_fvl = self._atom_fvl[ww]
            self._atom_gam = self._atom_gam[ww]
        if self._wmax is not None:
            ww = np.where(self._atom_wvl < self._wmax)
            self._atom_atm = self._atom_atm[ww]
            self._atom_ion = self._atom_ion[ww]
            self._atom_lbl = self._atom_lbl[ww]
            self._atom_wvl = self._atom_wvl[ww]
            self._atom_fvl = self._atom_fvl[ww]
            self._atom_gam = self._atom_gam[ww]

        # Ignore some lines
        ign = np.where((self._atom_ion!="I*  ")&(self._atom_ion!="II* ")&(self._atom_ion!="I** ")&(self._atom_ion!="II**")&(self._atom_wvl>=914.0))
        self._atom_atm = self._atom_atm[ign]
        self._atom_ion = self._atom_ion[ign]
        self._atom_lbl = self._atom_lbl[ign]
        self._atom_wvl = self._atom_wvl[ign]
        self._atom_fvl = self._atom_fvl[ign]
        self._atom_gam = self._atom_gam[ign]

        # Assign solar abundances to these lines
        self.solar()

        # Load the lines file
        print("Loading a list of molecular transitions...")
        try:
            infile = open("/Users/rcooke/Software/ALIS_dataprep/molecule.dat", "r")
        except IOError:
            print("The lines file:\n" + "molecule.dat\ndoes not exist!")
            sys.exit()
        molecule_list=infile.readlines()
        leninfile=len(molecule_list)
        infile.close()
        infile = open("/Users/rcooke/Software/ALIS_dataprep/molecule.dat", "r")
        for i in range(0, leninfile):
            self._molecule_atm.append(infile.read(2))
            self._molecule_ion.append(infile.read(4))
            self._molecule_lbl.append((self._molecule_atm[i]+self._molecule_ion[i]).strip())
            line2 = infile.readline()
            wfg = line2.split()
            self._molecule_wvl.append(eval(wfg[0]))
            self._molecule_fvl.append(eval(wfg[1]))
            self._molecule_gam.append(eval(wfg[2]))

        # Convert to numpy array
        self._molecule_atm = np.array(self._molecule_atm)
        self._molecule_ion = np.array(self._molecule_ion)
        self._molecule_lbl = np.array(self._molecule_lbl)
        self._molecule_wvl = np.array(self._molecule_wvl)
        self._molecule_fvl = np.array(self._molecule_fvl)
        self._molecule_gam = np.array(self._molecule_gam)

        # Now sort lines data according to wavelength
        argsrt = np.argsort(self._molecule_wvl)
        self._molecule_atm = self._molecule_atm[argsrt]
        self._molecule_ion = self._molecule_ion[argsrt]
        self._molecule_lbl = self._molecule_lbl[argsrt]
        self._molecule_wvl = self._molecule_wvl[argsrt]
        self._molecule_fvl = self._molecule_fvl[argsrt]
        self._molecule_gam = self._molecule_gam[argsrt]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    prop = props(QSO("HS1700p6416"))

    # Ignore lines outside of wavelength range
    wmin, wmax = np.min(prop._wave)/(1.0+prop._zem), np.max(prop._wave)/(1.0+prop._zem)

    # Load atomic data
    atom = atomic(wmin=wmin, wmax=wmax)

    spec = Line2D(prop._wave, prop._flux, linewidth=1, linestyle='solid', color='k', drawstyle='steps', animated=True)

    fig, ax = plt.subplots(figsize=(16,9), facecolor="white")
    ax.add_line(spec)
    reg = SelectRegions(fig.canvas, ax, spec, prop, atom)

    ax.set_title("Press '?' to list the available options")
    #ax.set_xlim((prop._wave.min(), prop._wave.max()))
    ax.set_ylim((0.0, prop._flux.max()))
    plt.show()
