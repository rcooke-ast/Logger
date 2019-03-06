import os
import sys
from shutil import rmtree
import datetime
import numpy as np
import matplotlib
import pickle
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
from quasars import QSO
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.special import wofz
from alis.alis import alis
from alis.alsave import save_model as get_alis_string
from copy import deepcopy
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


def convolve_spec(wave, flux, vfwhm):
    """
    Define the functional form of the model
    --------------------------------------------------------
    wave  : array of wavelengths
    flux  : model flux array
    vfwhm  : array of parameters for this model
    --------------------------------------------------------
    """
    sigd = vfwhm / (2.99792458E5 * (2.0*np.sqrt(2.0*np.log(2.0))))
    ysize = flux.size
    fsigd = 6.0*sigd
    dwav = 0.5 * (wave[2:] - wave[:-2]) / wave[1:-1]
    dwav = np.append(np.append(dwav[0], dwav), dwav[-1])
    df = int(np.min([np.int(np.ceil(fsigd/dwav).max()), ysize/2 - 1]))
    yval = np.zeros(2*df+1)
    yval[df:2*df+1] = (wave[df:2 * df + 1] / wave[df] - 1.0) / sigd
    yval[:df] = (wave[:df] / wave[df] - 1.0) / sigd
    gaus = np.exp(-0.5*yval*yval)
    size = ysize + gaus.size - 1
    fsize = 2 ** np.int(np.ceil(np.log2(size)))  # Use this size for a more efficient computation
    conv = np.fft.fft(flux, fsize)
    conv *= np.fft.fft(gaus/gaus.sum(), fsize)
    ret = np.fft.ifft(conv).real.copy()
    del conv
    return ret[df:df+ysize]


def newstart(covar, num):
    """
    Using a covariance matrix, calculate num realisations of the model parameters
    """
    # Find the non-zero elements in the covariance matrix
    cvsize = covar.shape[0]
    cxzero, cyzero = np.where(covar == 0.0)
    bxzero, byzero = np.bincount(cxzero), np.bincount(cyzero)
    wxzero, wyzero = np.where(bxzero == cvsize)[0], np.where(byzero == cvsize)[0]
    zrocol = np.intersect1d(wxzero, wyzero)  # This is the list of columns (or rows), where all elements are zero
    # Create a mask for the non-zero elements
    mask = np.zeros_like(covar)
    mask[:, zrocol], mask[zrocol, :] = 1, 1
    cvnz = np.zeros((cvsize-zrocol.size, cvsize-zrocol.size))
    cvnz[np.where(cvnz == 0.0)] = covar[np.where(mask == 0.0)]
    # Generate a new set of starting parameters from the covariance matrix
    x_covar_fit = np.matrix(np.random.standard_normal((cvnz.shape[0], num)))
    c_covar_fit = np.matrix(cvnz)
    u_covar_fit = np.linalg.cholesky(c_covar_fit)
    return u_covar_fit * x_covar_fit


class SelectRegions(object):
    """
    Generate a model and regions to be fit with ALIS
    """

    def __init__(self, canvas, axs, axi, specs, prop, atom, vel=500.0, lines=None, resid=None):
        """
        axs : ndarray
          array of all data axes
        axi : Axes instance
          axis used to display information
        vel : float
          Default +/- plotting window in km/s
        resloc : list (two element)
          data location of the centre and 1 sigma value of the plotted residuals. Note: +/-2 sigma residuals are plotted
        """
        if resid is None:
            resid = [1.2, 0.05]

        self.axs = axs
        self.axi = axi
        self.naxis = len(axs)
        self.specs = specs
        self.prop = prop
        self.atom = atom
        self.veld = vel
        self.resloc = resid
        self._xmnsv, self._xmxsv = [], []
        self._zqso = self.prop._zem     # The plotted redshift at the centre of each panel
        self.curreg = [None for ii in range(self.naxis)]
        self.fitreg = [None for ii in range(self.naxis)]
        self.backgrounds = [None for ii in range(self.naxis)]
        self.mouseidx = 0   # Index of wavelength array where mouse is located
        self.mmx = 0  # x position of mouse
        self.mmy = 0  # y position of mouse
        self._addsub = 0    # Adding a region (1) or removing (0)
        self._start = 0     # Start of a region
        self._end = 0       # End of a region
        self._resid = False  # Are the residuals currently being plotted?
        self._qconf = False  # Confirm quit message
        self._fitchng = True  # Check that a fit with ALIS has been performed before merging to master
        self._respreq = [False, None]  # Does the user need to provide a response before any other operation will be permitted? Once the user responds, the second element of this array provides the action to be performed.
        self.annlines = []
        self.anntexts = []
        self._actions = []   # Save the state of the 'self' dictionary at various points.
        self._nstore = 5     # How many previous operations can be stored (and restored)?

        if lines is None:
            lines = 1215.6701*np.ones(self.naxis)
        self.lines = lines

        # Create the model lines variables
        self.modelLines_act = [None for ii in range(self.naxis)]  # The Line instance of the plotted actors model
        self.modelLines_mst = [None for ii in range(self.naxis)]  # The Line instance of the plotted master model
        self.modelLines_upd = [None for ii in range(self.naxis)]  # The Line instance of the plotted model that is being updated
        self.modelCont_upd = [None for ii in range(self.naxis)]  # The Line instance of the plotted continuum that is being updated
        self.lines_act = AbsorptionLines()  # Stores all of the information about the actor absorption lines.
        self.lines_mst = AbsorptionLines()  # Stores all of the information about the master absorption lines.
        self.lines_upd = AbsorptionLines()  # Stores all of the information about the updated absorption line.
        self.actors = [np.zeros(self.prop._wave.size) for ii in range(self.naxis)]
        self.lactor = np.zeros(self.prop._wave.size)  # Just the previously selected actor
        self.model_act = np.ones(self.prop._wave.size)  # Model of the actors
        self.model_mst = np.ones(self.prop._wave.size)  # Model of the master spectrum
        self.model_cnv = np.ones(self.prop._wave.size)  # Model of the master spectrum, convolved with the line spread function
        self.model_upd = np.ones(self.prop._wave.size)  # Model of the absorption lines for the spectrum currently being updated
        self.model_cnt = np.ones(self.prop._wave.size)  # Model of the continuum fitted with ALIS.
        self._update_model = None

        # Unset some of the matplotlib keymaps
        matplotlib.pyplot.rcParams['keymap.fullscreen'] = ''        # toggling fullscreen (Default: f, ctrl+f)
        matplotlib.pyplot.rcParams['keymap.home'] = ''              # home or reset mnemonic (Default: h, r, home)
        matplotlib.pyplot.rcParams['keymap.back'] = ''              # forward / backward keys to enable (Default: left, c, backspace)
        matplotlib.pyplot.rcParams['keymap.forward'] = ''           # left handed quick navigation (Default: right, v)
        #matplotlib.pyplot.rcParams['keymap.pan'] = ''              # pan mnemonic (Default: p)
        matplotlib.pyplot.rcParams['keymap.zoom'] = ''              # zoom mnemonic (Default: o)
        matplotlib.pyplot.rcParams['keymap.save'] = ''              # saving current figure (Default: s)
        matplotlib.pyplot.rcParams['keymap.quit'] = ''              # close the current figure (Default: ctrl+w, cmd+w)
        matplotlib.pyplot.rcParams['keymap.grid'] = ''              # switching on/off a grid in current axes (Default: g)
        matplotlib.pyplot.rcParams['keymap.yscale'] = ''            # toggle scaling of y-axes ('log'/'linear') (Default: l)
        matplotlib.pyplot.rcParams['keymap.xscale'] = ''            # toggle scaling of x-axes ('log'/'linear') (Default: L, k)
        matplotlib.pyplot.rcParams['keymap.all_axes'] = ''          # enable all axes (Default: a)

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.mouse_move_callback)
        self.canvas = canvas

        # Update the wavelength range of the spectrum being plot
        self.update_waverange()

        # Just before the canvas is drawn, generate the autosave preamble
        # to save the properties of this execution of the code
        self.autoload()
        # Draw the canvas
        self.draw_lines()
        self.draw_model()
        self.canvas.draw()

    def draw_lines(self):
        """
        Draw labels on the absorption lines for the input redshift
        """
#        if self.lines_act.size == 0:
            # There are no model lines
#            return
        for i in self.annlines: i.remove()
        for i in self.anntexts: i.remove()
        self.annlines = []
        self.anntexts = []
        for i in range(self.naxis):
            for j in range(self.lines_act.size):
                for k in range(len(self.lines_act.fitidx[j])):
                    # Determine the rest wavelength of the line
                    wave0 = float(self.lines_act.label[j][k].split("-")[-1])
                    lam = self.lines[i] * (1.0 + self._zqso)
                    # Determine the redshift
                    zabs = self.lines_act.redshift[j]
                    if self.lines_act.shifts[j][k] != -1.0:
                        zabs = self.lines_act.shifts[j][k]
                    # Determine the color of the line
                    col = 'r'
                    if self.lines_act.fitidx[j][k] != -1:
                        col = 'b'
                    velo = 299792.458 * (wave0 * (1.0 + zabs) - lam) / lam
                    self.annlines.append(self.axs[i].axvline(velo, color=col))
        return

        # annotations = [child for child in self.ax.get_children() if isinstance(child, matplotlib.text.Annotation)]
        # for i in self.annlines: i.remove()
        # for i in self.anntexts: i.remove()
        # self.annlines = []
        # self.anntexts = []
        # for ax in self.axs:
        #     # Plot the lines
        #     xmn, xmx = ax.get_xlim()
        #     ymn, ymx = ax.get_ylim()
        #     xmn /= (1.0+self.prop._zem)
        #     xmx /= (1.0+self.prop._zem)
        #     w = np.where((self.atom._atom_wvl > xmn) & (self.atom._atom_wvl < xmx))[0]
        #     for i in range(w.size):
        #         dif = i%5
        #         self.annlines.append(ax.axvline(self.atom._atom_wvl[w[i]]*(1.0+self.prop._zem), color='b'))
        #         txt = "{0:s} {1:s} {2:.1f}".format(self.atom._atom_atm[w[i]],self.atom._atom_ion[w[i]],self.atom._atom_wvl[w[i]])
        #         ylbl = ymn + (ymx-ymn)*(dif+1.5)/8.0
        #         self.anntexts.append(ax.annotate(txt, (self.atom._atom_wvl[w[i]]*(1.0+self.prop._zem), ylbl), rotation=90.0, color='b', ha='center', va='bottom'))
        # return

    def draw_model(self):
        # if self.lines_act.size == 0:
        #     # There are no model lines
        #     return
        for i in self.modelLines_act:
            if i is not None:
                i.pop(0).remove()
        for i in self.modelLines_mst:
            if i is not None:
                i.pop(0).remove()
        # Generate the model curve for the actors
        self.model_act = np.ones(self.prop._wave.size)
        for i in range(self.lines_act.size):
            for j in range(len(self.lines_act.shifts[i])):
                zabs = self.lines_act.redshift[i]
                wave0 = float(self.lines_act.label[i][j].split("-")[-1])
                if self.lines_act.shifts[i][j] != -1.0:
                    zabs = self.lines_act.shifts[i][j]
                p0, p1, p2 = self.lines_act.coldens[i], zabs, self.lines_act.bval[i]
                atidx = np.argmin(np.abs(wave0 - self.atom._atom_wvl))
                wv = self.atom._atom_wvl[atidx]
                fv = self.atom._atom_fvl[atidx]
                gm = self.atom._atom_gam[atidx]
                self.model_act *= voigt(self.prop._wave, p0, p1, p2, wv, fv, gm)
        # Plot the models
        for i in range(self.naxis):
            lam = self.lines[i]*(1.0 + self._zqso)
            velo = 299792.458*(self.prop._wave-lam)/lam
            self.modelLines_mst[i] = self.axs[i].plot(velo, self.model_cnv, 'b-', linewidth=2.0)
            self.modelLines_act[i] = self.axs[i].plot(velo, self.model_act, 'r-', linewidth=1.0)
        return

    def draw_model_update(self):
        for i in range(self.naxis):
            if self.modelLines_upd[i] is not None:
                self.modelLines_upd[i].pop(0).remove()
                self.modelLines_upd[i] = None
                self.modelCont_upd[i].pop(0).remove()
                self.modelCont_upd[i] = None
        # Generate the model curve for the actors
        self.model_upd = self.model_cnt.copy()
        # If not currently updating, return
        if self._update_model is None:
            self.canvas.draw()
            return
        # Calculate the model
        for i in range(self.lines_upd.size):
            for j in range(len(self.lines_upd.shifts[i])):
                zabs = self.lines_upd.redshift[i]
                wave0 = float(self.lines_upd.label[i][j].split("-")[-1])
                if self.lines_upd.shifts[i][j] != -1.0:
                    zabs = self.lines_upd.shifts[i][j]
                p0, p1, p2 = self.lines_upd.coldens[i], zabs, self.lines_upd.bval[i]
                atidx = np.argmin(np.abs(wave0 - self.atom._atom_wvl))
                wv = self.atom._atom_wvl[atidx]
                fv = self.atom._atom_fvl[atidx]
                gm = self.atom._atom_gam[atidx]
                self.model_upd *= voigt(self.prop._wave, p0, p1, p2, wv, fv, gm)
        # Plot the models
        for i in range(self.naxis):
            lam = self.lines[i] * (1.0 + self._zqso)
            velo = 299792.458 * (self.prop._wave - lam) / lam
            self.modelCont_upd[i] = self.axs[i].plot(velo, self.model_cnt, 'g-', linewidth=2.0, alpha=0.5)
            self.modelLines_upd[i] = self.axs[i].plot(velo, self.model_upd, 'g-', linewidth=1.0)
        self.canvas.draw()
        return

    def draw_callback(self, event):
        for i in range(self.naxis):
            trans = mtransforms.blended_transform_factory(self.axs[i].transData, self.axs[i].transAxes)
            self.backgrounds[i] = self.canvas.copy_from_bbox(self.axs[i].bbox)
            lam = self.lines[i]*(1.0 + self._zqso)
            velo = 299792.458*(self.prop._wave-lam)/lam
            if self.fitreg[i] is not None:
                self.fitreg[i].remove()
            self.fitreg[i] = self.axs[i].fill_between(velo, 0, 1, where=self.prop._regions == 1, facecolor='green', alpha=0.5, transform=trans)
            if self.curreg[i] is not None:
                self.curreg[i].remove()
                self.curreg[i] = self.axs[i].fill_between(velo, 0, 1, where=self.actors[i] == 1, facecolor='red', alpha=0.5, transform=trans)
            self.axs[i].set_yscale("linear")
            self.axs[i].draw_artist(self.specs[i])
        for i in range(self.naxis):
            self.axs[i].set_yscale("linear")

    def store_actions(self):
        # We don't want to save more than 5 previous operations
        dcopy = self.__dict__.copy()
        if len(self._actions) >= self._nstore:
            del self._actions[0]
        self._actions += [dcopy]
        return

    def undo(self):
        self.__dict__.update(self._actions[-1])
        self.canvas.draw()
        return

    def mouse_move_callback(self, event):
        """
        Get the index of the spectrum closest to the cursor
        """
        if event.inaxes is None:
            return
        axisID = self.get_axisID(event)
        if axisID is not None:
            if axisID < self.naxis:
                if self._update_model is not None:
                    if self._update_model[0] not in ['a', 'z']:
                        # Not manually updating absorption line, but updating something else
                        self.mmx, self.mmy = event.xdata, event.ydata
                        self.mouseidx = self.get_ind_under_point(axisID, event.xdata)
                        return
                    lidx, widx = self._update_model[1], self._update_model[2]
                    mouseidx = self.get_ind_under_point(axisID, event.xdata)
                    if self._update_model[0] == 'a':
                        lam = self.lines[lidx] * (1.0 + self.lines_act.redshift[widx])
                        newcold = self.lines_act.coldens[widx] + (self.mmy - event.ydata)
                        newbval = max(1.0, abs(299792.458 * (self.prop._wave[mouseidx] - lam) / lam))
                        newzabs = self.lines_act.redshift[widx]
                    elif self._update_model[0] == 'z':
                        newcold = self.lines_act.coldens[widx]
                        newbval = self.lines_act.bval[widx]
                        newzabs = self.prop._wave[mouseidx] / self.lines[lidx] - 1.0
                    self.lines_upd.update_absline(0, coldens=newcold, bval=newbval, redshift=newzabs)
                    self.draw_model_update()
                else:
                    self.mmx, self.mmy = event.xdata, event.ydata
                    self.mouseidx = self.get_ind_under_point(axisID, event.xdata)

    def get_ind_under_point(self, axisID, xdata):
        """
        Get the index of the spectrum closest to the event point (could be a mouse move or key press, or button press)
        """
        lam = self.lines[axisID]*(1.0 + self._zqso)
        velo = 299792.458*(self.prop._wave-lam)/lam
        ind = np.argmin(np.abs(velo-xdata))
        return ind

    def get_axisID(self, event):
        for ii in range(self.naxis):
            if event.inaxes == self.axs[ii]:
                return ii
        if event.inaxes == self.axi:
            return self.naxis
        return None

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
        axisID = self.get_axisID(event)
        # Check that the cursor is in one of the data panels
        if axisID is not None:
            if axisID < self.naxis:
                self._start = self.get_ind_under_point(axisID, event.xdata)

    def button_release_callback(self, event):
        """
        whenever a mouse button is released
        """
        if event.inaxes is None:
            return
        if event.inaxes == self.axi:
            if event.xdata > 0.8 and event.xdata < 0.9:
                answer = "y"
            elif event.xdata >= 0.9:
                answer = "n"
            else:
                return
            self.operations(answer, -1, -1)
            self.update_infobox(default=True)
            return
        elif self._respreq[0]:
            # The user is trying to do something before they have responded to a question
            return
        if self.canvas.toolbar.mode != "":
            return
        # Draw an actor
        axisID = self.get_axisID(event)
        # Check that the cursor is in one of the data panels
        if axisID is not None:
            if axisID < self.naxis:
                self._end = self.get_ind_under_point(axisID, event.xdata)
                if self._end == self._start:
                    # The mouse button was pressed (not dragged)
                    if self.lines_act.size != 0:
                        self.operations('uli', axisID, self.mouseidx, params=[self._start, self._addsub])
                else:
                    # The mouse button was dragged
                    # Now update the actors
                    self.operations('ua', axisID, self.mouseidx, params=[self._start, self._end, self._addsub])
                    self.plot_actor(axisID)

    def key_press_callback(self, event):
        """
        whenever a key is pressed
        """
        # Check that the event is in an axis...
        if not event.inaxes:
            return
        # ... but not the information box!
        if event.inaxes == self.axi:
            return
        axisID = self.get_axisID(event)
        self.operations(event.key, axisID, self.mouseidx, autosave=True)
        self.canvas.draw()

    def operations(self, key, axisID, mouseidx, params=None, autosave=True):

        # Check if the user really wants to quit
        if key == 'q' and self._qconf:
            sys.exit()
        elif self._qconf:
            self.update_infobox(default=True)
            self._qconf = False

        # Manage responses from questions posed to the user.
        if self._respreq[0] and key not in ['r', 'g', 'b']:
            # When a response is required, we still want to:
            #   - toggle residuals
            #   - go to/back a given line
            if key != "y" and key != "n":
                return
            else:
                # Switch off the required response
                self._respreq[0] = False
                # Deal with the response
                if self._respreq[1] == "fit_alis":
                    self.merge_alis(key)
                    if autosave: self.autosave(key, axisID, mouseidx)
                elif self._respreq[1] == "mst_merge":
                    if key == "y":
                        self.update_master()
                        self._fitchng = True
                    if autosave: self.autosave(key, axisID, mouseidx)
                elif self._respreq[1] == "write":
                    # First remove the old file, and save the new one
                    os.remove(self.lines_mst.masterLines_name(self.prop))
                    self.write_data()
                else:
                    return
            # Reset the info box
            self.update_infobox(default=True)
            return

        # Used keys include:  abcdfgiklmnopqruwyz?[]<>-#123456789
        if key == '?':
            print("============================================================")
            print("       MAIN OPERATIONS")
            print("p       : toggle pan/zoom with the cursor")
            print("w       : write the model and a spectrum with the associated fitting regions")
            print("q       : exit")
            print("------------------------------------------------------------")
            print("       FITTING COMMANDS")
            print("k       : Add a line at the specified location (line ID will depend on panel)")
            print("l       : Add a Lya line at the specified location (line ID will always be Lya - regardless of panel)")
            print("o       : Add a metal (assumed to be 'aluminium') line at the specified location")
            print("d       : Delete the line nearest to the cursor")
            print("a       : Manually adjust the column density and  bval of the line nearest to the cursor")
            print("z       : Manually adjust the redshift of the line nearest to the cursor")
            print("f       : Fit the current regions in all panels with ALIS")
            print("c       : Clear current fitting (start over)")
            print("m       : Merge current fitting into master model and master regions")
            print("u       : Undo most recent operation")
            print("------------------------------------------------------------")
            print("       INTERACTION COMMANDS")
            print("[ / ]   : pan left and right")
            print("< / >   : zoom in and out")
            print("g       : Go to (centre the mouse position in any panel as Lya in the top left panel)")
            print("1-9     : Go to (centre the mouse position, assuming the line is Ly1-9)")
            print("b       : Go back (centre the mouse position in any panel as the position before moving)")
            print("i       : Obtain information on the line closest to the cursor")
            print("r       : toggle residuals plotting (i.e. remove master model from data)")
            print("------------------------------------------------------------")
        elif key == 'a' or key == 'z':
            if self._update_model is None:
                if self.lines_act.size == 0:
                    return
                cls = self.lines_act.find_closest(self.prop._wave[mouseidx], self.lines)
                self.lines_upd.clear()
                self.lines_upd.add_absline_inst(self.lines_act, cls[1])
                self._update_model = [key, cls[0], cls[1]]
            elif self._update_model[0] == key:
                # Update the model
                params = [self._update_model[2], self.lines_upd.coldens[0], self.lines_upd.redshift[0],
                          self.lines_upd.bval[0], self.lines_upd.label[0], None]
                # Update the absorption line
                self.operations('ul', axisID, mouseidx, params=params)
                self.clear(axisID, kind='update')
            else:
                # If any other key is pressed, ignore the update
                self.clear(axisID, kind='update')
            self._fitchng = True
        elif key == 'b':
            self.goback()
        elif key == 'c':
            self.clear(axisID, kind='actors')
            if autosave: self.autosave('c', axisID, mouseidx)
            self._fitchng = True
        elif key == 'd':
            self.delete_line(mouseidx)
            if autosave: self.autosave('d', axisID, mouseidx)
            self._fitchng = True
        elif key == 'f':
            self._respreq = self.fit_alis()
            if self._respreq[0]:
                self.update_infobox(message="Accept fit?", yesno=True)
            if autosave: self.autosave('f', axisID, mouseidx)
        elif key == 'g':
            self.goto(mouseidx)
        # 1 2 3 4 5 6 7 8 9 ...
        elif key in ['{0:d}'.format(ii) for ii in range(self.lines.size)]:
            self.goto(mouseidx, waveid=int(key)-1)
        elif key == 'i':
            self.lineinfo(mouseidx)
        elif key == 'k':
            self.add_absline(axisID, mouseidx)
            if autosave: self.autosave('k', axisID, mouseidx)
            self._fitchng = True
        elif key == 'l':
            self.add_absline(axisID, mouseidx, kind='lya')
            if autosave: self.autosave('l', axisID, mouseidx)
            self._fitchng = True
        elif key == 'm':
            if self._fitchng:
                self.update_infobox(message="You must fit the actors with ALIS before merging!", yesno=False)
            else:
                self._respreq = [True, "mst_merge"]
                self.update_infobox(message="Confirm merge to master", yesno=True)
            if autosave:
                self.autosave('m', axisID, mouseidx)
                self.autosave_quick()
        elif key == 'o':
            self.add_absline(axisID, mouseidx, kind='metal')
            if autosave: self.autosave('l', axisID, mouseidx)
            self._fitchng = True
        # Don't need to explicitly put p in here
        elif key == 'q':
            if self.lines_act.size != 0:
                self.update_infobox(message="WARNING: There are unsaved changes!!\nPress q again to exit", yesno=False)
                self._qconf = True
            else:
                sys.exit()
        elif key == 'r':
            if params is None:
                resid = self._resid
            else:
                resid = params[0]
            self.toggle_residuals(resid)
            if autosave: self.autosave('r', axisID, mouseidx, params=[resid])
        elif key == 'u':
            self.undo()
        elif key == 'w':
            if os.path.exists(self.lines_mst.masterLines_name(self.prop)):
                self._respreq = [True, "write"]
                self.update_infobox(message="Overwrite lines file?", yesno=True)
            else:
                self.write_data()
        elif key == ']':
            self.shift_waverange(shiftdir=+1)
            if autosave: self.autosave(']', axisID, mouseidx)
        elif key == '[':
            self.shift_waverange(shiftdir=-1)
            if autosave: self.autosave('[', axisID, mouseidx)
        elif key == '>':
            self.zoom_waverange(zoomdir=+1)
        elif key == '<':
            self.zoom_waverange(zoomdir=-1)
        elif key == 'ua':
            self.update_actors(axisID, locs=params)
            self._fitchng = True
            if autosave: self.autosave('ua', axisID, mouseidx, params=params)
        elif key == 'ul':
            self.update_absline(params=params)
            if autosave: self.autosave('ul', axisID, mouseidx, params=params)
            self._fitchng = True
        elif key == 'uli':
            self.toggle_fitidx(self.prop._wave[params[0]], self.lines, params[1])
            if autosave: self.autosave('uli', axisID, mouseidx, params=params)
            self._fitchng = True
        elif key == 'ulc':
            self.lines_act.copy(self.lines_upd)
            if autosave: self.autosave('ulc', axisID, mouseidx)

    def autosave(self, kbID, axisID, mouseidx, params=None):
        """
        For each operation performed on the data, save information about the operation performed.
        """
        # TODO :: I'm not sure either of the autosave features are working.
        # Save the current state of the class
        self.store_actions()
        # Save the data
        f = open("{0:s}.logger".format(self.prop._outp), "a+")
        if params is None:
            f.write("'{0:s}', {1:d}, {2:d}\n".format(kbID, axisID, mouseidx))
        else:
            strlist = ','.join(str(pp) for pp in params)
            f.write("'{0:s}', {1:d}, {2:d}, params=[{3:s}]\n".format(kbID, axisID, mouseidx, strlist))
        f.close()
        return

    def autosave_quick(self):
        """
        Save the master lines if we want to perform a quick load,
        without having to recalculate the entire analysis process
        performed on the data.
        """
        # First write the regions
        np.save("{0:s}_master.regions".format(self.prop._outp), self.prop._regions)
        # Now write the lines
        with open("{0:s}_master.lines".format(self.prop._outp), 'wb') as f:
            # Pickle the master lines using the highest protocol available.
            pickle.dump(self.lines_mst, f, pickle.HIGHEST_PROTOCOL)
        return

    def autoload(self):
        outname = "{0:s}.logger".format(self.prop._outp)
        answer = ''
        if os.path.exists(outname):
            print("The following file already exists:\n{0:s}".format(outname))
            while (answer != 'o') and (answer != 'l') and (answer != 'q'):
                answer = input("Would you like to overwrite (o), load from file (l), or load quickly (q)? ")
        else:
            answer = 'o'
        if answer == 'o':
            f = open(outname, "w")
            f.write("# This LOGGER file was generated on {0:s}\n".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            # Now save relevant information about how this code is run
            f.write("self.naxis=={0:d}\n".format(self.naxis))
            f.write("self.prop._qsoname=='{0:s}'\n".format(self.prop._qsoname))
            f.write("self.prop._qsopath=='{0:s}'\n".format(self.prop._qsopath))
            for ll in range(self.lines.size):
                f.write("abs(self.lines[{0:d}]-{1:f})<1.0E-4\n".format(ll, self.lines[ll]))
            # Separate preamble from code operations with a series of dashes
            f.write("------------------------------\n")
            f.close()
        elif answer == 'l':
            print("Loading file...")
            lines = open(outname, "r").readlines()
            loadops = False
            for ll, line in enumerate(lines):
                if line[0] == "#":
                    # A comment line
                    continue
                elif line[0] == "-":
                    # End of file checks, begin loading operations
                    print("All checks passed! Loading operations (this may take a while...)")
                    loadops = True
                    continue
                if not loadops:
                    # Check that the file is consistent with this run of the code
                    if not eval(line.strip("\n")):
                        print("The Logger file:\n{0:s}\nis not consistent with the current setup.".format(outname))
                        print("Please delete this file, or change the current setup to match, then rerun the code.")
                        sys.exit()
                else:
                    progress = int(100*ll/(len(lines)-1))
                    sys.stdout.write("Load progress: {0:d}%   \r".format(progress))
                    sys.stdout.flush()
                    # Load the operations
                    txtop = "self.operations({0:s}, autosave=False)".format(line.strip('\n'))
                    eval(txtop)
        elif answer == 'q':
            # Perform a quick load of the master lines and master regions
            self.prop._regions = np.load("{0:s}_master.regions.npy".format(self.prop._outp))
            with open("{0:s}_master.lines".format(self.prop._outp), 'rb') as f:
                self.lines_mst = pickle.load(f)
            # Once the master lines are loaded, generate the model curve for the actors
            self.model_mst = np.ones(self.prop._wave.size)
            for i in range(self.lines_mst.size):
                for j in range(self.naxis):
                    p0, p1, p2 = self.lines_mst.coldens[i], self.lines_mst.redshift[i], self.lines_mst.bval[i]
                    atidx = np.argmin(np.abs(self.lines[j]-self.atom._atom_wvl))
                    wv = self.lines[j]
                    fv = self.atom._atom_fvl[atidx]
                    gm = self.atom._atom_gam[atidx]
                    self.model_mst *= voigt(self.prop._wave, p0, p1, p2, wv, fv, gm)
            self.model_cnv = convolve_spec(self.prop._wave.copy(), self.model_mst.copy(), self.prop._vfwhm)
        return

    def key_release_callback(self, event):
        """
        whenever a key is released
        """
        if not event.inaxes:
            return

    def add_absline(self, axisID, mouseidx, kind=None):
        # Take the rest wavelength directly from the panel (unless kind is specified)
        wave0 = self.lines[axisID]
        zp1 = self.prop._wave[mouseidx] / wave0
        wlin = np.where(self.lines > np.min(self.prop._wave) / zp1)
        label = self.lines_act.auto_label(self.lines[wlin])
        if kind == 'lya':
            # Use H I Lyman alpha
            wave0 = 1215.6701
            zp1 = self.prop._wave[mouseidx] / wave0
            wlin = np.where(self.lines > np.min(self.prop._wave) / zp1)
            label = self.lines_act.auto_label(self.lines[wlin])
        elif kind == 'metal':
            # An example metal line
            wave0 = 1670.78861
            label = ["27Al_II-{0:s}".format(str(wave0))]
        # Get a quick fit to estimate some parameters
        fitvals = self.fit_oneline(wave0, mouseidx)
        if fitvals is not None:
            coldens, zabs, bval = fitvals
            inbounds = self.check_absline_bounds(coldens, zabs, bval)
            if inbounds:
                self.lines_act.add_absline(coldens, zabs, bval, label, shifts=len(label))
                self.draw_lines()
                self.draw_model()
            else:
                self.update_infobox("Single line fit failed (out of bounds parameters) - Try again", yesno=False)
        self.canvas.draw()

    def update_absline(self, params=None):
        self.lines_act.update_absline(params[0], coldens=params[1], redshift=params[2], bval=params[3],
                                      label=params[4], errs=params[5])
        return

    def delete_line(self, mouseidx):
        if self.lines_act.size == 0:
            return
        self.lines_act.delete_absline(self.prop._wave[mouseidx], self.lines)
        self.draw_lines()
        self.draw_model()
        self.canvas.draw()

    def fit_alis(self, nextra=50):
        # First prepare the ALIS fits
        allreg, wavidx = self.prep_alis_fits(nextra=nextra)
        nsnip = len(wavidx)
        if np.sum(allreg) == 0:
            self.update_infobox("You must select some data to be included in the fit", yesno=False)
            return [False, None]
        for dd in range(nsnip):
            # Find the endpoints
            lpix = wavidx[dd][0]
            rpix = wavidx[dd][1]
            # Extra the data to be fitted
            wave = self.prop._wave[lpix:rpix]
            flux = self.prop._flux[lpix:rpix]
            flue = self.prop._flue[lpix:rpix]
            # Extra the regions of the data to be fitted
            fito = allreg[lpix:rpix]
            # Get the continuum
            cont = self.model_cnv[lpix:rpix]
            # Save the temporary data
            if not os.path.exists("tempdata"):
                os.mkdir("tempdata")
            np.savetxt("tempdata/data_{0:02d}.dat".format(dd), np.transpose((wave, flux, flue, fito, cont)))

        # Get the ALIS parameter, data, and model lines
        parlines = self.lines_act.alis_parlines(name=self.prop._qsoname)
        datlines = self.lines_act.alis_datlines(nsnip, res=self.prop._vfwhm)
        modlines = self.lines_act.alis_modlines(nsnip)

        # Some testing to check the ALIS file is being constructed correctly
        writefile = False
        if writefile:
            #print(wavrng)
            fil = open("tempdata/testfile.mod", 'w')
            fil.write("\n".join(parlines) + "\n\n")
            fil.write("data read\n" + "\n".join(datlines) + "\ndata end\n\n")
            fil.write("model read\n" + "\n".join(modlines) + "\nmodel end\n")
            fil.close()

        # Run the fit with ALIS
        self.update_infobox("Commencing fit with ALIS...", yesno=False)
        try:
            result = alis(parlines=parlines, datlines=datlines, modlines=modlines, verbose=-1)
            self.update_infobox("Fit complete!", yesno=False)
        except:
            self.update_infobox("Fit failed - check terminal output", yesno=False)
            return [False, None]

        # Unpack all the ALIS fit, ready for storage in an instance of AbsorptionLines.
        res = self.unpack_alis_fits(result, nsnip)

        # Plot best-fitting model and residuals (send to lines_upd)
        nlines = res['coldens'].size
        self.lines_upd.clear()
        for ll in range(nlines):
            errs = [res['coldens_err'][ll], res['redshift_err'][ll], res['bval_err'][ll]]
            self.lines_upd.add_absline(res['coldens'][ll], res['redshift'][ll], res['bval'][ll], res['label'][ll],
                                       fitidx=res['fitidx'][ll], errs=errs,
                                       shifts=res['redshift_all'][ll], err_shifts=res['redshift_all_err'][ll])
        self._update_model = ['f']
        self.update_plot()
        # Clean up (delete the data used in the fitting)
        if not writefile:
            rmtree("tempdata")
        return [True, "fit_alis"]

    def prep_alis_fits(self, nextra=50):
        """
        Define all of the data 'snips' and update the fitidx accordingly

        Return:
            wavidx : data indices to extract data.
        """
        # Find all fitting regions
        allreg = np.zeros(self.prop._regions.size)
        for axisID in range(self.naxis):
            # Find all regions
            regwhr = np.copy(self.actors[axisID] == 1)
            # Fudge to get the leftmost pixel shaded in too
            lpix = np.where((self.actors[axisID][:-1] == 0) & (self.actors[axisID][1:] == 1))
            regwhr[lpix] = True
            allreg[regwhr] = 1

        # Find where the gap between two fitted regions is more than 2*nextra pixels
        lpix = np.where((allreg[:-1] == 0) & (allreg[1:] == 1))[0]
        rpix = np.where((allreg[:-1] == 1) & (allreg[1:] == 0))[0]
        wavidx = []
        lpixend = lpix[0]
        for tt in range(0, lpix.size-1):
            if rpix[tt]+2*nextra < lpix[tt+1]:
                # Too much space between fitted regions - store data values
                rpixend = rpix[tt]
                wavidx.append([lpixend - nextra - 1, rpixend + nextra])
                lpixend = lpix[tt+1]
            else:
                # Still consider this to be the same fitted region
                pass
        wavidx.append([lpixend - nextra - 1, rpix[-1] + nextra])

        # Now associate each of the absorption lines into one of the above snips
        for tt in range(len(wavidx)):
            wmin = self.prop._wave[wavidx[tt][0]]
            wmax = self.prop._wave[wavidx[tt][1]]
            self.lines_act.update_fitidx(tt, rng=[wmin, wmax])

        return allreg, wavidx

    def unpack_alis_fits(self, result, nsnip, numsims=100000):
        """
        Convert the returned instance of ALIS into absorption line parameters and errors that can be stored
        """
        # First create a dictionary that will store the relevant results
        resdict = dict({})

        # Convert the results into an easy read format
        fres = result._fitresults
        info = [(result._tend - result._tstart)/3600.0, fres.fnorm, fres.dof, fres.niter, fres.status]
        alis_lines = get_alis_string(result, fres.params, fres.perror, info,
                                     printout=False, getlines=True, save=False)

        # Extract parameter values and the continuum
        # Scan through the model to find the velocity shifts of each portion of spectrum
        alspl = alis_lines.split("\n")
        vshift = np.zeros(nsnip)
        err_vshift = np.zeros(nsnip)
        flag = 0
        for spl in range(len(alspl)):
            if "# Shift Models:" in alspl[spl]:
                flag = 1
                continue
            elif "# Errors:" in  alspl[spl]:
                flag = 2
                continue
            if flag != 0:
                for snp in range(nsnip):
                    if snp == 0:
                        vshift[snp] = 0.0
                        err_vshift[snp] = 0.0
                        continue
                    shtxt = "shift{0:02d}".format(snp)
                    if shtxt in alspl[spl]:
                        vspl = alspl[spl].split()[2]
                        if flag == 1:
                            vshift[snp] = float(vspl.split("s")[0])
                        elif flag == 2:
                            err_vshift[snp] = float(vspl.split("s")[0])
                            break

        # Extract absorption line parameters and their errors
        nlines = self.lines_act.size
        coldens, err_coldens = np.zeros(nlines), np.zeros(nlines)
        redshift, err_redshift = np.zeros(nlines), np.zeros(nlines)
        redshift_all, err_redshift_all = [], []
        fitidx = []
        bval, err_bval = np.zeros(nlines), np.zeros(nlines)
        label = []
        for ll in range(nlines):
            label.append(deepcopy(self.lines_act.label[ll]))
            redshift_all.append(-1 * np.ones(len(self.lines_act.label[ll])))
            err_redshift_all.append(-1 * np.ones(len(self.lines_act.label[ll])))
            fitidx.append(-1 * np.ones(len(self.lines_act.label[ll])))

        # Search through alis_lines for the appropriate string
        alspl = alis_lines.split("\n")
        flag = 0
        for spl in range(len(alspl)):
            if " absorption" in alspl[spl]:
                # Values
                flag = 1
                continue
            elif "model end" in alspl[spl]:
                flag = 0
                continue
            elif "#absorption" in alspl[spl]:
                # Errors
                flag = 2
                continue
            elif len(alspl[spl].strip()) == 0:
                # An empty line
                flag = 0
                continue
            # Go through all the lines
            for ll in range(nlines):
                for snp in range(len(self.lines_act.fitidx[ll])):
                    if self.lines_act.fitidx[ll][snp] == -1:
                        # A line is not being fit
                        continue
                    # Extract the line profile information
                    if flag == 1:
                        if "specid=line{0:02d}".format(self.lines_act.fitidx[ll][snp]) in alspl[spl]:
                            vspl = alspl[spl].split()
                            if vspl[2].split("n")[1] == str(ll):
                                label[ll][snp] = self.lines_act.label[ll][snp]
                                coldens[ll] = float(vspl[2].split("n")[0])
                                redshift[ll] = float(vspl[3].split("z")[0])
                                bval[ll] = float(vspl[4].split("b")[0])
                                redshift_all[ll][snp] = vshift[self.lines_act.fitidx[ll][snp]]
                                fitidx[ll][snp] = self.lines_act.fitidx[ll][snp]
                    elif flag == 2:
                        if "specid=line{0:02d}".format(self.lines_act.fitidx[ll][snp]) in alspl[spl]:
                            vspl = alspl[spl].split()
                            if vspl[3].split("n")[1] == str(ll):
                                err_coldens[ll] = float(vspl[3].split("n")[0])
                                err_redshift[ll] = float(vspl[4].split("z")[0])
                                err_bval[ll] = float(vspl[5].split("b")[0])
                                err_redshift_all[ll][snp] = 0.0

        # Generate a new set of starting parameters
        ptb = newstart(fres.covar, numsims)
        inarr = np.array(fres.params)
        cntr = np.zeros(fres.covar.shape[0])
        cntr[np.where(np.diag(fres.covar) > 0.0)] = 1.0
        cntr = np.cumsum(cntr) - 1

        # Take into account covariance between the velocity shift and redshift
        idx, pidx = np.zeros(2, dtype=np.int), np.zeros(2, dtype=np.int)
        for ll in range(nlines):
            # For each cloud
            lmin = np.argmin(np.abs(inarr - redshift[ll]))
            idx[0] = lmin
            pidx[0] = cntr[lmin]
            for ss in range(redshift_all[ll].size):
                if redshift_all[ll][ss] == -1.0:
                    continue
                # For each Lyman series absorption line
                smin = np.argmin(np.abs(inarr - redshift_all[ll][ss]))
                idx[1] = smin
                pidx[1] = cntr[smin]
                # Calculate the perturbed values
                ptrbvals = np.outer(inarr[idx], np.ones(numsims)) + ptb[pidx, :]
                # Determine the redshift of the line (take into account the vshift) and store the result
                newzabs = (1.0 + ptrbvals[0, :]) / (1.0 - ptrbvals[1, :] / 299792.458) - 1.0  # This has been explicitly checked
                redshift_all[ll][ss] = np.mean(newzabs)
                err_redshift_all[ll][ss] = np.std(newzabs)

        # Store the continuum fits
        # TODO :: continuum is lost when accepting a fit - is this OK/desirible?
        # It means that the residuals won't look right, but the stored parameters will be OK
        self.model_cnt[:] = 1.0
        for cc in range(len(result._contfinal)):
            val = np.nonzero(np.in1d(self.prop._wave, result._wavefull[cc]))
            ww = np.where(result._contfinal[cc] > -10.0)
            self.model_cnt[val[0][ww]] *= result._contfinal[cc][ww]

        # Store everything compactly in a dictionary
        resdict['label'] = deepcopy(label)
        resdict['fitidx'] = deepcopy(fitidx)
        resdict['coldens'] = coldens
        resdict['coldens_err'] = err_coldens
        resdict['redshift'] = redshift
        resdict['redshift_err'] = err_redshift
        resdict['bval'] = bval
        resdict['bval_err'] = err_bval
        resdict['redshift_all'] = redshift_all
        resdict['redshift_all_err'] = err_redshift_all

        debug = False
        if debug:
            np.save("work/fres_covar", fres.covar)
            np.save("work/inarr", inarr)
            np.save("work/redshift", redshift)
            np.save("work/err_redshift", err_redshift)
            np.save("work/vshift", vshift)
            np.save("work/err_vshift", err_vshift)
            print(vshift)
            for ll in range(nlines):
                print("-----------------")
                print(resdict['label'][ll])
                print(resdict['coldens'][ll])
                print(resdict['coldens_err'][ll])
                print(resdict['redshift'][ll])
                print(resdict['redshift_err'][ll])
                print(resdict['bval'][ll])
                print(resdict['bval_err'][ll])
                print(resdict['redshift_all'][ll][:])
                print(resdict['redshift_all_err'][ll][:])
                print(resdict['fitidx'][ll][:])
                print("-----------------")

        # Return the dictionary
        return resdict

    def merge_alis(self, resp):
        # If the user is happy with the ALIS fit, update the actors with the new model parameters
        if resp == "y":
            self.operations('ulc', -1, -1)
            self._fitchng = False  # Saving is up to date
        else:
            pass
        self.clear(None, kind='update')
        return

    def fit_oneline(self, wave0, mouseidx):
        """ This performs a very quick fit to the line (using only one actor) """
        w = np.where(self.lactor == 1)
        if w[0].size <= 3:
            print("WARNING : not enough pixels to fit a single line - at least 3 pixels are needed")
            return

        # Pick some starting parameters
        coldens0 = 14.0
        zabs0 = self.prop._wave[mouseidx] / wave0 - 1.0
        bval0 = 10.0
        p0 = [coldens0, zabs0, bval0]

        # Get the atomic parameters of the line
        atidx = np.argmin(np.abs(wave0-self.atom._atom_wvl))
        wv = wave0
        fv = self.atom._atom_fvl[atidx]
        gm = self.atom._atom_gam[atidx]
        # Prepare the data to be fitted
        if self._resid:
            flxfit = self.prop._flux / (self.model_cnv*self.model_act*self.model_cnt)
            flefit = self.prop._flue / (self.model_cnv*self.model_act*self.model_cnt)
        else:
            flxfit = self.prop._flux
            flefit = self.prop._flue
        # Perform the fit
        try:
            popt, pcov = curve_fit(lambda x, ng, zg, bg : voigt(x, ng, zg, bg, wv, fv, gm), self.prop._wave[w], flxfit[w], sigma=flefit[w], method='lm', p0=p0)
            self.update_infobox(default=True)
        except RuntimeError:
            self.update_infobox(message="ERROR: Optimal parameters not found in fit_oneline")
            return None
        # Check if any of the parameters have gone "out of bounds"
        return popt[0], popt[1], popt[2]

    def check_absline_bounds(self, coldens, zabs, bval):
        inbounds = True
        if coldens < 8.0 or coldens > 22.0:
            inbounds = False
        if zabs < 0.0 or zabs > 7.0:
            inbounds = False
        if bval < 0.1 or bval > 100.0:
            inbounds = False
        return inbounds

    def goto(self, mouseidx, waveid=0):
        # Store the old values
        tmp_mn, tmp_mx = self.axs[0].get_xlim()
        self._xmnsv += [tmp_mn]
        self._xmxsv += [tmp_mx]
        # Set the new values
        lam = self.lines[waveid]*(1.0+self.prop._zem)
        vnew = 299792.458*(self.prop._wave[mouseidx]-lam)/lam
        xmn, xmx = vnew-self.veld, vnew+self.veld
        ymn, ymx = self.axs[0].get_ylim()
        for i in range(len(self.lines)):
            self.axs[i].set_xlim([xmn, xmx])
            self.axs[i].set_ylim([ymn, ymx])
        self.canvas.draw()

    def goback(self):
        # If we're back to the original value
        if len(self._xmnsv) == 0:
            return
        # Go back to the old x coordinate values
        xmn, xmx = self._xmnsv[-1], self._xmxsv[-1]
        del self._xmnsv[-1]
        del self._xmxsv[-1]
        ymn, ymx = self.axs[0].get_ylim()
        for i in range(len(self.lines)):
            self.axs[i].set_xlim([xmn, xmx])
            self.axs[i].set_ylim([ymn, ymx])
        self.canvas.draw()

    def shift_waverange(self, shiftdir=-1):
        xmn, xmx = self.axs[0].get_xlim()
        ymn, ymx = self.axs[0].get_ylim()
        shft = shiftdir * 0.5 * (xmx - xmn) / 3.0
        xmn += shft
        xmx += shft
        for i in range(len(self.lines)):
            self.axs[i].set_xlim([xmn, xmx])
            self.axs[i].set_ylim([ymn, ymx])
        self.canvas.draw()

    def zoom_waverange(self, zoomdir=-1):
        xmn, xmx = self.axs[0].get_xlim()
        ymn, ymx = self.axs[0].get_ylim()
        shft = zoomdir * 0.5 * (xmx - xmn)
        if zoomdir == -1:
            shft *= 0.5
        xmn -= shft
        xmx += shft
        for i in range(len(self.lines)):
            self.axs[i].set_xlim([xmn, xmx])
            self.axs[i].set_ylim([ymn, ymx])
        self.canvas.draw()

    def lineinfo(self, mouseidx):
        self.lines_act.lineinfo(self.prop._wave[mouseidx], np.append(self.lines, 1670.78861))

    def toggle_fitidx(self, west, lines, flip):
        self.lines_act.toggle_fitidx(west, lines, flip)
        self.draw_lines()
        self.canvas.draw()

    def toggle_residuals(self, resid):
        for i in range(self.naxis):
            if resid:
                # If residuals are currently being shown, plot the original data
                self.specs[i].set_ydata(self.prop._flux)
            else:
                # Otherwise, plot the residuals
                #self.specs[i].set_ydata(self.prop._flux/(self.model_cnv*self.model_act*self.model_cnt))
                if self._update_model is None:
                    # If a model is not being updated, show the ratio
                    self.specs[i].set_ydata(self.prop._flux / (self.model_cnv * self.model_act))
                else:
                    if self._update_model[0] == 'f':
                        # A fit from ALIS is being shown, so show the updated model
                        self.specs[i].set_ydata(self.resloc[0] + self.resloc[1] * (self.prop._flux - (self.model_cnv * self.model_upd))/self.prop._flue)
                    else:
                        self.specs[i].set_ydata(self.resloc[0] + self.resloc[1] * (self.prop._flux - (self.model_cnv * self.model_act)) / self.prop._flue)
        self.canvas.draw()
        self.canvas.flush_events()
        self._resid = not resid

    def clear(self, axisID, kind='update'):
        if kind == 'actors':
            self.update_actors(axisID, clear=True)
        elif kind == 'update':
            self.lines_upd.clear()
            self.model_cnt[:] = 1.0
            self._update_model = None  # lines_upd should no longer be plotted.
        # Update the plot
        self.update_plot()
        return

    def update_plot(self):
        self.draw_lines()
        self.draw_model()
        self.draw_model_update()
        self.canvas.draw()
        self.canvas.flush_events()

    def plot_actor(self, axisID):
        # Plot the new actor
        if self.curreg[axisID] is not None:
            self.curreg[axisID].remove()
            self.curreg[axisID] = None
        # Plot the selected region
        trans = mtransforms.blended_transform_factory(self.axs[axisID].transData, self.axs[axisID].transAxes)
        self.canvas.restore_region(self.backgrounds[axisID])
        lam = self.lines[axisID]*(1.0 + self._zqso)
        velo = 299792.458*(self.prop._wave-lam)/lam
        # Find all regions
        regwhr = np.copy(self.actors[axisID] == 1)
        # Fudge to get the leftmost pixel shaded in too
        regwhr[np.where((self.actors[axisID][:-1] == 0) & (self.actors[axisID][1:] == 1))] = True
        self.curreg[axisID] = self.axs[axisID].fill_between(velo, 0, 1, where=regwhr, facecolor='red', alpha=0.5, transform=trans)
        self.canvas.draw()

    def update_actors(self, axisID, locs=None, clear=False):
        # Clear all actors if the user requests
        if clear:
            for i in range(self.naxis):
                self.actors[i][:] = 0
                self.lactor[:] = 0
                self.lines_act.clear()
            return
        else:
            start, end, addsub = locs[0], locs[1], locs[2]
        # Otherwise, update the actors
        if end != start:
            # Reset start if start > end
            if start > end:
                tmp = start
                start = end
                end = tmp
            # Set the pixels for the current selection
            self.lactor[:] = 0
            self.lactor[start:end] = addsub
            # Set the corresponding pixels in the actor
            self.actors[axisID][start:end] = addsub

    def update_infobox(self, message="Press '?' to list the available options",
                       yesno=True, default=False):
        self.axi.clear()
        if default:
            self.axi.text(0.5, 0.5, "Press '?' to list the available options", transform=self.axi.transAxes,
                          horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
        # Display the message
        self.axi.text(0.5, 0.5, message, transform=self.axi.transAxes,
                      horizontalalignment='center', verticalalignment='center')
        if yesno:
            self.axi.fill_between([0.8, 0.9], 0, 1, facecolor='green', alpha=0.5, transform=self.axi.transAxes)
            self.axi.fill_between([0.9, 1.0], 0, 1, facecolor='red', alpha=0.5, transform=self.axi.transAxes)
            self.axi.text(0.85, 0.5, "YES", transform=self.axi.transAxes,
                          horizontalalignment='center', verticalalignment='center')
            self.axi.text(0.95, 0.5, "NO", transform=self.axi.transAxes,
                          horizontalalignment='center', verticalalignment='center')
        self.axi.set_xlim((0, 1))
        self.axi.set_ylim((0, 1))
        self.canvas.draw()

    def update_master(self):
        # Store the master regions and lines.
        self.update_master_regions()
        self.update_master_lines()
        # Merge models and reset
        self.model_mst *= self.model_act*self.model_cnt
        self.model_cnv = convolve_spec(self.prop._wave.copy(), self.model_mst.copy(), self.prop._vfwhm)
        self.model_act[:] = 1.0
        self.model_cnt[:] = 1.0
        # Clear the actors
        self.clear(None, kind='actors')
        # Update the plotted lines
        self.update_plot()
        return

    def update_master_regions(self):
        for ii in range(self.naxis):
            ww = np.where(self.actors[ii] == 1)
            self.prop._regions[ww] = 1

    def update_master_lines(self):
        for ii in range(self.lines_act.size):
            self.lines_mst.add_absline_inst(self.lines_act, 0)  # Don't change this zero
            self.lines_act.delete_absline_idx(0)

    def update_waverange(self):
        for i in range(len(self.lines)):
            #wcen = self.lines[i]*(1.0+self.prop._zem)
            #xmn = wcen * (1.0 - self.veld/299792.458)
            #xmx = wcen * (1.0 + self.veld/299792.458)
            xmn, xmx = -self.veld, self.veld
            self.axs[i].set_xlim([xmn, xmx])
            self.axs[i].set_ylim([-0.1, self.resloc[0]+3*self.resloc[1]])
        self.canvas.draw()

    def write_data(self):
        self.lines_mst.write_data(self.prop)
        return


class Props:
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
        # Store some properties of the quasar
        self._qsoname = qso._name
        self._qsopath = qso._path
        self._qsofilename = qso._filename
        self._outp = qso._path + qso._filename.replace(".dat", "")
        self._outf = outf
        self._zem = qso._zem
        self._vfwhm = qso._vFWHM
        self._ra = qso._RA
        self._dec = qso._DEC
        self._mjdobs = qso._MJDobs

    def set_regions(self, arr):
        self._regions = arr.copy()
        return

    def add_region(self, arr):
        self._regions += arr.copy()
        np.clip(self._regions, 0, 1)
        return


class AbsorptionLines:
    def __init__(self):
        self.clear()
        return

    @property
    def size(self):
        return self.coldens.size

    def alis_datlines(self, nsnip, res=7.0):
        """
        nsnip = number of data regions extracted.
        res = instrument resolution in km/s
        """
        datlines = []
        for snp in range(nsnip):
            if snp == 0:
                shtxt = "0.0SFIX"
            else:
                shtxt = "0.0shift{0:02d}".format(snp)
            datlines += ["tempdata/data_{0:02d}.dat  specid=line{0:02d}  fitrange=columns  loadrange=all  resolution=vfwhm({1:.3f}RFIX) shift=vshift({2:s}) columns=[wave:0,flux:1,error:2,fitrange:3,continuum:4]".format(snp, res, shtxt)]
        return datlines

    def alis_modlines(self, nsnip):
        modlines = []
        # Do the emission
        modlines += ["emission"]
        for snp in range(nsnip):
            modlines += ["legendre  1.0 0.0 0.0 0.0  scale=1.0,1.0,1.0,1.0  specid=line{0:02d}".format(snp)]
        # Do the absorption
        modlines += ["absorption"]
        for ll in range(self.size):
            ion = self.label[ll][0].split("-")[0]
            for xx in range(len(self.fitidx[ll])):
                if self.fitidx[ll][xx] == -1:
                    continue
                modlines += ["voigt ion={0:s}  {1:.4f}n{4:d}  {2:.9f}z{4:d}  {3:.3f}b{4:d}  0.0TFIX  specid=line{5:02d}".format(
                    ion, self.coldens[ll], self.redshift[ll], self.bval[ll], ll, self.fitidx[ll][xx])]
        return modlines

    def alis_parlines(self, name="quasarname", basic=True):
        parlines = []
        parlines += ["run  ncpus  6"]
        parlines += ["run ngpus 0"]
        parlines += ["run nsubpix 5"]
        parlines += ["run blind False"]
        parlines += ["run convergence False"]
        parlines += ["run convcriteria 0.2"]
        parlines += ["chisq atol 0.001"]
        parlines += ["chisq xtol 0.0"]
        parlines += ["chisq ftol 0.0"]
        parlines += ["chisq gtol 0.0"]
        parlines += ["chisq fstep 1.3"]
        parlines += ["chisq miniter 10"]
        parlines += ["chisq maxiter 1000"]
        if basic:
            parlines += ["out model False"]
            parlines += ["out fits False"]
            parlines += ["out verbose -1"]
        else:
            parlines += ["out model True"]
            parlines += ["out covar {0:s}.mod.out.covar".format(name)]
            parlines += ["out fits True"]
            parlines += ["out verbose 1"]
            parlines += ["out plots {0:s}_fits.pdf".format(name)]
        parlines += ["out overwrite True"]
        if basic:
            parlines += ["plot fits False"]
        else:
            parlines += ["plot dims 3x1"]
            parlines += ["plot ticklabels True"]
            parlines += ["plot labels True"]
            parlines += ["plot fitregions True"]
            parlines += ["plot fits True"]
        return parlines

    def auto_label(self, lines, ion="1H_I"):
        return ["{0:s}-{1:s}".format(ion, str(lines[ll])) for ll in range(len(lines))]

    def add_absline(self, coldens, redshift, bval, label, fitidx=None, errs=None, shifts=None, err_shifts=None):
        # Add the values
        self.coldens = np.append(self.coldens, coldens)
        self.redshift = np.append(self.redshift, redshift)
        self.bval = np.append(self.bval, bval)
        # Add the errors
        if errs is None:
            self.err_coldens = np.append(self.err_coldens, coldens)
            self.err_redshift = np.append(self.err_redshift, redshift)
            self.err_bval = np.append(self.err_bval, bval)
        else:
            self.err_coldens = np.append(self.err_coldens, errs[0])
            self.err_redshift = np.append(self.err_redshift, errs[1])
            self.err_bval = np.append(self.err_bval, errs[2])
        # Append the shifts and errors
        if type(shifts) is int:
            self.shifts.append(-1*np.ones(shifts))
            self.err_shifts.append(-1*np.ones(shifts))
        elif type(shifts) is np.ndarray and type(err_shifts) is np.ndarray:
            self.shifts.append(shifts.copy())
            self.err_shifts.append(err_shifts.copy())
        else:
            self.shifts.append(-1*np.ones(len(label)))
            self.err_shifts.append(-1*np.ones(len(label)))
        # Append the label
        self.label.append(deepcopy(label))
        if fitidx is None:
            self.fitidx.append(-1*np.ones(len(label), dtype=np.int))
        else:
            self.fitidx.append(fitidx.copy())
        return

    def add_absline_inst(self, inst, idx):
        errs = [inst.err_coldens[idx], inst.err_redshift[idx], inst.err_bval[idx]]
        self.add_absline(inst.coldens[idx], inst.redshift[idx], inst.bval[idx], inst.label[idx],
                         fitidx=inst.fitidx[idx], errs=errs, shifts=inst.shifts[idx], err_shifts=inst.err_shifts[idx])
        return

    def clear(self):
        """
        Delete all absorption lines - start from scratch
        """
        # Values
        self.coldens = np.array([])
        self.redshift = np.array([])
        self.bval = np.array([])
        # Errors
        self.err_coldens = np.array([])
        self.err_redshift = np.array([])
        self.err_bval = np.array([])
        # Shifts
        self.shifts = []
        self.err_shifts = []
        # A label
        self.label = []
        # Which lines are included in the fit
        self.fitidx = []
        return

    def copy(self, inst):
        # Values
        self.coldens = inst.coldens.copy()
        self.redshift = inst.redshift.copy()
        self.bval = inst.bval.copy()
        # Errors
        self.err_coldens = inst.err_coldens.copy()
        self.err_redshift = inst.err_redshift.copy()
        self.err_bval = inst.err_bval.copy()
        # Shifts
        self.shifts = deepcopy(inst.shifts)
        self.err_shifts = deepcopy(inst.err_shifts)
        # A label
        self.label = deepcopy(inst.label)
        # Fit index
        self.fitidx = deepcopy(inst.fitidx)
        return

    def delete_absline(self, west, lines):
        """
        west : The estimated wavelength of the line to be deleted
        """
        # Find the line to be deleted
        lidx, widx = self.find_closest(west, lines)
        # Delete the
        self.delete_absline_idx(widx)
        return

    def delete_absline_idx(self, idx):
        # Delete the values
        self.coldens = np.delete(self.coldens, (idx,), axis=0)
        self.redshift = np.delete(self.redshift, (idx,), axis=0)
        self.bval = np.delete(self.bval, (idx,), axis=0)
        # Delete the errors
        self.err_coldens = np.delete(self.err_coldens, (idx,), axis=0)
        self.err_redshift = np.delete(self.err_redshift, (idx,), axis=0)
        self.err_bval = np.delete(self.err_bval, (idx,), axis=0)
        # Delete the shifts
        del self.shifts[idx]
        del self.err_shifts[idx]
        # Delete the label
        del self.label[idx]
        # Delete the fitidx
        del self.fitidx[idx]
        return

    def find_closest(self, west, lines):
        """
        west : The estimated wavelength of the line to be deleted
        """
        amin = np.argmin(np.abs(np.outer(lines, 1 + self.redshift) - west))
        idx = np.unravel_index(amin, (lines.size, self.redshift.size))
        lidx, widx = int(idx[0]), int(idx[1])
        return lidx, widx

    def toggle_fitidx(self, west, lines, flip):
        """
        west : The estimated wavelength of the line to be deleted from linelist 'lines'
        flip : 0 to turn off fitidx, 1 to turn it on.
        """
        lidx, widx = self.find_closest(west, lines)
        if flip == 0:
            self.fitidx[widx][lidx] = -1
        else:
            self.fitidx[widx][lidx] = 0

    def getabs(self, idx, errs=False):
        # Return the values
        retval = (self.coldens[idx], self.redshift[idx], self.bval[idx])
        # and errors if requested
        if errs:
            retval += (self.err_coldens[idx], self.err_redshift[idx], self.err_bval[idx])
        # Finally, include the label and fitidx
        retval += (self.label[idx], self.fitidx[idx])
        return retval

    def lineinfo(self, west, lines):
        """
        west : The estimated wavelength of the line to be deleted
        """
        lidx, widx = self.find_closest(west, lines)
        print("=======================================================")
        print("Line Information:")
        print("  {0:s}".format(self.label[widx]))
        print("  Col Dens = {0:.3f} +/- {1:.3f}".format(self.coldens[widx], self.err_coldens[widx]))
        print("  Redshift = {0:.6f} +/- {1:.6f}".format(self.redshift[widx], self.err_redshift[widx]))
        print("  Dopp Par = {0:.2f} +/- {1:.2f}".format(self.bval[widx], self.err_bval[widx]))
        return

    def update_fitidx(self, val, rng=None):
        """
        val = index to set
        rng = two element list giving wave range allowed
        """
        if type(rng) is not list:
            return  # Wrong type
        if len(rng) != 2:
            return  # Wrong size
        # Update fitidx
        for ll in range(self.size):
            for ff in range(len(self.fitidx[ll])):
                wv0 = float(self.label[ll][ff].split("-")[1])
                wvtst = wv0 * (1.0+self.redshift[ll])
                if rng[0] < wvtst < rng[1]:
                    self.fitidx[ll][ff] = val
        return

    def update_absline(self, indx, coldens=None, redshift=None, bval=None,
                       errs=None, shifts=None, err_shifts=None, label=None, fitidx=None):
        if coldens is not None:
            self.coldens[indx] = coldens
            if errs is not None:
                self.err_coldens[indx] = errs[0]
        if redshift is not None:
            self.redshift[indx] = redshift
            if errs is not None:
                self.err_redshift[indx] = errs[1]
        if bval is not None:
            self.bval[indx] = bval
            if errs is not None:
                self.err_bval[indx] = errs[2]
        if shifts is not None:
            self.shifts[indx] = shifts.copy()
            if err_shifts is not None:
                self.err_shifts[indx] = err_shifts.copy()
        if label is not None:
            self.label[indx] = deepcopy(label)
        if fitidx is not None:
            self.fitidx[indx] = deepcopy(fitidx)
        return

    def masterLines_name(self, prop):
        mjdstr = str(prop._mjdobs).replace(".", "p")
        outname = "{0:s}_{1:s}_masterLines.fits".format(prop._outp, mjdstr)
        return outname

    def write_data(self, prop):
        """
        The information that should be saved includes:
        A header that lists the properties of the spectrum used in the analysis
         - QSO name
         - OBS date/time
         - spectral resolution
         - Maybe include the entire header from the original fits file?
        Then, the table data will include:
         - Line ID number?
         - Approximate central wavelength of the line
         - Line ID (i.e. H I Lya, H I Lyb, metal)
         - Column density and error
         - Bval and error
         - Redshift and error
        """
        outname = self.masterLines_name(prop)

        # Generate some header information
        hdr = fits.Header()
        hdr['QSOname'] = (prop._qsoname, 'Name of QSO')
        hdr['RA'] = (prop._ra, "Target right ascension")
        hdr['DEC'] = (prop._dec, "Target declination")
        hdr['MJDOBS'] = (prop._mjdobs, "MJD at the time of observation")
        hdr['specres'] = (prop._vfwhm, "Spectral resolution (km/s)")
        hduh = fits.PrimaryHDU(header=hdr)

        # initialise the variables
        ions = []
        waverest = []
        waveobs = []
        coldens = []
        coldens_err = []
        zabs = []
        zabs_err = []
        doppler = []
        doppler_err = []
        # Go through all lines and add the information
        print(self.size)
        print(self.shifts[0].size)
        print(self)
        for ll in range(self.size):
            for tt in range(self.shifts[ll].size):
                if self.fitidx[ll][tt] == -1:
                    # This line has not be included in the fit - ignore it!
                    continue
                wave0 = float(self.label[ll][tt].split("-")[1])
                ions.append(self.label[ll][tt].split("-")[0])
                waverest.append(wave0)
                waveobs.append(wave0*(1.0+self.shifts[ll][tt]))
                coldens.append(self.coldens[ll])
                coldens_err.append(self.err_coldens[ll])
                zabs.append(self.shifts[ll][tt])
                zabs_err.append(self.err_shifts[ll][tt])
                doppler.append(self.bval[ll])
                doppler_err.append(self.err_bval[ll])

        # Put the data into columns
        c1 = fits.Column(name='ion', array=ions, format='7a')
        c2 = fits.Column(name='waverest', array=waverest, format='d')
        c3 = fits.Column(name='waveobs', array=waveobs, format='d')
        c4 = fits.Column(name='coldens', array=coldens, format='d')
        c5 = fits.Column(name='coldens_err', array=coldens_err, format='d')
        c6 = fits.Column(name='zabs', array=zabs, format='d')
        c7 = fits.Column(name='zabs_err', array=zabs_err, format='d')
        c8 = fits.Column(name='doppler', array=doppler, format='d')
        c9 = fits.Column(name='doppler_err', array=doppler_err, format='d')
        # Prepare and write the fits file
        hdut = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9])
        # Create a HDU list and save to file
        hdul = fits.HDUList([hduh, hdut])
        hdul.writeto(outname)
        return


class Atomic:
    def __init__(self, filename="atomic.dat", wmin=None, wmax=None):
        # Get the filename
        self._wmin = wmin
        self._wmax = wmax
        self._atom_atm = []
        self._atom_ion = []
        self._atom_lbl = []
        self._atom_wvl = []
        self._atom_fvl = []
        self._atom_gam = []
        self._molecule_atm = []
        self._molecule_ion = []
        self._molecule_lbl = []
        self._molecule_wvl = []
        self._molecule_fvl = []
        self._molecule_gam = []
        self.load_lines(filename)
        self._solar = self.solar()

    def solar(self):
        elem = np.array(['H ', 'He','Li','Be','B ', 'C ', 'N ', 'O ', 'F ', 'Ne','Na','Mg','Al','Si','P ', 'S ', 'Cl','Ar','K ', 'Ca','Sc','Ti','V ', 'Cr','Mn','Fe','Co','Ni','Cu','Zn'])
        mass = np.array([1.0, 4.0, 7.0, 8.0, 11.0, 12.0,14.0,16.0,19.0,20.0,23.0,24.0,27.0,28.0,31.0,32.0,35.0,36.0,39.0,40.0,45.0,48.0,51.0,52.0,55.0,56.0,59.0,60.0,63.0,64.0])
        solr = np.array([12.0,10.93,3.26,1.30,2.79,8.43,7.83,8.69,4.42,7.93,6.26,7.56,6.44,7.51,5.42,7.14,5.23,6.40,5.06,6.29,3.05,4.91,3.95,5.64,5.48,7.47,4.93,6.21,4.25,4.63])
        solar = np.zeros(self._atom_atm.size)
        for i in range(elem.size):
            w = np.where(self._atom_atm == elem[i])
            solar[w] = solr[i]
        return solar

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
        self._solar = self.solar()

        # Load the lines file
        print("Loading a list of molecular transitions...")
        mfilename = "molecule.dat"
        try:
            infile = open(mfilename, "r")
        except IOError:
            print("The lines file:\n" + "molecule.dat\ndoes not exist!")
            sys.exit()
        molecule_list = infile.readlines()
        leninfile = len(molecule_list)
        infile.close()
        infile = open(mfilename, "r")
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
