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
from scipy.optimize import curve_fit
from scipy.special import wofz
from alis.alis import alis
from alis.alsave import save_model as get_alis_string
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


class SelectRegions(object):
    """
    Generate a model and regions to be fit with ALIS
    """

    def __init__(self, canvas, axs, axi, specs, prop, atom, vel=500.0, lines=None):
        """
        axs : ndarray
          array of all data axes
        axi : Axes instance
          axis used to display information
        vel : float
          Default +/- plotting window in km/s
        """
        self.axs = axs
        self.axi = axi
        self.naxis = len(axs)
        self.specs = specs
        self.prop = prop
        self.atom = atom
        self.veld = vel
        self._xmnsv, self._xmxsv = -self.veld, self.veld
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

        if lines is None:
            lines = 1215.6701*np.ones(self.naxis)
        self.lines = lines

        # Create the model lines variables
        self.modelLines_act = [None for ii in range(self.naxis)]  # The Line instance of the plotted actors model
        self.modelLines_mst = [None for ii in range(self.naxis)]  # The Line instance of the plotted master model
        self.modelLines_upd = [None for ii in range(self.naxis)]  # The Line instance of the plotted model that is being updated
        self.lines_act = AbsorptionLines()  # Stores all of the information about the actor absorption lines.
        self.lines_mst = AbsorptionLines()  # Stores all of the information about the master absorption lines.
        self.lines_upd = AbsorptionLines()  # Stores all of the information about the updated absorption line.
        self.actors = [np.zeros(self.prop._wave.size) for ii in range(self.naxis)]
        self.lactor = np.zeros(self.prop._wave.size)  # Just the previously selected actor
        self.model_act = np.ones(self.prop._wave.size)
        self.model_mst = np.ones(self.prop._wave.size)
        self.model_upd = np.ones(self.prop._wave.size)
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
            lam = self.lines[i] * (1.0 + self._zqso)
            for j in range(self.lines_act.size):
                velo = 299792.458 * (self.lines[i] * (1.0 + self.lines_act.redshift[j]) - lam) / lam
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
#        if self.lines_act.size == 0:
            # There are no model lines
#            return
        for i in self.modelLines_act:
            if i is not None:
                i.pop(0).remove()
        for i in self.modelLines_mst:
            if i is not None:
                i.pop(0).remove()
        # Generate the model curve for the actors
        # TODO :: Need to deal with instrument line spread function, somewhere...
        self.model_act = np.ones(self.prop._wave.size)
        for i in range(self.lines_act.size):
            for j in range(self.naxis):
                p0, p1, p2 = self.lines_act.coldens[i], self.lines_act.redshift[i], self.lines_act.bval[i]
                atidx = np.argmin(np.abs(self.lines[j]-self.atom._atom_wvl))
                wv = self.lines[j]
                fv = self.atom._atom_fvl[atidx]
                gm = self.atom._atom_gam[atidx]
                self.model_act *= voigt(self.prop._wave, p0, p1, p2, wv, fv, gm)
        # Plot the models
        for i in range(self.naxis):
            lam = self.lines[i]*(1.0 + self._zqso)
            velo = 299792.458*(self.prop._wave-lam)/lam
            self.modelLines_mst[i] = self.axs[i].plot(velo, self.model_mst, 'b-', linewidth=2.0)
            self.modelLines_act[i] = self.axs[i].plot(velo, self.model_act, 'r-', linewidth=1.0)
        return

    def draw_model_update(self):
        for i in range(self.naxis):
            if self.modelLines_upd[i] is not None:
                self.modelLines_upd[i].pop(0).remove()
                self.modelLines_upd[i] = None
        # Generate the model curve for the actors
        self.model_upd = np.ones(self.prop._wave.size)
        # If not currently updating, return
        if self._update_model is None:
            self.canvas.draw()
            return
        # Calculate the model
        for i in range(self.lines_upd.size):
            for j in range(self.naxis):
                p0, p1, p2 = self.lines_upd.coldens[i], self.lines_upd.redshift[i], self.lines_upd.bval[i]
                atidx = np.argmin(np.abs(self.lines[j] - self.atom._atom_wvl))
                wv = self.lines[j]
                fv = self.atom._atom_fvl[atidx]
                gm = self.atom._atom_gam[atidx]
                self.model_upd *= voigt(self.prop._wave, p0, p1, p2, wv, fv, gm)
        # Plot the models
        for i in range(self.naxis):
            lam = self.lines[i] * (1.0 + self._zqso)
            velo = 299792.458 * (self.prop._wave - lam) / lam
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
            answer = ""
            if event.xdata > 0.8 and event.xdata < 0.9:
                answer = "y"
            elif event.xdata >= 0.9:
                answer = "n"
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
        if self._respreq[0]:
            if key != "y" and key != "n":
                return
            else:
                if self._respreq[1] == "fit_alis":
                    self.merge_alis(key)
                    if autosave: self.autosave(key, axisID, mouseidx)
                elif self._respreq[1] == "mst_merge":
                    if key == "y":
                        self.update_master()
                        self._fitchng = True
                    if autosave: self.autosave(key, axisID, mouseidx)
                else:
                    return
                # Switch off the required response
                self._respreq[0] = False
            # Reset the info box
            self.update_infobox(default=True)
            return

        # Used keys include:  abcdfgiklmnoprquwyz?[]<>-#
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
            print("b       : Go back (centre the mouse position in any panel as the position before moving)")
            print("i       : Obtain information on the line closest to the cursor")
            print("r       : toggle residuals plotting (i.e. remove master model from data)")
            print("------------------------------------------------------------")
        elif key == 'a' or key == 'z':
            if self._update_model is None:
                if self.lines_act.size == 0:
                    return
                cls = self.lines_act.find_closest(self.prop._wave[mouseidx], self.lines)
                self.lines_upd.add_absline_inst(self.lines_act, cls[1])
                self._update_model = [key, cls[0], cls[1]]
            elif self._update_model[0] == key:
                # Update the model
                params = [self._update_model[2], self.lines_upd.coldens[0], self.lines_upd.redshift[0],
                          self.lines_upd.bval[0], self.lines_upd.label[0], None]
                # Update the absorption line
                self.operations('ul', axisID, mouseidx, params=params)
                self.lines_upd.delete_absline_idx(0)
                self._update_model = None
                self.draw_model_update()
            else:
                # If any other key is pressed, ignore the update
                if self.lines_upd.size != 0:
                    self.lines_upd.delete_absline_idx(0)
                self._update_model = None
                self.draw_model_update()
            self._fitchng = True
        elif key == 'b':
            self.goback()
        elif key == 'c':
            self.update_actors(axisID, clear=True)
            if autosave: self.autosave('c', axisID, mouseidx)
            self._fitchng = True
        elif key == 'd':
            self.delete_line(mouseidx)
            if autosave: self.autosave('d', axisID, mouseidx)
            self._fitchng = True
        elif key == 'f':
            self.fit_alis()
            self._respreq = [True, "fit_alis"]
            self.update_infobox(message="Accept fit?", yesno=True)
        elif key == 'g':
            # TODO :: Maybe add 1-8 as possible keywords, corresponding to a goto for Ly1-Ly8?
            self.goto(mouseidx)
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
            # TODO :: undo previous operation -- maybe pickle and unpickle a quickload/quicksave?
            pass
        elif key == 'w':
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

    def autosave(self, kbID, axisID, mouseidx, params=None):
        """
        For each operation performed on the data, save information about the operation performed.
        """
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
        label = "1H_I"
        if kind == 'lya':
            # Use H I Lyman alpha
            wave0 = 1215.6701
            label = "1H_I"
        elif kind == 'metal':
            # An example metal line
            wave0 = 1670.78861
            label = "27Al_II"
        # Get a quick fit to estimate some parameters
        fitvals = self.fit_oneline(wave0, mouseidx)
        if fitvals is not None:
            coldens, zabs, bval = fitvals
            self.lines_act.add_absline(coldens, zabs, bval, label)
            self.draw_lines()
            self.draw_model()
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
        # First store the data that will be needed in the analysis
        lines = []
        for axisID in range(self.naxis):
            # Find all regions
            regwhr = np.copy(self.actors[axisID] == 1)
            # Fudge to get the leftmost pixel shaded in too
            lpix = np.where((self.actors[axisID][:-1] == 0) & (self.actors[axisID][1:] == 1))
            regwhr[lpix] = True
            if np.sum(regwhr != 0):
                # Store this as a line that needs to be included in the fit
                lines += [axisID]
                # Find the endpoints
                rpixend = np.max(np.where((self.actors[axisID][:-1] == 1) & (self.actors[axisID][1:] == 0))[0])
                lpixend = np.min(lpix[0])
                # Extra the data to be fitted
                wave = self.prop._wave[lpixend - nextra - 1:rpixend + nextra]
                flux = self.prop._flux[lpixend - nextra - 1:rpixend + nextra]
                flue = self.prop._flue[lpixend - nextra - 1:rpixend + nextra]
                # Extra the regions of the data to be fitted
                fitr = np.zeros(self.prop._wave.size)
                fitr[np.where(regwhr)] = 1
                fito = fitr[lpixend - nextra - 1:rpixend + nextra]
                # Get the continuum
                cont = self.model_mst[lpixend - nextra - 1:rpixend + nextra]
                # Save the temporary data
                if not os.path.exists("tempdata"):
                    os.mkdir("tempdata")
                np.savetxt("tempdata/data_{0:02d}.dat".format(axisID), np.transpose((wave, flux, flue, fito, cont)))
            elif axisID == 0:
                self.update_infobox("You must include fitting data in the top left (i.e. Lya) panel", yesno=False)
                return
        # Make sure there are data to be fit!
        if len(lines) == 0:
            return
        # Get the ALIS parameter, data, and model lines
        parlines = self.lines_act.alis_parlines(name=self.prop._qsoname)
        datlines = self.lines_act.alis_datlines(lines)
        modlines = self.lines_act.alis_modlines(lines)
        # Some testing to check the ALIS file is being constructed correctly
        writefile = False
        if writefile:
            fil = open("testfile.mod", 'w')
            fil.write("\n".join(parlines) + "\n\n")
            fil.write("data read\n" + "\n".join(datlines) + "\ndata end\n\n")
            fil.write("model read\n" + "\n".join(modlines) + "\nmodel end\n")
            fil.close()
        # Run the fit with ALIS
        result = alis(parlines=parlines, datlines=datlines, modlines=modlines)
        # Convert the results into an easy read format
        fres = result._fitresults
        info = [(result._tend - result._tstart)/3600.0, fres.fnorm, fres.dof, fres.niter, fres.status]
        alis_lines = get_alis_string(result, fres.params, fres.perror, info,
                                     printout=False, getlines=True, save=False)
        #print(alis_lines)
        #print(result._contfinal)

        # Extract parameter values and continuum
        self.lines_upd.clear()   # Start by clearing the update lines, just in case.
        # Scan through the model to find the velocity shifts of each portion of spectrum
        vshift, vshift_err = [], []
        flag = False
        for ll in range(len(lines)):
            if lines[ll] == 0:
                vshift += [0.0]
                vshift_err += [0.0]
            else:
                shtxt = "0.0shift{0:02d}".format(lines[ll])
                # Search through alis_lines for the appropriate string
                alspl = alis_lines.split("\n")
                for spl in range(len(alspl)):
                    if "# Shift Models:" in alspl[spl]:
                        flag = True
                        continue
                    if flag:
                        cntr = 0
                        if shtxt in alspl[spl]:
                            vspl = alspl[spl].split()[2]
                            if cntr == 0:
                                vshift += [float(vspl.split("s")[0])]
                                cntr += 1
                            elif cntr == 1:
                                vshift_err += [float(vspl.split("s")[0])]
                                break
        # Extract absorption line parameters and their errors

        # TODO :: Update how the master absorption lines are stored.
        # We either need to append directly to file here all of the results from the ALIS fitting (not ideal,
        # because the master model won't be plotted correctly), or we need to change how master absorption lines
        # are stored. In the latter case, we will need to individually store Lya, Lyb etc. from the same system.

        # Plot best-fitting model and residuals (send to lines_upd)

        # Clean up (delete the data used in the fitting)
        rmtree("tempdata")
        return

    def merge_alis(self, resp):
        # If the user is happy with the ALIS fit, update the actors with the new model parameters
        # TODO :: complete the two options (y or n) here
        if resp == "y":
#            for ll in range(self.lines_upd.size):
#                errs = [err_coldens, err_redshift, err_bval]
#                params = [INDEX, coldens, redshift, bval, label, errs=errs]
#                self.operations('ul', -1, -1, params=params)
            self._fitchng = False
        else:
            self.lines_upd.clear()
            pass

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
            flxfit = self.prop._flux / (self.model_mst*self.model_act)
            flefit = self.prop._flue / (self.model_mst*self.model_act)
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

    def goto(self, mouseidx):
        # Store the old values
        self._xmnsv, self._xmxsv = self.axs[0].get_xlim()
        # Set the new values
        lam = self.lines[0]*(1.0+self.prop._zem)
        vnew = 299792.458*(self.prop._wave[mouseidx]-lam)/lam
        xmn, xmx = vnew-self.veld, vnew+self.veld
        ymn, ymx = self.axs[0].get_ylim()
        for i in range(len(self.lines)):
            self.axs[i].set_xlim([xmn, xmx])
            self.axs[i].set_ylim([ymn, ymx])
        self.canvas.draw()

    def goback(self):
        # Go back to the old x coordinate values
        xmn, xmx = self._xmnsv, self._xmxsv
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

    def toggle_residuals(self, resid):
        for i in range(self.naxis):
            if resid:
                # If residuals are currently being shown, plot the original data
                self.specs[i].set_ydata(self.prop._flux)
            else:
                # Otherwise, plot the residuals
                self.specs[i].set_ydata(self.prop._flux/(self.model_mst*self.model_act))
        self.canvas.draw()
        self.canvas.flush_events()
        self._resid = not resid

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
        self.canvas.draw()

    def update_master(self):
        # Store the master regions and lines.
        self.update_master_regions()
        self.update_master_lines()
        # Merge models and reset
        self.model_mst *= self.model_act
        self.model_act[:] = 1.0
        # Clear the actors
        self.update_actors(None, clear=True)
        # Update the plotted lines
        self.draw_lines()
        self.draw_model()
        self.canvas.draw()
        return

    def update_master_regions(self):
        for ii in range(self.naxis):
            ww = np.where(self.actors[ii] == 1)
            self.prop._regions[ww] = 1

    def update_master_lines(self):
        for ii in range(self.lines_act.size):
            self.lines_mst.add_absline_inst(self.lines_act, 0)
            self.lines_act.delete_absline_idx(0)

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
        self.lines_mst.write_data()
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

    def alis_datlines(self, lines, res=7.0):
        """
        lines = list of integers indicating which lines are used in the fitting.
        res = instrument resolution in km/s
        """
        datlines = []
        for line in lines:
            if line == 0:
                shtxt = "0.0SFIX"
            else:
                shtxt = "0.0shift{0:02d}".format(line)
            datlines += ["tempdata/data_{0:02d}.dat  specid=line{0:02d}  fitrange=columns  loadrange=all  resolution=vfwhm({1:.3f}RFIX) shift=vshift({2:s}) columns=[wave:0,flux:1,error:2,fitrange:3,continuum:4]".format(line, res, shtxt)]
        return datlines

    def alis_modlines(self, lines):
        modlines = []
        # Do the emission
        modlines += ["emission"]
        for axID in lines:
            modlines += ["legendre  1.0 0.0 0.0 0.0  scale=1.0,1.0,1.0,1.0  specid=line{0:02d}".format(axID)]
        # Do the absorption
        modlines += ["absorption"]
        for ll in range(self.size):
            for axID in lines:
                modlines += ["voigt ion={0:s}  {1:.4f}n{4:d}  {2:.9f}z{4:d}  {3:.3f}b{4:d}  0.0TFIX  specid=line{5:02d}".format(
                    self.label[ll], self.coldens[ll], self.redshift[ll], self.bval[ll], ll, axID)]
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
            parlines += ["out verbose 0"]
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

    def add_absline(self, coldens, redshift, bval, label):
        # Add the values
        self.coldens = np.append(self.coldens, coldens)
        self.redshift = np.append(self.redshift, redshift)
        self.bval = np.append(self.bval, bval)
        # Add the errors
        self.err_coldens = np.append(self.err_coldens, coldens)
        self.err_redshift = np.append(self.err_redshift, redshift)
        self.err_bval = np.append(self.err_bval, bval)
        # Append the label
        self.label += [label]
        return

    def add_absline_inst(self, inst, idx):
        self.add_absline(inst.coldens[idx], inst.redshift[idx], inst.bval[idx], inst.label[idx])
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
        # A label
        self.label = []
        return

    def delete_absline(self, west, lines):
        """
        west : The estimated wavelength of the line to be deleted
        """
        # Find the line to be deleted
        lidx, widx = self.find_closest(west, lines)
        # Delete the values
        self.coldens = np.delete(self.coldens, (widx,), axis=0)
        self.redshift = np.delete(self.redshift, (widx,), axis=0)
        self.bval = np.delete(self.bval, (widx,), axis=0)
        # Delete the errors
        self.err_coldens = np.delete(self.err_coldens, (widx,), axis=0)
        self.err_redshift = np.delete(self.err_redshift, (widx,), axis=0)
        self.err_bval = np.delete(self.err_bval, (widx,), axis=0)
        # Delete the label
        del self.label[widx]
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
        # Delete the label
        del self.label[idx]
        return

    def find_closest(self, west, lines):
        """
        west : The estimated wavelength of the line to be deleted
        """
        amin = np.argmin(np.abs(np.outer(lines, 1 + self.redshift) - west))
        idx = np.unravel_index(amin, (lines.size, self.redshift.size))
        lidx, widx = int(idx[0]), int(idx[1])
        return lidx, widx

    def getabs(self, idx, errs=False):
        if errs:
            return self.coldens[idx], self.redshift[idx], self.bval[idx],\
                   self.err_coldens[idx], self.err_redshift[idx], self.err_bval[idx],\
                   self.label[idx]
        else:
            return self.coldens[idx], self.redshift[idx], self.bval[idx], self.label[idx]

    def lineinfo(self, west, lines):
        """
        west : The estimated wavelength of the line to be deleted
        """
        lidx, widx = self.find_closest(west, lines)
        print("=======================================================")
        print("Line Information:")
        print("  {0:s} {1:.2f}".format(self.label[widx], lines[lidx]))
        print("  Col Dens = {0:.3f} +/- {1:.3f}".format(self.coldens[widx], self.err_coldens[widx]))
        print("  Redshift = {0:.6f} +/- {1:.6f}".format(self.redshift[widx], self.err_redshift[widx]))
        print("  Dopp Par = {0:.2f} +/- {1:.2f}".format(self.bval[widx], self.err_bval[widx]))
        return

    def update_absline(self, indx, coldens=None, redshift=None, bval=None, errs=None, label=None):
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
        if label is not None:
            self.label[indx] = label
        return

    def write_data(self):
        # TODO :: Save the master absorption line infomation
        # The information that should be saved includes:
        # A header that lists the properties of the spectrum used in the analysis
        #  - QSO name
        #  - OBS date/time
        #  - spectral resolution
        #  - Maybe include the entire header from the original fits file?
        # Then, the table data will include:
        #  - Line ID number?
        #  - Approximate central wavelength of the line
        #  - Line ID (i.e. H I Lya, H I Lyb, metal)
        #  - Column density and error
        #  - Bval and error
        #  - Redshift and error
        pass


class Atomic:
    def __init__(self, filename="/Users/rcooke/Software/ALIS_dataprep/atomic.dat", wmin=None, wmax=None):
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

    prop = Props(QSO("HS1700p6416"))

    # Ignore lines outside of wavelength range
    wmin, wmax = np.min(prop._wave)/(1.0+prop._zem), np.max(prop._wave)/(1.0+prop._zem)

    # Load atomic data
    atom = Atomic(wmin=wmin, wmax=wmax)

    spec = Line2D(prop._wave, prop._flux, linewidth=1, linestyle='solid', color='k', drawstyle='steps', animated=True)

    fig, ax = plt.subplots(figsize=(16,9), facecolor="white")
    ax.add_line(spec)
    reg = SelectRegions(fig.canvas, ax, spec, prop, atom)

    ax.set_title("Press '?' to list the available options")
    #ax.set_xlim((prop._wave.min(), prop._wave.max()))
    ax.set_ylim((0.0, prop._flux.max()))
    plt.show()
