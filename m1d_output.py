import numpy as np
from m1d_output_auxfuncs import vacuum2obs
from scipy import integrate
from scipy.io import FortranFile
from scipy.signal import argrelextrema

hh = 6.626176E-27
cc = 2.99792458E+10


def read_str(f, dtype='a20'):
    if dtype == None:
        size = f._read_size(eof_ok=True)
        dtype = 'a' + str(int(size))

        data = np.fromfile(f._fp, dtype=dtype, count=1)
        waste = f._read_size(eof_ok=True)

        return str(data)

    else:
        data = f.read_record(dtype)

        if len(data) > 1:
            return np.array(list(map(lambda s: str(s).split("'")[1].rstrip(), data)))
        else:
            return str(data).split("'")[1].rstrip()


class m1d:
    def __init__(self, file, lines=None, readny=False):
        f = FortranFile(file, 'r')

        # clear_output(wait=True)

        iformat = 0

        self.ndep, self.nk, self.nline, self.nwide, self.nrad, \
            self.nrfix, self.nmu, self.mq = f.read_ints(np.int32)

        if self.ndep == 0:
            iformat = f.read_ints(np.int32)[0]

            if iformat == 2:
                self.nsel = f.read_ints(np.int32)[0]
                self.krsel = f.read_ints(np.int32)

        if self.nrad > 0: self.nq = f.read_ints(np.int32)

        self.qnorm = f.read_ints(np.float32)[0]

        self.abnd, self.awgt = f.read_ints(np.float32)

        self.ev = f.read_reals(np.float32)
        self.g = f.read_reals(np.float32)
        self.ion = f.read_ints(np.int32)

        self.hn3c2 = f.read_reals(np.float32)[0]

        if self.nrad > 0:
            self.ktrans = f.read_ints(np.int32)
            self.jrad = f.read_ints(np.int32)
            self.irad = f.read_ints(np.int32)
            self.f = f.read_reals(np.float32)
            self.iwide = f.read_ints(np.int32)
            self.ga = f.read_reals(np.float32)
            self.gw = f.read_reals(np.float32)
            self.gq = f.read_reals(np.float32)

        if iformat == 0:
            self.krad = f.read_ints(np.int32)

        else:
            self.krad = np.zeros([self.nrad, self.nrad])

            for kr in range(self.nrad):
                i = irad[kr] - 1
                j = jrad[kr] - 1
                self.krad[i, j] = kr + 1
                self.krad[j, i] = kr + 1

        if iformat == 0: self.z = f.read_reals(np.float32)

        if self.nwide > 0:
            self.alfac = f.read_reals(np.float32)

        self.hny4p = f.read_reals(np.float32)[0]

        if self.nrad > 0: self.alamb = f.read_reals(np.float32)
        if self.nline > 0: self.a = f.read_reals(np.float32)

        if iformat == 0:
            self.b = f.read_reals(np.float32).reshape((self.nk, self.nk), order='F')

        else:
            self.b = np.zeros([self.nk, self.nk])

            for kr in range(self.nrad):
                i = irad[kr] - 1
                j = jrad[kr] - 1
                hn3c2_line = 2. * hh * cc / (self.alamb[kr] * 1e-8) ^ 3
                b[j, i] = a[kr] / hn3c2_line
                b[i, j] = self.g[j] / self.g[i] * b[j, i]

        self.totn = f.read_reals(np.float32)

        if self.nrad > 0 and iformat == 0:
            self.bp = f.read_reals(np.float32)

        self.nstar = f.read_reals(np.float32).reshape([-1, self.ndep], order='F')
        self.n = f.read_reals(np.float32).reshape([-1, self.ndep], order='F')

        self.iformat = iformat
        if iformat == 0:
            self.c = f.read_reals(np.float32).reshape([self.nk, self.nk, self.ndep], order='F')

        if self.nrfix > 0:
            self.jfx = f.read_ints(np.int32)
            self.ifx = f.read_ints(np.int32)
            self.ipho = f.read_ints(np.int32)
            self.a0 = f.read_reals(np.float32)
            self.trad = f.read_reals(np.float32)
            self.itrad = f.read_ints(np.int32)

        self.dnyd = f.read_reals(np.float32)

        if self.nrad > 0 and iformat == 0:
            self.adamp = f.read_reals(np.float32).reshape([-1, self.ndep], order='F')

        self.label = read_str(f, dtype='a20')
        self.atomid = read_str(f, dtype='a20')
        self.crout = read_str(f, dtype='a06')

        self.grav = f.read_reals(np.float32)[0]
        self.cmass = f.read_reals(np.float32)
        self.temp = f.read_reals(np.float32)
        self.nne = f.read_reals(np.float32)
        self.vel = f.read_reals(np.float32)
        self.tau = f.read_reals(np.float32)
        self.xnorm = f.read_reals(np.float32)
        self.height = f.read_reals(np.float32)

        string_arr = read_str(f, dtype=None).split()
        self.atmosid = string_arr[0]
        self.dptype = string_arr[-1]
        self.dpid = ' '.join(string_arr[1:-1])

        self.vturb = f.read_reals(np.float32)

        if iformat == 0:
            self.bh = f.read_reals(np.float32).reshape((-1, self.ndep), order='F')

        self.nh = f.read_reals(np.float32).reshape((-1, self.ndep), order='F')

        if iformat == 0:
            self.rho = f.read_reals(np.float32)

        if self.nrad > 0:
            self.qmax = f.read_reals(np.float32)
            self.q0 = f.read_reals(np.float32)
            self.ind = f.read_ints(np.int32)
            self.diff = f.read_reals(np.float32)[0]
            self.q = f.read_reals(np.float32).reshape((self.mq, self.nrad), order='F')

            if iformat == 0:
                self.wq = f.read_reals(np.float32)

        self.wqmu = f.read_reals(np.float32)

        if self.nwide > 0:
            self.frq = f.read_reals(np.float32)

        if self.nrad > 0:
            if iformat == 0:
                self.wphi = f.read_reals(np.float32)

            self.sl = f.read_reals(np.float32)

        if self.nline > 0:
            self.weqlte = f.read_reals(np.float32)
            self.weq = f.read_reals(np.float32)

        if self.nrad > 0:
            if iformat == 0:
                self.rij = f.read_reals(np.float32).reshape((self.nrad, self.ndep))
                self.rji = f.read_reals(np.float32).reshape((self.nrad, self.ndep))

            if iformat != 2:
                self.flux = f.read_reals(np.float32).reshape((self.mq + 1, self.nrad), order='F') / (2 * np.pi)
                self.outint = f.read_reals(np.float32).reshape((self.mq + 1, self.nmu, self.nrad), order='F')

            else:
                flux = np.zeros([self.mq + 1, self.nrad])
                outint = np.zeros([self.mq + 1, self.nmu, self.nrad])

                for kr in range(nsel):
                    flux[:, self.krsel[self.kr0] - 1] = f.read_reals(np.float32)

                for kr in range(nsel):
                    outint[:, :, self.krsel[self.kr0] - 1] = f.read_reals(np.float32)

        if iformat == 0: self.cool = f.read_reals(np.float32).reshape((-1, self.ndep), order='F')

        self.xmu = f.read_reals(np.float32)
        self.wmu = f.read_reals(np.float32)

        self.mus = self.xmu

        ee, hh, cc, bk, em, uu, hce, hc2, hck, ek, pi = f.read_reals(np.float32)

        if iformat != 0:
            self.bp = np.zeros([self.ndep, self.nrad])

            l = alamb[:self.nline] * 1e-8
            k = 1.380662e-16

            for kr in range(self.nline):
                self.bp[:, kr] = 2. * hh * cc / l / l / l / (exp(hh * cc / l[kr] / k / self.temp) - 1.)

        self.line = [None] * self.nline

        if lines != None:
            for kr in lines:
                self.line[kr] = m1dline(self, kr, readny)

        else:
            for kr in range(self.nline):
                self.line[kr] = m1dline(self, kr, readny)

        if self.weq[0] == self.weqlte[0]:
            self.mode = 'LTE'
        else:
            self.mode = 'NLTE'

        self.dim = '1D'


class m1dline:
    def __init__(self, parent, kr, readny):
        self.parent = parent

        nq = parent.nq[kr]

        self.kr = kr
        self.ga = parent.ga[kr]
        self.gw = parent.gw[kr]
        self.gq = parent.gq[kr]
        self.f = parent.f[kr]
        self.a = parent.a[kr]
        self.i = parent.irad[kr]
        self.j = parent.jrad[kr]
        self.qmax = parent.qmax[kr]
        self.q0 = parent.q0[kr]
        self.ind = parent.ind[kr]
        self.nq = parent.nq[kr]
        self.iwide = parent.iwide[kr]

        self.loggf = np.log10(self.f * parent.g[self.i - 1])

        if parent.iformat == 0:
            self.b = parent.b[self.i, self.j]
            self.c = parent.c[self.i, self.j]

        self.lambda0 = parent.alamb[kr]
        self.mus = parent.xmu

        self.flux = parent.flux[1:nq + 1, kr]
        self.i3 = parent.outint[1:nq + 1, :, kr]

        qn = parent.qnorm * 1e5 / cc  # in natural units

        if self.ind == 2:
            self.q = parent.q[:nq, kr]
            self.flux = self.flux[::-1]
            self.i3 = self.i3[::-1, :]

        else:
            q2 = parent.q[:nq, kr]

            self.q = np.concatenate([-q2[:0:-1], q2])

            self.flux = np.concatenate([self.flux[:0:-1], self.flux])
            self.i3 = np.concatenate([self.i3[:0:-1, :], self.i3])

        self.dlam = -parent.alamb[kr] * self.q * qn / (1.0 + self.q * qn)
        self.laa = self.dlam + parent.alamb[kr]

        self.lam = vacuum2obs(self.laa)[::-1]
        self.lam0 = vacuum2obs(self.lambda0)

        #self.cntm = continuum(self.lam, self.flux)
        #self.nflux = self.flux / self.cntm

        #self.weq = self.calc_weq()
        #self.wi3 = self.calc_wi3()

        if readny:
            self.nyrd()

    def calc_weq(self, ang=None, qmax=None, norm=True, use_mask=False):
        """
        Calculates the equivalent width of a spectral line by integration using the Simpson method.
        If an angle index (ang) is specified, the intensity of that angle will be integrated,
        otherwise the flux spectrum is integrated.
        The considered wavelength range is defined by qmax,
        which is the distance from the line center in doppler shift units (km/s).
        By default qmax is taken from the model atom, but a smaller range can also be specified.
        If norm=True the spectrum is renormalised after constraining the wavelength range by qmax.
        If use_mask=True the profile is integrated between the two local maxima around the line center.
        """

        if qmax is None:
            qmax = self.qmax

        xx, yy = self.crop(qmax, ang=ang, norm=norm)

        if use_mask:
            ilam0 = np.argmin(abs(xx - self.lam0))
            mask = between_ext(yy, ilam0, np.greater_equal)
            xx, yy = xx[mask], yy[mask]

        return integrate.simps(1 - yy, x=xx) * 1e3

    def crop(self, qmax, ang=None, norm=True, renorm=True):
        """
        Returns wavelenghts and intensity/flux cropped to given qmax setting.
        The line profile is normalised to the continuum if norm is True.
        If the normalisation happens before or after cropping is set by renorm.
        """

        if qmax is not None:
            qmask = np.abs(self.q) < qmax
            qmask = qmask[::-1]
        else:
            qmask = slice(None)

        xx = self.lam[qmask]

        if ang is None:
            if norm:
                if renorm:
                    yy = self.flux[qmask]
                    yy = yy / continuum(xx, yy)
                else:
                    yy = self.nflux[qmask]
            else:
                yy = self.flux[qmask]

        else:
            if norm:
                if renorm:
                    yy = self.i3[qmask, ang]
                    yy = yy / continuum(xx, yy)
                else:
                    yy = self.i3[:, ang]
                    yy = yy / continuum(self.lam, yy)
                    yy = yy[qmask]
            else:
                yy = self.i3[qmask, ang]

        return xx, yy

    def calc_wi3(self, qmax=None, norm=True, reduce=False, use_mask=False):
        """
        Calculates the equivalent width at all angles using the method calc_weq above.
        If reduce=True the result of angles sharing the same mu value are combined.
        """

        weqs = np.array([self.calc_weq(qmax=qmax, ang=i, norm=norm, use_mask=use_mask) for i in range(self.mus.size)])

        if reduce and self.parent.dim == '3D':
            mus = np.unique(self.mus)[::-1]
            wi3 = np.array([np.mean(weqs[self.mus == mu]) for mu in mus])

        else:
            wi3 = weqs

        return wi3

    def nyrd(self):
        run = self.parent

        folder = run.folder
        nyfile = '/idlny.' + run.fname
        jnyfile = '/jny.' + run.fname

        i = np.sum(run.nq[:self.kr])
        o = i + self.nq

        ny_data = np.memmap(folder + nyfile, dtype='<f4').reshape([run.ndep, 11, -1], order='F')[..., i:o]
        jny = np.memmap(folder + jnyfile, dtype='<f4').reshape([run.ndep, -1], order='F')[..., i:o].T  # mean intensity

        ny_data = np.moveaxis(ny_data, [0, 1], [2, 0])

        self.pms = ny_data[0]  # P-S
        self.iplus = ny_data[1]  # IPLUS
        self.iminus = ny_data[2]  # IMINUS
        self.p = ny_data[3]  # Feautrier mean intensity
        self.s = ny_data[4]  # Source function
        self.tauq = ny_data[5]  # Monochromatic optical depth
        self.dtauq = ny_data[6]  # dtauq(k)=tauq(k)-tauq(k-1)
        self.xcont = ny_data[7]  # continuum opacity relative to standard opacity
        self.sc = ny_data[8]  # absorption part of source function
        self.scat = ny_data[9]  # scattering part of source function
        self.x = ny_data[10]  # total opacity relative to standard opacity


def continuum(xx, yy, axis=-1):
    """
    Calculates straight line continuum between first
    and last point of xx, yy data.
    """
    yspace = yy.take([0, -1], axis)
    xspace = xx.take([0, -1], axis)

    yspace = np.moveaxis(yspace, axis, -1)

    output = LinArrayInterp(xspace, yspace, xx)

    return np.moveaxis(output, -1, axis)


def LinArrayInterp(x, y, newx):
    """
    Interpolation of 3D cube with non-uniform grid to uniform grid.
    """
    if type(y) == list:
        y = np.array(y)

    if type(x) == list:
        x = np.array(x)

    if type(newx) == list:
        newx = np.array(newx)

    if np.max(newx) > np.max(x):
        raise ValueError("Value above interpolation range")
    if np.min(newx) < np.min(x):
        raise ValueError("Value below interpolation range")

    dimnew = len(newx.shape)
    shape = x.shape[:-dimnew] + newx.shape

    dig = np.searchsorted(newx, x, side='right')
    diff = np.diff(dig, prepend=0).ravel()
    idx = np.repeat(np.arange(diff.size), diff).reshape(shape)

    _x = x.ravel()
    _y = y.reshape(y.shape[:-x.ndim] + (-1,))

    x1 = _x[idx - 1]
    x2 = _x[idx]
    y1 = _y[..., idx - 1]
    y2 = _y[..., idx]

    return (y2 - y1) / (x2 - x1) * (newx - x1) + y1


def between_ext(arr, pos, criterion=np.greater_equal):
    """
    Returns mask between two extrema identified by specified criterion
    """
    ext = argrelextrema(arr, criterion)[0]
    ext = np.pad(ext, 1, constant_values=[0, len(arr)])

    # mid = np.argmin(abs(self.lam - self.lam0))
    idx = np.searchsorted(ext, pos)
    lo, hi = ext[idx - 1], ext[idx]

    mask = np.zeros(len(arr))
    mask[lo:hi] = True

    return mask.astype(np.bool)
