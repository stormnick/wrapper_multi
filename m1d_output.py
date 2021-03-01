import numpy as np
from scipy.io import FortranFile
# from IPython.display import clear_output, display
from m1d_output_auxfuncs import vacuum2obs

hh=6.626176E-27
cc=2.99792458E+10

def read_str(f, dtype='a20'):
    if dtype==None:
        size = f._read_size(eof_ok=True)
        dtype = 'a' + str(int(size))

        data = np.fromfile(f._fp, dtype=dtype, count=1)
        waste = f._read_size(eof_ok=True)

        return str(data)

    else:
        data = f.read_record(dtype)

        if len(data) > 1:
            return np.array(list(map( lambda s: str(s).split("'")[1].rstrip(), data)))
        else:
            return str(data).split("'")[1].rstrip()

def f77_string(f, dtype='a20'):
    data = f.read_record(dtype)

    if len(data) > 1:
        return np.array(list(map( lambda s: str(s).split("'")[1].rstrip(), data)))
    else:
        return str(data).split("'")[1].rstrip()


class m1d(object):
    def __init__(self, file, lines=[]):
        f = FortranFile(file, 'r')

        # clear_output(wait=True)

        iformat = 0

        self.ndep, self.nk, self.nline, self.nwide, self.nrad, \
        self.nrfix, self.nmu, self.mq  = f.read_ints(np.int32)

        if self.ndep == 0:
            iformat = f.read_ints(np.int32)[0]

            if iformat == 2:
                self.nsel  = f.read_ints(np.int32)[0]
                self.krsel = f.read_ints(np.int32)

        if self.nrad > 0: self.nq = f.read_ints(np.int32)

        self.qnorm = f.read_ints(np.float32)[0]

        self.abnd, self.awgt = f.read_ints(np.float32)

        self.ev  = f.read_reals(np.float32)
        self.g   = f.read_reals(np.float32)
        self.ion = f.read_ints(np.int32)

        self.hn3c2 = f.read_reals(np.float32)[0]

        if self.nrad > 0:
            self.ktrans = f.read_ints(np.int32)
            self.jrad   = f.read_ints(np.int32)
            self.irad   = f.read_ints(np.int32)
            self.f      = f.read_reals(np.float32)
            self.iwide  = f.read_ints(np.int32)
            self.ga     = f.read_reals(np.float32)
            self.gw     = f.read_reals(np.float32)
            self.gq     = f.read_reals(np.float32)

        if iformat == 0:
            self.krad = f.read_ints(np.int32)

        else:
            self.krad = np.zeros([self.nrad, self.nrad])

            for kr in range(self.nrad):
                i=irad[kr]-1
                j=jrad[kr]-1
                self.krad[i,j]=kr+1
                self.krad[j,i]=kr+1


        if iformat == 0: self.z = f.read_reals(np.float32)


        if self.nwide > 0: self.alfac = f.read_reals(np.float32)

        self.hny4p = f.read_reals(np.float32)[0]

        if self.nrad  > 0: self.alamb = f.read_reals(np.float32)
        if self.nline > 0: self.a     = f.read_reals(np.float32)


        if iformat == 0:
            self.b = f.read_reals(np.float32).reshape((self.nk, self.nk), order='F')

        else:
            self.b = np.zeros([self.nk, self.nk])

            for kr in range(self.nrad):
                i=irad[kr]-1
                j=jrad[kr]-1
                hn3c2_line=2.*hh*cc/(self.alamb[kr]*1e-8)^3
                b[j,i]=a[kr]/hn3c2_line
                b[i,j]=self.g[j]/self.g[i]*b[j,i]


        self.totn = f.read_reals(np.float32)

        if self.nrad > 0 and iformat == 0: self.bp = f.read_reals(np.float32)

        self.nstar = f.read_reals(np.float32)
        self.n     = f.read_reals(np.float32)

        if iformat == 0: self.c = f.read_reals(np.float32)

        if self.nrfix > 0:
            self.jfx    = f.read_ints(np.int32)
            self.ifx    = f.read_ints(np.int32)
            self.ipho   = f.read_ints(np.int32)
            self.a0     = f.read_reals(np.float32)
            self.trad   = f.read_reals(np.float32)
            self.itrad  = f.read_ints(np.int32)

        self.dnyd = f.read_reals(np.float32)

        if self.nrad > 0 and iformat == 0: self.adamp = f.read_reals(np.float32)

        self.label  = f77_string(f, dtype='a20')
        self.atomid = f77_string(f, dtype='a20')
        self.crout  = f77_string(f, dtype='a06')

        self.grav   = f.read_reals(np.float32)[0]
        self.cmass  = f.read_reals(np.float32)
        self.temp   = f.read_reals(np.float32)
        self.nne    = f.read_reals(np.float32)
        self.vel    = f.read_reals(np.float32)
        self.tau    = f.read_reals(np.float32)
        self.xnorm  = f.read_reals(np.float32)
        self.height = f.read_reals(np.float32)

        # print(read_str(f, dtype=None).split())
        print(read_str(f, dtype=None).split())

        self.atmosid, self.dpid, self.dptype = read_str(f, dtype=None).split()

        self.vturb = f.read_reals(np.float32)

        if iformat == 0: self.bh = f.read_reals(np.float32).reshape((-1, self.ndep), order='F')

        self.nh = f.read_reals(np.float32).reshape((-1, self.ndep), order='F')

        if iformat == 0: self.rho = f.read_reals(np.float32)

        if self.nrad > 0:
            self.qmax  = f.read_reals(np.float32)
            self.q0    = f.read_reals(np.float32)
            self.ind   = f.read_ints(np.int32)
            self.diff  = f.read_reals(np.float32)[0]
            self.q     = f.read_reals(np.float32).reshape((self.mq, self.nrad), order='F')

            if iformat == 0: self.wq = f.read_reals(np.float32)

        self.wqmu = f.read_reals(np.float32)

        if self.nwide > 0: self.frq = f.read_reals(np.float32)

        if self.nrad > 0:
            if iformat == 0:
                self.wphi = f.read_reals(np.float32)

            self.sl = f.read_reals(np.float32)

        if self.nline > 0:
            self.weqlte = f.read_reals(np.float32)
            self.weq    = f.read_reals(np.float32)

        if self.nrad > 0:
            if iformat == 0:
                self.rij = f.read_reals(np.float32)
                self.rji = f.read_reals(np.float32)

            if iformat != 2:
                self.flux   = f.read_reals(np.float32).reshape((self.mq+1, self.nrad), order='F')
                self.outint = f.read_reals(np.float32).reshape((self.mq+1, self.nmu, self.nrad), order='F')

            else:
                flux   = np.zeros([self.mq + 1, self.nrad])
                outint = np.zeros([self.mq + 1, self.nmu, self.nrad])

                for kr in range(nsel):
                    flux[:,self.krsel[self.kr0]-1] = f.read_reals(np.float32)

                for kr in range(nsel):
                    outint[:,:, self.krsel[self.kr0]-1] = f.read_reals(np.float32)

        if iformat == 0: self.cool = f.read_reals(np.float32)

        self.xmu = f.read_reals(np.float32)
        self.wmu = f.read_reals(np.float32)

        ee, hh, cc, bk, em, uu, hce, hc2, hck, ek, pi = f.read_reals(np.float32)

        if iformat != 0:
            self.bp = np.zeros([self.ndep, self.nrad])

            l = alamb[:self.nline]*1e-8
            k = 1.380662e-16

            for kr in range(self.nline):
                self.bp[:, kr] = 2.*hh*cc/l/l/l/(exp(hh*cc/l[kr]/k/self.temp)-1.)


        self.line = [None] * self.nline

        if lines != None:
            for kr in range(self.nline):
                self.line[kr] = m1dline(self, kr)


        if self.weq[0] == self.weqlte[0]:
            self.mode = 'LTE'
        else:
            self.mode = 'NLTE'

        self.dim = '1D'


class m1dline(object):
    def __init__(self, parent, kr):
        nq = parent.nq[kr]

        self.ga   = parent.ga[kr]
        self.gw   = parent.gw[kr]
        self.gq   = parent.gq[kr]
        self.f    = parent.f[kr]
        self.i    = parent.irad[kr]
        self.j    = parent.jrad[kr]
        self.qmax = parent.qmax[kr]
        self.q0   = parent.q0[kr]
        self.ind  = parent.ind[kr]
        self.nq   = parent.nq[kr]

        self.lambda0 = parent.alamb[kr]

        self.kr  = kr

        self.flux = parent.flux[1:nq+1, kr]
        self.i3   = parent.outint[1:nq+1, :, kr]

        qn = parent.qnorm * 1e5/cc # in natural units


        if self.ind == 2:
            self.q = parent.q[:nq, kr]

        else:
            q2 = parent.q[:nq, kr]

            self.q = np.concatenate([-q2[:0:-1], q2])

            self.flux = np.concatenate([self.flux[:0:-1], self.flux])
            self.i3   = np.concatenate([self.i3[:0:-1, :], self.i3])

        self.dlam = -parent.alamb[kr]*self.q*qn / (1.0+self.q*qn)
        self.laa  = self.dlam + parent.alamb[kr]

        self.lam = vacuum2obs(self.laa)[::-1]
        self.lam0 = vacuum2obs(self.lambda0)

        nlam = len(self.lam)

        self.cntm = np.linspace(self.flux[0], self.flux[-1], len(self.flux))

        self.nflux = self.flux/self.cntm

        self.w3 = np.trapz(1-self.nflux, x=self.lam) * 1e3
