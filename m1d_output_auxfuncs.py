import numpy as np
from scipy.fft import fft, ifft

def lobatto(ntheta, nphi):
    nrays = ntheta + 1
    nmuphi = ntheta * nphi + 1

    pi = np.pi
    pi2 = 2*pi

    if nrays == 1:
        mu = 1
        wts = pi
        phi = 0

    else:
        if nrays == 2:
            mu = np.array([1, 0.44721360])
            wts = np.array([0.16666667, 0.83333333]) * mu * pi2

        elif nrays == 3:
            mu = np.array([1.0, 0.76505532, 0.28523152])
            wts = np.array([0.06666667, 0.37847496, 0.55485838]) * mu * pi2

        elif nrays == 4:
            mu = np.array([1.0, 0.87174015, 0.59170018, 0.20929922])
            wts = np.array([0.03571428, 0.21070422, 0.34112270, 0.41245880]) * mu * pi2

        elif nrays == 5:
            mu = np.array([1.0, 0.9195339082, 0.7387738651, 0.4779249498, 0.1652789577])
            wts = np.array([0.0222222222, 0.1333059908, 0.2248893420, 0.2920426836, 0.3275397612]) * mu * pi2

        elif nrays == 6:
            mu = np.array([1.0, 0.944899272223, 0.819279321644, 0.632876153032, 0.399530940965, 0.136552932855])
            wts = np.array([0.015151515153, 0.091684517440, 0.157974705655, 0.212508417834, 0.251275603210, 0.271405240910]) * mu * pi2

        elif nrays > 6:
            print('ntheta set too high!')

        phi = pi2 / nphi * np.arange(nphi)
        wts[1:nrays-1] = wts[1:nrays-1] / nphi

    mux  = np.empty(nmuphi)
    muy  = np.empty(nmuphi)
    muz  = np.empty(nmuphi)
    xphi = np.empty(nmuphi)
    xmu  = np.empty(nmuphi)
    wmu  = np.empty(nmuphi)

    wmu[0] = wts[0]
    xmu[0] =  mu[0]

    for nt in range(1, ntheta + 1):
        for nph in range(nphi):
            i = (nt-1) * nphi + nph + 1
            xmu[i]  =  mu[nt]
            xphi[i] = phi[nph]
            wmu[i]  = wts[nt]

    theta = np.arccos(xmu)

    mux = np.sin(theta) * np.cos(xphi)
    muy = np.sin(theta) * np.sin(xphi)
    muz = xmu

    for imu in  range(nmuphi):
        if (abs(mux[imu]) < 1e-3): mux[imu] = 0
        if (abs(muy[imu]) < 1e-3): muy[imu] = 0
        if (abs(muz[imu]) < 1e-3): muz[imu] = 0

    return mux, muy, muz, xphi, xmu, wmu



def vacuum2obs(vac):
    #
    # Converts vacuum wavelength to observed wavelengths
    #
    convf = 1 + 2.735182e-4 + 131.4182 / vac.astype(np.float64)**2 + 2.76249e8 / vac.astype(np.float64)**4
    obs = vac / convf

    return obs


def conv_profile(xx, yy, px, py):
    norm = np.trapz(py, x=px)

    n = len(xx)

    dxn = (xx[-1] - xx[0]) / (n - 1)

    conv = dxn * ifft(fft(yy)*fft(np.roll(py/norm, int(n/2))))

    return xx, np.real(conv)


def convol(sx, sy, vrot=None, zeta_rt=None, vshift=None):
    beta = 1.5
    cc   = 299792.458 # VELOCITY OF LIGHT (KM/S)
    sxx  = np.log(sx.astype(np.float64)) # original xscale in
    syy  =        sy.astype(np.float64)

    min_rd = 0.1 / cc    # RESAMPLING-DISTANCE

    rd = 0.5 * np.min(np.diff(sxx))
    rd = np.max([rd, min_rd])


    npres = ((sxx[-1] - sxx[0]) // rd) + 1
    npresn = npres + npres%2

    rd  = (sxx[-1] - sxx[0]) / (npresn-1)

    sxn = sxx[0] + np.arange(npresn) * rd

    syn = np.interp(sxn, sxx, syy)

    px = (np.arange(npresn) - npresn//2) * rd


    if vrot is not None:
        normf = cc/vrot
        xi = normf*px

        xi[abs(xi) > 1] = 1

        py = (2*np.sqrt(1-xi**2) / np.pi + beta*(1-xi**2) / 2) * normf / (1 + 6/9*beta)

        sxn, syn = conv_profile(sxn, syn, px, py)

    if zeta_rt is not None:
        wave_rt = np.arange(20)/10.
        flux_rt = [1.128,0.939,0.773,0.628,0.504,0.399,0.312,0.240,0.182,0.133,
                  0.101,0.070,0.052,0.037,0.024,0.017,0.012,0.010,0.009,0.007]
        wave_rt = np.concatenate([-wave_rt[:0:-1],wave_rt])
        flux_rt = np.concatenate([ flux_rt[:0:-1],flux_rt])
        zeta_rt1 = zeta_rt/cc
        wave_rt = wave_rt * zeta_rt1
        flux_rt = flux_rt / zeta_rt1
        py = np.interp(px, wave_rt, flux_rt)
        mask = (px < wave_rt[0]) + (px > wave_rt[-1])
        py[mask] = 0

        sxn, syn = conv_profile(sxn, syn, px, py)

    xx = np.exp(sxn)
    yy = syn

    if vshift is not None:
        vshift = vshift / cc # vshift has to be given in km/s
        lshift = xx * vshift / (1+vshift)

        xx += lshift


    return xx, yy
