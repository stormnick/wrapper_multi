import numpy as np
from copy import deepcopy
from astropy import constants as const

"""
    Read and manipulate model atmospheres
"""


periodic_table_element_names = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U"]



def write_atmos_m1d(atmos, file):
    """
    Write model atmosphere in MULTI 1D input format, i.e. atmos.*
    input:
    (object of class model_atmosphere): atmos
    (string) file: path to output file
    """

    with open(file, 'w') as f:
        header = f"\
* {atmos.header.strip()}\n\
{atmos.id} \n\
* Depth scale: log(tau500nm) (T), log(column mass) (M), height [km] (H)\n\
{atmos.depth_scale_type} \n\
*log(g) \n {atmos.logg:.3f} \n\
*Teff = {atmos.teff}\n\
*Number of depth points \n {atmos.ndep:.0f}\n\
*depth scale, temperature, N_e, Vmac, Vturb \n\
        "
        f.write(header)
        # write physical values
        for i in range(len(atmos.depth_scale)):
            f.write('    '.join(f"{atmos.__dict__[k][i]:15.5e}" for k in ['depth_scale', 'temp', 'ne', 'vmac', 'vturb'] ) + '\n')

def write_atmos_m1d4TS(atmos, file):
    """
    Write model atmosphere in MULTI 1D input format, i.e. atmos.*
    but very specific so that TS recognises it
    input:
    (object of class model_atmosphere): atmos
    (string) file: path to output file
    """

    with open(file, 'w') as f:
        header = f"\
{atmos.id}\n\
{atmos.depth_scale_type}\n\
* LOG (G) \n {atmos.logg} \n\
* NDEP \n {atmos.ndep} \n\
* depth scale, temperature, N_e, Vmac, Vturb \n\
        "
        f.write(header)
        # write structure
        for i in range(len(atmos.depth_scale)):
            f.write('    '.join(f"{atmos.__dict__[k][i]:15.5e}" for k in ['depth_scale', 'temp', 'ne', 'vmac', 'vturb'] ) + '\n')

def write_dscale_m1d(atmos, file):
    """
    Write MULTI1D DSCALE input file with depth scale to be used for NLTE computations
    """
    with open(file, 'w') as f:
        header = f"\
{atmos.id} \n\
* Depth scale: log(tau500nm) (T), log(column mass) (M), height [km] (H) \n {atmos.depth_scale_type}\n\
* Number of depth points, top point \n {atmos.ndep:.0f} {atmos.depth_scale[0]:10.5e}\n\
"
        # write formatted header
        f.write(header)
        # write structure
        for i in range(len(atmos.depth_scale)):
            f.write(f"{atmos.depth_scale[i]:15.5e}\n" )


class ModelAtmosphere:
    def __init__(self, file=None, file_format=None):
        self.atmospheric_abundance: dict = None
        self.ndep = None
        self.file: str = file
        self.file_format: str = file_format


    def read(self, file, file_format):
        """
        Model atmosphere for NLTE calculations
        input:
        (string) file: file with model atmosphere, default: atmos.sun
        (string) format: m1d, marcs, stagger, see function calls below
        """
        if file_format.lower() == 'marcs':
            self.read_atmos_marcs(file)
            #print(f"Setting depth scale to tau500 from the model {file}")
            self.depth_scale_type = 'TAU500'
            self.depth_scale = self.tau500
        elif file_format.lower() == 'm1d':
            self.read_atmos_m1d(file)
            try:
                feh = float(self.id.split('_z')[-1].split('_a')[0])
                alpha = float(self.id.split('_a')[-1].split('_c')[0])
                self.feh = feh
                self.alpha = alpha
                print(f"Guessed [Fe/H]={self.feh}, [alpha/Fe]={self.alpha}")
            except:
                try:
                    feh = float(self.id.split('m')[-1].split('_')[0])
                    self.feh = feh
                    self.alpha = self.feh
                except:
                    self.feh = np.nan
                    self.alpha=np.nan
                    print("WARNING: [Fe/H] and [alpha/Fe] are unknown from the model atmosphere")
                    exit()
        elif file_format.lower() == 'stagger':
            self.read_atmos_m1d(file)
            print(f"Guessing [Fe/H] and [alpha/Fe] from the file name {self.id}..")
            teff = float(self.id.split('g')[0].replace('t',''))
            if teff != 5777:
                teff = teff*1e2
            feh = float(self.id[-2:]) /10
            if self.id[-3] == 'm':
                feh = feh * (-1)
            elif self.id[-3] == 'p':
                pass
            else:
                print("WARNING: [Fe/H] and [alpha/Fe] are unknown from the model atmosphere")
                exit()
            self.feh = feh
            self.alpha = self.feh
            self.teff = teff
            print(f"Guessed [Fe/H]={self.feh}, [alpha/Fe]={self.alpha}")
        else:
            raise Warning("Unrecognized format of model atmosphere: {format}. Supported formats: 'marcs' (*.mod), 'm1d' (atmos.*), or 'stagger' for stagger average 3D formatted for MULTI1D.)")

    def FillIn(self):
        for k in ['logg', 'teff', 'ndep']:
            if k not in self.__dict__.keys():
                self.__dict__[k] = np.nan
        if 'header' not in self.__dict__.keys():
            self.header  = ''
        if 'vmac' not in self.__dict__.keys():
            self.vmac = np.zeros( len(self.depth_scale ))


    def copy(self):
        return deepcopy(self)

    def write(self, path, format = 'm1d'):
        self.FillIn()
        if format == 'm1d':
            write_atmos_m1d(self, path)
        elif format == 'ts':
            write_atmos_m1d4TS(self, path)
        else:
            raise Warning(f"Format {format} not supported for writing yet.")

    def read_atmos_marcs(self, file):
        """
        Read model atmosphere in standart MARCS format i.e. *.mod
        input:
        (string) file: path to model atmosphere
        """
        # Boltzmann constant

        data = [l.strip() for l in open(file, 'r').readlines() if not l.startswith('*') or l == '']
        # MARCS model atmospheres are by default strictly formatted
        self.id = data[0]
        if self.id.startswith('p'):
            self.pp = True
            self.spherical = False
        elif self.id.startswith('s'):
            self.spherical = True
            self.pp = False
        else:
            print(f"Could not understand model atmosphere geometry (spherical/plane-parallel) for {file}")
        self.teff = float(data[1].split()[0])
        self.flux = float(data[2].split()[0])
        self.logg = np.log10(float(data[3].split()[0]))
        self.vturb = float(data[4].split()[0])
        self.mass = float(data[5].split()[0])
        self.feh, self.alpha = np.array(data[6].split()[:2]).astype(float)
        self.X, self.Y, self.Z = np.array(data[10].split()[:3]).astype(float)

        atmospheric_abundance = {}
        elemental_number = 0

        for line in data[12:22]:
            for elemental_abundance in np.array(line.split()).astype(float):
                atmospheric_abundance[periodic_table_element_names[elemental_number].upper()] = elemental_abundance
                elemental_number += 1
        self.atmospheric_abundance = atmospheric_abundance

        # read structure
        for line in data:
            if 'Number of depth points' in line:
                self.ndep = int(line.split()[0])
        for key in ['k', 'tau500', 'height', 'temp', 'pe', 'ne']:
            self.__dict__[key] = np.full(self.ndep, np.nan)
        self.k, self.tau500, self.height, self.temp, self.pe = np.loadtxt(data[25:25 + self.ndep], unpack=True,
                                                                          usecols=(0, 2, 3, 4, 5))
        self.ne = self.pe / self.temp / const.k_B.cgs.value
        self.vturb = np.full(self.ndep, self.vturb)
        self.vmac = np.zeros(self.ndep)
        # add comments
        self.header = f"Converted from MARCS formatted model atmosphere {self.id}"

    def read_atmos_m1d(self, file):
        """
        Read model atmosphere in MULTI 1D input format, i.e. atmos.*
        M1D input model atmosphere is strictly formatted
        input:
        (string) file: path to model atmosphere file
        """
        data = [l.strip() for l in open(file, 'r').readlines() if not l.startswith('*') or l == '']
        for l in data:
            if 'Teff' in l:
                self.teff = float(l.split()[-1].split('=')[-1])
                break
        # read header
        self.id = data[0]
        self.depth_scale_type = data[1]
        self.logg = float(data[2])
        self.ndep = int(data[3])
        # read structure
        for k in ['depth_scale', 'temp', 'ne', 'vmac', 'vturb']:
            self.__dict__[k] = np.full(self.ndep, np.nan)
        self.depth_scale, self.temp, self.ne, self.vmac, self.vturb = np.loadtxt(data[4:], unpack=True)

        # info that's not provided in the model atmosphere file:
        if not 'teff' in self.__dict__.keys():
            self.teff = np.nan
        self.X = np.nan
        self.Y = np.nan
        self.Z = np.nan
        self.mass = np.nan
        # add comments here
        self.header = "Read from M1D formatted model atmosphere {self.id}"
        return

