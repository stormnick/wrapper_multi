import sys
import os
import numpy as np
from atom_package import model_atom

def distribute_jobs(self, atmos_list = None, ncpu=1):
    """
    Distributing model atmospheres over a number of processes
    input:
    (array) atmos_list: contains all model atmospheres requested for the run
    (integer) ncpu: number of CPUs to use
    """

    atmos_list = self.atmos
    ncpu = self.ncpu

    """
    abundance dimension:
    every NLTE run with M1D has unique model atmospehere,
            model atom and abundance of the NLTE element
    assuming one run is set up for one NLTE element,
    one needs to iterate over model atmospheres and abundances
    """
    if self.use_abund == 0:
        abund_list = [self.atom.abund]
    elif self.use_abund == 1:
        abund_list = [self.new_abund]
    elif self.use_abund == 2:
        abund_list = np.arange(self.start_abund, self.end_abund, self.step_abund)
        # [start, end) --> [start, end]
        abund_list = np.hstack((abund_list,  self.end_abund ))
    else:
        print("Unrecognized use_abund=%.f" %(self.use_abund))
        exit(1)

    totn_jobs = len(atmos_list) * len(abund_list)


    atmos_list, abund_list= np.meshgrid(atmos_list, abund_list)
    atmos_list = atmos_list.flatten()
    abund_list = abund_list.flatten()


    if (ncpu > totn_jobs):
        print("Requested more CPUs (%.0f) than model atmospheres provided (%.0f)" %( ncpu, totn_jobs ) )
        exit(1)

    step = totn_jobs//ncpu
    self.jobs = { }
    for i in range(ncpu-1):
        job = serial_job(self, i)
        job.atmos = atmos_list[step*i: step*(i+1)]
        job.abund = abund_list[step*i: step*(i+1)]
        self.jobs.update({ i : job })
    job = serial_job(self, ncpu-1)
    job.atmos = atmos_list[step*i : ]
    job.abund = abund_list[step*i : ]
    self.jobs.update({ ncpu-1 : job })

    return

class serial_job(object):
    def __init__(self, parent, i):
        self.id = i
        self.common_wd = parent.common_wd
        self.tmp_wd = parent.common_wd + '/job_%03d/' %(self.id)
        self.atmos = []
        self.abudn = []

# a setup of the run to compute NLTE grid, e.g. Mg over all MARCS grid
class setup(object):
    def __init__(self, file='config.txt'):#, lines=None, conts=None, output=True, ratmos=False, rpop=False):
        """
        Reads in specifications for the future parallel jobs from a configuration file
        input:
        (string) file: filename for the config file  (default: 'config.txt')
        """

        print('Reading configuration file %s' %(file ) )
        for line in open(file, 'r'):
            line=line.strip()
            if not line.startswith("#") and line != '':

                    key, val = line.split('=')
                    key, val = key.strip(), val.strip()
                    if val.startswith("'") or val.startswith('"'):
                        self.__dict__[key] = val[1:-1]
                    elif val.startswith("["):
                        self.__dict__[key] = eval('np.array(' + val + ')')
                    elif '.' in val:
                        self.__dict__[key] = float(val)
                    else:
                        self.__dict__[key] = int(val)

        """
        Make sure to have an absolute path to the common wd
        common wd is defined in the config. file, otherwise set to ./)
        """
        if 'common_wd' not in self.__dict__.keys():
            print("Setting common working directory to ./")
            self.common_wd = os.getcwd()
        else:
            if self.common_wd == './':
                self.common_wd = os.getcwd()
            self.common_wd = self.common_wd + '/'

        """
        Read *a list* of all requested model atmospheres
        Add a path to the filenames
        Model atmospheres themselves are not read here,
        as the parallel worker will iterate over them
        """
        print("Reading a list of model atmospheres from %s" %( self.atmos_list ))
        atmos_list = np.loadtxt( self.atmos_list, dtype=str, ndmin=1)
        self.atmos = []
        for atm in atmos_list:
            self.atmos.append( self.atmos_path + './' + atm )

        """
        Read model atom
        As each run is intended to include one element at a time,
        *model atom can be read only once* at the beginning
        Object model_atom is then passed to run_job() inside parallel worker,
        where abundance will be modififed if needed
        and model atom will be written in a temporary M1D formatted input file ATOM
         """
        print("Reading model atom from %s" %self.atom_path)
        self.atom = model_atom(self.atom_path + '/atom.' + self.atom_id)
        # M1D input file that comes with model atom
        self.m1d_input_file = self.atom_path + '/input.' + self.atom_id

        print(50*"-")
        print("Distributing model atmospheres over %.0f CPUs" %(self.ncpu) )
        distribute_jobs(self, self.atmos, ncpu=self.ncpu)
