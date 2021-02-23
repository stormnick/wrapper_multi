import multiprocessing
import sys
import os
import shutil
from atom_package import model_atom, write_atom
from atmos_package import model_atmosphere, write_atmos_m1d, write_dscale_m1d


def mkdir(s):
    if (not os.path.isdir(s)):
        os.mkdir(s)
    return

def setup_multi_job(setup, job):
    """
    Setting up and running an individual serial job of NLTE calculations
    for a set of model atmospheres (stored in setup.jobs[k]['atmos'])
    Several indidvidual jobs can be run in parallel, set ncpu in the config. file
    to the desired number of processes
    Note: Multi 1D expects all input files to be named only in upper case

    input:
    (string) directory: common working directory, default: "./"
    (integer) k: an ID of an individual job within the run
    (object) setup: object of class setup, regulates a setup for the whole run
    """

    """ Make and change to a temporary directory """
    tmp_wd = setup.common_wd + '/job_%03d/' %(job['id'])
    mkdir(tmp_wd)
    job.update({'wd':tmp_wd})
    # os.chdir(tmp_wd)

    """ Link input files to a temporary directory """
    for file in ['absmet', 'abslin', 'abund', 'absdat']:
        os.symlink( setup.common_wd + setup.m1d_input + '/' + file, tmp_wd + file.upper() )


def run_multi( wd, atom, atmos, input_path ):
    """
    Run MULTI1D
    """

    """ Create ATOM input file for M1D """
    write_atom(atom, wd +  '/ATOM' )

    """ Create ATMOS input file for M1D """
    write_atmos_m1d(atmos, wd +  '/ATMOS' )
    write_dscale_m1d(atmos, wd +  '/DSCALE' )

    """ Link INPUT file """
    os.symlink( input_path, wd +  '/INPUT' )

    # nohup mul23lus.x
    return

def read_m1d_output():
    return



def run_job(setup, job):
        setup_multi_job( setup, job )
        print("job # %5.0f: %5.0f M1D runs" %( job['id'], len(job['atmos']) ) )
        for i in range(len(job['atmos'])):
            atom = setup.atom
            atom.abund  =  job['abund'][i]
            atmos = model_atmosphere(file = job['atmos'][i], format = setup.atmos_format)
            run_multi( job['wd'], atom, atmos, setup.m1d_input_path)
            # read output
        # shutil.rmtree(job['wd'])



    # """ Start individual jobs """
    # workers = []
    # for k in set.jobs.keys():
    #     p = multiprocessing.Process( target=run_multi( k, set ) )
    #     workers.append(p)
    #     p.start()
