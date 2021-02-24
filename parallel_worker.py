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

    """ Make and a temporary directory """
    job.update({'common_wd':setup.common_wd})
    tmp_wd = setup.common_wd + '/job_%03d/' %(job['id'])
    mkdir(tmp_wd)
    job.update({'tmp_wd':tmp_wd})

    """ Link input files to a temporary directory """
    for file in ['absmet', 'abslin', 'abund', 'absdat']:
        os.symlink( setup.m1d_input + '/' + file, tmp_wd + file.upper() )

    """ Link INPUT file (M1D input file complimenting the model atom) """
    os.symlink( setup.m1d_input_file, tmp_wd +  '/INPUT' )

    """ Link executable """
    os.symlink(setup.m1d_exe, tmp_wd + 'multi1d.exe')


def run_multi( job, atom, atmos):
    """
    Run MULTI1D
    input:
    (string) wd: path to a temporary working directory,
        created in setup_multi_job
    (object) atom:  object of class model_atom
    (object) atmos: object of class model_atmosphere
    """

    """ Create ATOM input file for M1D """
    write_atom(atom, job['tmp_wd'] +  '/ATOM' )

    """ Create ATMOS input file for M1D """
    write_atmos_m1d(atmos, job['tmp_wd'] +  '/ATMOS' )
    write_dscale_m1d(atmos, job['tmp_wd'] +  '/DSCALE' )

    """ Got to directory and run MULTI 1D """
    os.chdir(job['tmp_wd'])
    # nohup multi1d.exe
    print("Hi there")
    os.chdir(job['common_wd'])



    # nohup mul23lus.x
    return

def read_m1d_output():
    return



def run_job(setup, job):
        setup_multi_job( setup, job )
        print("job # %5.0f: %5.0f M1D runs" %( job['id'], len(job['atmos']) ) )
        for i in range(len(job['atmos'])):
            # model atom is only read once
            atom = setup.atom
            atom.abund  =  job['abund'][i]
            atmos = model_atmosphere(file = job['atmos'][i], format = setup.atmos_format)
            run_multi( job, atom, atmos)
            # read output
        # shutil.rmtree(job['wd'])



    # """ Start individual jobs """
    # workers = []
    # for k in set.jobs.keys():
    #     p = multiprocessing.Process( target=run_multi( k, set ) )
    #     workers.append(p)
    #     p.start()
