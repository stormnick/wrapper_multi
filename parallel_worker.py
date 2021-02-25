import multiprocessing
import sys
import subprocess as sp
import os
import shutil
import numpy as np
from atom_package import model_atom, write_atom
from atmos_package import model_atmosphere, write_atmos_m1d, write_dscale_m1d
from multi_package.m1d import m1d


def mkdir(s):
    if os.path.isdir(s):
        shutil.rmtree(s)
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
    # TODO:
    (string) directory: common working directory, default: "./"
    (integer) k: an ID of an individual job within the run
    (object) setup: object of class setup, regulates a setup for the whole run
    """

    """ Make a temporary directory """
    mkdir(job.tmp_wd)

    """ Link input files to a temporary directory """
    for file in ['absmet', 'abslin', 'abund', 'absdat']:
        os.symlink( setup.m1d_input + '/' + file, job.tmp_wd + file.upper() )

    """ Link INPUT file (M1D input file complimenting the model atom) """
    os.symlink( setup.m1d_input_file, job.tmp_wd +  '/INPUT' )

    """ Link executable """
    os.symlink(setup.m1d_exe, job.tmp_wd + 'multi1d.exe')


    """
    find a smarter way to do all of this...
    It doesn't need to be here, but..
    What kind of output from M1D should be saved?
    Read from the config file, passed here throught the object setup
    """
    job.output = { 'write_ew':setup.write_ew, 'write_profiles':setup.write_profiles, 'write_ts':setup.write_ts }
    if job.output['write_ew'] == 1 or job.output['write_ew'] == 2:
        # create file to dump output
        with open(job.tmp_wd + '/output_EW.dat', 'w') as f:
            f.write("# Lambda, temp, logg.... \n")
        job.output.update({'file_ew' : job.tmp_wd + '/output_EW.dat' } )
    elif job.output['write_ew'] == 0:
        pass
    else:
        print("write_ew flag unrecognised, stoppped")
        exit(1)

    ## departure coefficients for TS?
    # if job['output']['write_ts'] == 1:
        # f = open(job['tmp_wd'] + '/output_EW.dat', 'w')





    return


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
    write_atom(atom, job.tmp_wd +  '/ATOM' )

    """ Create ATMOS input file for M1D """
    write_atmos_m1d(atmos, job.tmp_wd +  '/ATMOS' )
    write_dscale_m1d(atmos, job.tmp_wd +  '/DSCALE' )

    """ Go to directory and run MULTI 1D """
    os.chdir(job.tmp_wd)
    sp.call(['multi1d.exe'])

    """ Read MULTI1D output and print to the common file """
    out = m1d('./IDL1')
    if job.output['write_ew'] > 0:
        if job.output['write_ew'] == 1:
            mask = np.arange(out.nline)
        elif job.output['write_ew'] == 2:
            mask = np.where(out.nq[:out.nline] > min(out.nq[:out.nline]))[0]

        with open(job.output['file_ew'], 'a')as f:

            # print(out.nline[mask])
            for kr in mask:
                line = out.line[kr]
                f.write('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n' \
                    %(atmos.temp, atmos.logg, atmos.feh, out.abnd, out.g[kr], out.ev[kr],\
                        line.lam0, out.f[kr], out.weq[kr], out.weqlte[kr], np.mean(atmos.vmic)) )
    print("Dooone")
    os.chdir(job.common_wd)
    return

def read_m1d_output():
    return



def run_serial_job(setup, job):
        setup_multi_job( setup, job )
        print("job # %5.0f: %5.0f M1D runs" %( job.id, len(job.atmos) ) )
        for i in range(len(job.atmos)):
            # model atom is only read once
            atom = setup.atom
            atom.abund  =  job.abund[i]
            atmos = model_atmosphere(file = job.atmos[i], format = setup.atmos_format)
            run_multi( job, atom, atmos)
            # read output
        # shutil.rmtree(job['tmp_wd'])



    # """ Start individual jobs """
    # workers = []
    # for k in set.jobs.keys():
    #     p = multiprocessing.Process( target=run_multi( k, set ) )
    #     workers.append(p)
    #     p.start()
