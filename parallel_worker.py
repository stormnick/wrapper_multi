import multiprocessing
import sys
import subprocess as sp
import os
import shutil
import numpy as np
from atom_package import model_atom, write_atom
from atmos_package import model_atmosphere, write_atmos_m1d, write_dscale_m1d
from m1d_output import m1d, m1dline


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

    """ Save EWs """
    if job.output['write_ew'] == 1 or job.output['write_ew'] == 2:
        # create file to dump output
        job.output.update({'file_ew' : job.tmp_wd + '/output_EW.dat' } )
        with open(job.output['file_ew'], 'w') as f:
            ## TODO: write a proper comment string
            f.write("# Lambda, temp, logg.... \n")
    elif job.output['write_ew'] == 0:
        pass
    else:
        print("write_ew flag unrecognised, stoppped")
        exit(1)

    """ Output for TS? """
    if job.output['write_ts'] == 1:
        # # TODO: write a proper header
        header = "departure coefficients from serial job # %.0f" %(job.id)
        header = str.encode('%1000s' %(header) )
        # create a file to dump output from this serial job
        # a pointer starts with 1, because Fortran starts with 1 while Python starts with 0
        job.output.update({'file_4ts' : job.tmp_wd + '/output_4TS.bin', \
                'file_4ts_aux' : job.tmp_wd + '/auxFile_4TS.txt',\
                'pointer': 1 + len(header)} )
        with open(job.output['file_4ts'], 'wb') as f:
            f.write(header)
        with open(job.output['file_4ts_aux'], 'w') as f:
            ## TODO: write a proper header
            f.write("# ")
    elif job.output['write_ts'] == 0:
        pass
    else:
        print("write_ts flag unrecognised, stoppped")
        exit(1)



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
    if job.output['write_ew'] > 0:
        out = m1d('./IDL1')
        if job.output['write_ew'] == 1:
            mask = np.arange(out.nline)
        elif job.output['write_ew'] == 2:
            mask = np.where(out.nq[:out.nline] > min(out.nq[:out.nline]))[0]

        with open(job.output['file_ew'], 'a')as f:

            # print(out.nline[mask])
            for kr in mask:
                line = out.line[kr]
                print(line)
                f.write('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n' \
                    %(atmos.teff, atmos.logg, atmos.feh, out.abnd, out.g[kr], out.ev[kr],\
                        line.lam0, out.f[kr], out.weq[kr], out.weqlte[kr], np.mean(atmos.vturb)) )
    """ Read MULTI1D output and save in a common binary file in the format for TS """
    if job.output['write_ts'] == 1:
        out = m1d('./IDL1')
        fbin = open(job.output['file_4ts'], 'ab')
        faux = open(job.output['file_4ts_aux'], 'a')

        faux.write("%10.0 \n" %(job.output['pointer']))

        atmosID = str.encode('%500s' %atmos.id)
        job.output['pointer'] = job.output['pointer'] + 500
        fbin.write(atmosID)

        ndep = int(out.ndep).to_bytes(4, 'little')
        job.output['pointer'] = job.output['pointer'] + 4
        fbin.write(ndep)

        nk = int(out.nk).to_bytes(4, 'little')
        job.output['pointer'] = job.output['pointer'] + 4
        fbin.write(nk)

        tau500 = np.array(out.tau, dtype='f8')
        fbin.write(tau500.tobytes())
        job.output['pointer'] = job.output['pointer'] + out.ndep * 8
        # #
        depart = np.array((out.n/out.nstar).reshape(out.ndep, out.nk), dtype='f8')
        fbin.write(depart.tobytes())
        job.output['pointer'] = job.output['pointer'] + out.ndep * out.nk * 8

        fbin.close()
        faux.clos()

    os.chdir(job.common_wd)
    return

def collect_output(setup):
    from datetime import date
    today = date.today().strftime("%b-%d-%Y")

    """ Collect all EW grids into one """
    if setup.write_ew > 0:
        print("Collecting EW grids...")
        with open(setup.common_wd + '/output_EWgrid_%s.dat' %(today), 'w') as com_f:
            for k in setup.jobs.keys():
                job = setup.jobs[k]
                data = open(job.output['file_ew'], 'r').readlines()
                com_f.writelines(data)
    """ Collect all TS formatted NLTE grids into one """
    if setup.write_ts > 0:
        print("Collecting TS formatted grids...")
        com_f = open(setup.common_wd + '/output_NLTEgrid4TS_%s.dat' %(today), 'wb')
        com_aux = open(setup.common_wd + '/auxData_NLTEgrid4TS_%s.dat' %(today), 'w')

        for k in setup.jobs.keys():
            job = setup.jobs[k]
            with open(job.output['file_4ts'], 'rb') as f:
                com_f.write(f.read())
            with open(job.output['file_4ts_aux'], 'r') as f:
                com_aux.write(f.read())
                
        com_f.close()
        com_aux.close()
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
        # shutil.rmtree(job['tmp_wd'])



    # """ Start individual jobs """
    # workers = []
    # for k in set.jobs.keys():
    #     p = multiprocessing.Process( target=run_multi( k, set ) )
    #     workers.append(p)
    #     p.start()
