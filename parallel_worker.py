import os
import shutil
import numpy as np
from atom_package import write_atom_noReFormatting
from atmos_package import *
from combine_grids import addRec_to_NLTEbin
from m1d_output import m1d
import datetime
from dask.distributed import get_worker


def mkdir(s):
    if os.path.isdir(s):
        shutil.rmtree(s)
    os.mkdir(s)


def setup_multi_job(setup, job, temporary_directory):
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

    """
    Make sure that only one process at a time can access input files
    """
    #lock = multiprocessing.Lock()
    #lock.acquire()

    #mkdir(temporary_directory)

    job.output.update({'write_ew': setup.write_ew, 'write_ts': setup.write_ts})
    """ Save EWs """
    if job.output['write_ew'] == 1 or job.output['write_ew'] == 2:
        # create file to dump output
        job.output.update({'file_ew': temporary_directory + '/output_EW.dat'})

    """ Output for TS? """
    if job.output['write_ts'] == 1:
        # create a file to dump output from this serial job
        # array rec_len stores a length of the record in bytes
        job.output.update({'file_4ts': temporary_directory + '/output_4TS.bin', \
                           'file_4ts_aux': temporary_directory + '/auxdata_4TS.txt', \
                           'rec_len': np.zeros(len(job.atmos), dtype=int)})

    job.output.update({'save_idl1': setup.save_idl1})
    if setup.save_idl1 == 1:
        job.output.update({'idl1_folder': setup.idl1_folder})

    #lock.release()
    return job


def run_multi(job, atom, atmos, temporary_directory):
    """
    Run MULTI1D
    input:
    (object) setup:
    (object) job:
    (integer) i: index pointing to the current model atmosphere and abundance
                 (in job.atmos, job.abund)
                 this model atmosphere and abund will be used to run M1D
    (object) atom:  object of class model_atom
    """

    """ Create ATOM input file for M1D """
    write_atom_noReFormatting(atom, temporary_directory + '/ATOM')

    """ Create ATMOS input file for M1D """
    write_atmos_m1d(atmos, temporary_directory + '/ATMOS')
    write_dscale_m1d(atmos, temporary_directory + '/DSCALE')

    """ Go to directory and run MULTI 1D """
    os.chdir(temporary_directory)

    os.system('time ./multi1d.exe')

    """ Read M1D output if M1D run was successful """
    if os.path.isfile('./IDL1'):
        out = m1d('./IDL1')

        """ print MULTI1D output to the temporary grid of EWs """
        if job.output['write_ew'] > 0:
            if job.output['write_ew'] == 1:
                mask = np.arange(out.nline)
            elif job.output['write_ew'] == 2:
                mask = np.where(out.nq[:out.nline] == max(out.nq[:out.nline]))[0]

            with open(job.output['file_ew'], 'a') as f:
                for kr in mask:
                    line = out.line[kr]
                    f.write("%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %20.10f %20.10f %20.4f # '%s'\n" \
                            % (atmos.teff, atmos.logg, atmos.feh, out.abnd, out.g[out.irad[kr]], out.ev[out.irad[kr]], \
                               line.lam0, out.f[kr], out.weq[kr], out.weqlte[kr], np.mean(atmos.vturb), atmos.id))

        """ save  MULTI1D output in a common binary file in the format for TS """
        if job.output['write_ts'] == 1:
            faux = open(job.output['file_4ts_aux'], 'a')
            # append record to binary grid file
            with np.errstate(divide='ignore'):
                depart = np.array((out.n / out.nstar), dtype='f8')
                # transpose to match Fortran order of things
                # (nk, ndep)
                depart = depart.T
            record_len = addRec_to_NLTEbin(job.output['file_4ts'], atmos.id, out.ndep, out.nk, out.tau, depart)

            faux.write(" '%s' %10.4f %10.4f %10.4f %10.4f %10.2f %10.2f %10.4f %60.0f \n" \
                       % (atmos.id, atmos.teff, atmos.logg, atmos.feh, atmos.alpha, atmos.mass, np.mean(atmos.vturb),
                          out.abnd, record_len))
            faux.close()

        if job.output['save_idl1'] == 0:
            os.remove('./IDL1')
        elif job.output['save_idl1'] == 1:
            destin = job.output['idl1_folder'] + "/idl1.%s_%s_A(X)_%5.5f" % (
            atmos.id, atom.element.replace(' ', ''), atom.abund)
            shutil.move('./IDL1', destin)
    # no IDL1 file created after the run
    else:
        print("IDL1 file not found for %s A(X)=%.2f" % (atmos.id, atom.abund))

    os.chdir(job.common_wd)


def collect_output(setup, jobs):
    today = datetime.date.today().strftime("%b-%d-%Y")

    """ Collect all EW grids into one """
    datetime0 = datetime.datetime.now()
    print("Collecting grids of EWs")
    if setup.write_ew > 0:
        with open(os.path.join(setup.common_wd, '/output_EWgrid_%s.dat' % (today)), 'w') as com_f:
            for job in jobs:
                data = open(job.output['file_ew'], 'r').readlines()
                com_f.writelines(data)
        """ Checks to raise warnings if there're repeating entrances """
        with open(os.path.join(setup.common_wd + '/output_EWgrid_%s.dat' % (today)), 'r') as f:
            data_all = f.readlines()
        params = []
        for line in data_all:
            if not line.startswith('#'):
                abund = line.split()[3]
                atmosID = line.split()[-1]
                wave = line.split()[6]
                if not [wave, abund, atmosID] in params:
                    params.append([wave, abund, atmosID])
                else:
                    print("WARNING: found repeating entrance at \n %s AA  A(X)=%s, atmos: %s " \
                          % (wave, abund, atmosID))
        datetime1 = datetime.datetime.now()
        print(datetime1 - datetime0)
        print(10 * "-")
        """ #TODO sort the grids of EWs """

    """ Collect all TS formatted NLTE grids into one """
    print("Collecting TS formatted NLTE grids")
    datetime0 = datetime.datetime.now()
    if setup.write_ts > 0:
        com_f = open(os.path.join(setup.common_wd + '/output_NLTEgrid4TS_%s.bin' % (today)), 'wb')
        com_aux = open(os.path.join(setup.common_wd + '/auxData_NLTEgrid4TS_%s.dat' % (today)), 'w')

        header = "NLTE grid (grid of departure coefficients) in TurboSpectrum format. \nAccompanied by an auxilarly file and model atom. \n" + \
                 "NLTE element: %s \n" % (setup.atom.element) + \
                 "Model atom: %s \n" % (setup.atom_id) + \
                 "Comments: '%s' \n" % (setup.atom.info) + \
                 "Number of records: %10.0f \n" % (setup.njobs) + \
                 f"Computed with MULTI 1D (using EkaterinaSe/wrapper_multi (github)), {today} \n"
        header = str.encode('%1000s' % (header))
        com_f.write(header)

        # Fortran starts with 1 while Python starts with 0
        pointer = len(header) + 1

        for job in jobs:
            # departure coefficients in binary format
            with open(job.output['file_4ts'], 'rb') as f:
                com_f.write(f.read())
            for line in open(job.output['file_4ts_aux'], 'r').readlines():
                if not line.startswith('#'):
                    rec_len = int(line.split()[-1])
                    com_aux.write('    '.join(line.split()[0:-1]))
                    com_aux.write("%20.0f \n" % (pointer))
                    pointer = pointer + rec_len
                # simply copy comment lines
                else:
                    com_aux.write(line)

        com_f.close()
        com_aux.close()
        datetime1 = datetime.datetime.now()
        print(datetime1 - datetime0)
        print(10 * "-")


def run_serial_job(setup, job):
    print(f"job # {job.id}: {len(job.atmos)} M1D runs")
    worker = get_worker()
    temporary_directory: str = worker.temporary_directory

    job = setup_multi_job(setup, job, temporary_directory)

    # model atom is only read once
    atom = setup.atom
    atmos = ModelAtmosphere()
    atmos.read(file=job.atmos, format=setup.atmos_format)
    # scale abundance with [Fe/H] of the model atmosphere
    if np.isnan(atmos.feh):
        atmos.feh = 0.0
    if not atom.element.lower() == 'h':
        atom.abund = job.abund + atmos.feh

    run_multi(job, atom, atmos, temporary_directory)

    # shutil.rmtree(job['tmp_wd'])
    return job
