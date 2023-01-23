import sys
import os
from init_run import Setup, SerialJob
from parallel_worker import run_serial_job, collect_output
import time
import numpy as np
from dask.distributed import Client, get_worker
import socket
import shutil

def mkdir(directory: str):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

def setup_temp_dirs(setup, temporary_directory):
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
    # lock = multiprocessing.Lock()
    # lock.acquire()

    """ Make a temporary directory """
    mkdir(temporary_directory)

    """ Link input files to a temporary directory """
    for file in ['absmet', 'abslin', 'abund', 'absdat']:
        os.symlink(setup.m1d_input + '/' + file, temporary_directory + file.upper())

    """ Link INPUT file (M1D input file complimenting the model atom) """
    os.symlink(setup.m1d_input_file, temporary_directory + '/INPUT')

    """ Link executable """
    os.symlink(setup.m1d_exe, temporary_directory + 'multi1d.exe')

    """
    What kind of output from M1D should be saved?
    Read from the config file, passed here through the object setup
    """
    #job.output.update({'write_ew': setup.write_ew, 'write_ts': setup.write_ts})
    """ Save EWs """
    if setup.write_ew == 1 or setup.write_ew == 2:
        # create file to dump output
        #job.output.update({'file_ew': temporary_directory + '/output_EW.dat'})
        with open(temporary_directory + '/output_EW.dat', 'w') as f:
            f.write(
                "# Teff [K], log(g) [cgs], [Fe/H], A(X), stat. weight g_i, energy en_i, wavelength air [AA], osc. strength, EW(NLTE) [AA], EW(LTE) [AA], Vturb [km/s]    \n")

    elif setup.write_ew == 0:
        pass
    else:
        print("write_ew flag unrecognised, stoppped")
        exit(1)

    """ Output for TS? """
    if setup.write_ts == 1:
        # create a file to dump output from this serial job
        # array rec_len stores a length of the record in bytes
        #job.output.update({'file_4ts': temporary_directory + '/output_4TS.bin', \
        #                   'file_4ts_aux': temporary_directory + '/auxdata_4TS.txt', \
        #                   'rec_len': np.zeros(len(job.atmos), dtype=int)})
        # create the files
        with open(temporary_directory + '/output_4TS.bin', 'wb') as f:
            pass
        with open(temporary_directory + '/auxdata_4TS.txt', 'w') as f:
            f.write("# atmos ID, Teff [K], log(g) [cgs], [Fe/H], [alpha/Fe], mass, Vturb [km/s], A(X), pointer \n")
    elif setup.write_ts == 0:
        pass
    else:
        print("write_ts flag unrecognised, stopped")
        exit(1)
    #job.output.update({'save_idl1': setup.save_idl1})
    #if setup.save_idl1 == 1:
    #    job.output.update({'idl1_folder': setup.idl1_folder})

    # lock.release()
    #return job

def assign_temporary_directory(setup, temporary_directory):
    worker = get_worker()
    worker.tmp_dir = temporary_directory
    setup_temp_dirs(setup, temporary_directory)

    return 0

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = './config.txt'

    """ Read config. file and distribute individual jobs """
    setup = Setup(file=config_file)
    """ Start individual (serial) jobs in parallel """

    login_node_address = "gemini-login.mpia.de"

    #args = []
    #for one_job_index in setup.jobs:
    #    args.append([setup, setup.jobs[one_job_index]])
    #with Pool(processes=set.ncpu) as pool:
    #    jobs_with_result = pool.map( run_serial_job, args )
    """
    Read and organise output from each individual serial job
    into common output files
    """

    #collect_output(set, jobs_with_result)

    print("Preparing workers")
    client = Client(threads_per_worker=1,
                    n_workers=setup.ncpu)  # if # of threads are not equal to 1, then may break the program
    print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    print(f"Assuming that the cluster is ran at {login_node_address} (change in code if not the case)")

    # print(logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}"))
    print(f"ssh -N -L {port}:{host}:{port} {login_node_address}")

    all_temporary_directories = []
    for i in range(setup.ncpu):
        all_temporary_directories.append(setup.common_wd + '/job_%03d/' % i)

    futures_test = []
    for temp_dir in all_temporary_directories:
        future_test = client.submit(assign_temporary_directory, setup, temp_dir)
        futures_test.append(future_test)
    futures_test = client.gather(futures_test)

    print("Worker preparation complete")

    futures = []
    for one_job in setup.jobs:
        #big_future = client.scatter(args[i])  # good
        future = client.submit(run_serial_job, setup, setup.jobs[one_job])
        futures.append(future)  # prepares to get values

    print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
    futures = client.gather(futures)  # starts the calculations (takes a long time here)
    print("Worker calculation done")  # when done, save values

    collect_output(setup, futures)

    exit(0)
