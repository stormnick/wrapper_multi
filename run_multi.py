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

def load_aux_data(file):
    atmos, abunds = np.loadtxt(file, comments="#", usecols=(0, 7), unpack=True, dtype=str)
    abunds = abunds.astype(float)
    return atmos, abunds

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def check_same_element_loc_in_two_arrays(array1, array2_float, elem1: str, elem2_float, str_to_add_array1):
    """
    Checks whether elem1 array1 is located in the same location as elem2 in array2. If not or if not located there at
    all, returns False.
    """
    tolerance_closest_abund = 0.001

    array1 = np.asarray(array1)
    array2 = np.asarray(array2_float)
    loc1 = np.where(array1 == elem1.replace(str_to_add_array1, ""))[0]
    loc2_closest_index = find_nearest_index(array2, elem2_float)

    if np.size(loc1) == 0 or np.abs(array2[loc2_closest_index] - elem2_float) >= tolerance_closest_abund:
        return False

    if loc1[0] == loc2_closest_index:
        return True
    else:
        return False

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
        os.symlink(os.path.join(setup.m1d_input, file), os.path.join(temporary_directory, file.upper()))

    """ Link INPUT file (M1D input file complimenting the model atom) """
    os.symlink(setup.m1d_input_file, os.path.join(temporary_directory, 'INPUT'))

    """ Link executable """
    os.symlink(setup.m1d_exe, os.path.join(temporary_directory, 'multi1d.exe'))

    """
    What kind of output from M1D should be saved?
    Read from the config file, passed here through the object setup
    """
    #job.output.update({'write_ew': setup.write_ew, 'write_ts': setup.write_ts})
    """ Save EWs """
    if setup.write_ew == 1 or setup.write_ew == 2:
        # create file to dump output
        #job.output.update({'file_ew': temporary_directory + '/output_EW.dat'})
        with open(os.path.join(temporary_directory, 'output_EW.dat'), 'w') as f:
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
        with open(os.path.join(temporary_directory, 'output_4TS.bin'), 'wb') as f:
            pass
        with open(os.path.join(temporary_directory, 'auxdata_4TS.txt'), 'w') as f:
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

def assign_temporary_directory(args):
    setup, temporary_directory = args[0], args[1]
    worker = get_worker()
    worker.temporary_directory = temporary_directory
    setup_temp_dirs(setup, temporary_directory)

    return 0

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if len(sys.argv) > 2:
            aux_done_file = sys.argv[2]
            check_done_aux_files = True
        else:
            check_done_aux_files = False
            skip_fit = False
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

    print("Worker preparation complete")

    print("Creating temporary directories")

    all_temporary_directories = []
    for i in range(setup.ncpu):
        all_temporary_directories.append(setup.common_wd + '/job_%03d/' % i)

    futures_test = []
    for temp_dir in all_temporary_directories:
        big_future = client.scatter([setup, temp_dir])
        future_test = client.submit(assign_temporary_directory, big_future)
        futures_test.append(future_test)
    futures_test = client.gather(futures_test)

    #for temp_dir in all_temporary_directories:
    #    setup_temp_dirs(setup, temp_dir)

    if check_done_aux_files:
        done_atmos, done_abunds = load_aux_data(aux_done_file)

    print("Starting jobs")

    futures = []
    for one_job in setup.jobs:
        #big_future = client.scatter(args[i])  # good
        if check_done_aux_files:
            abund, atmo = setup.jobs[one_job].abund, setup.jobs[one_job].atmos
            skip_fit = check_same_element_loc_in_two_arrays(done_atmos, done_abunds, atmo, abund, setup.atmos_path)

        if not skip_fit:
            big_future = client.scatter([setup, setup.jobs[one_job]])
            future = client.submit(run_serial_job, big_future)
            futures.append(future)  # prepares to get values

    print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
    futures = client.gather(futures)  # starts the calculations (takes a long time here)
    print("Worker calculation done")  # when done, save values

    collect_output(setup, futures)
