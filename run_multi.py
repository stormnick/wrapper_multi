import sys
import os
from init_run import Setup, SerialJob
from parallel_worker import run_serial_job, collect_output
import time
import numpy as np
from dask.distributed import Client, get_worker
import socket
import shutil
import dask
import math

def mkdir(directory: str):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

def load_aux_data(file):
    atmos, feh, abunds = np.loadtxt(file, comments="#", usecols=(0, 3, 7), unpack=True, dtype=str)
    abunds = abunds.astype(float)
    feh = feh.astype(float)
    return atmos, feh, abunds

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def check_same_element_loc_in_two_arrays(array1, array2_float, elem1: str, elem2_float, str_to_add_array1):
    """
    Checks whether elem1 array1 is located in the same location as elem2 in array2. If not or if not located there at
    all, returns False.
    """
    tolerance_closest_abund = 0.001  # tolerance to consider abundances the same

    array1 = np.asarray(array1)
    array2 = np.asarray(array2_float)
    loc1 = np.where(array1 == f"'{elem1.replace(str_to_add_array1, '').replace('.mod', '')}'")[0]
    #loc2_closest_index = find_nearest_index(array2, elem2_float)

    #print(array1, array2, elem1, elem2_float, f"'{elem1.replace(str_to_add_array1, '').replace('.mod', '')}'")

    if np.size(loc1) == 0:
        return False

    for index_to_check in loc1:
        #np.abs(array2[loc1[0]] - elem2_float) < tolerance_closest_abund
        if np.abs(array2[index_to_check] - elem2_float) < tolerance_closest_abund:
            return True
    return False


def launch_job(job):
    run_serial_job(setup, job)


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
        check_done_aux_files = False
        skip_fit = False

    """ Read config. file and distribute individual jobs """
    setup = Setup(file=config_file)
    jobs = setup.distribute_jobs()
    setup.atmos = None
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

    dask_temp_dir = os.path.join(os.getcwd(), 'tmp_dask_worker_space', '')
    mkdir(dask_temp_dir)
    with dask.config.set({'temporary_directory': dask_temp_dir}):
        pass

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

    #print("Creating temporary directories")

    """all_temporary_directories = []
    for i in range(setup.ncpu):
        all_temporary_directories.append(setup.common_wd + '/job_%03d/' % i)

    futures_test = []
    for temp_dir in all_temporary_directories:
        big_future = client.scatter([setup, temp_dir])
        future_test = client.submit(assign_temporary_directory, big_future)
        futures_test.append(future_test)
    futures_test = client.gather(futures_test)"""

    #for temp_dir in all_temporary_directories:
    #    setup_temp_dirs(setup, temp_dir)

    if check_done_aux_files:
        done_atmos, feh, done_abunds = load_aux_data(aux_done_file)
        done_abunds = done_abunds - feh

    print("Starting jobs")

    jobs_amount: int = 0

    jobs_split = np.split(jobs, math.ceil(len(jobs) / 1000))

    all_futures_combined = []

    for one_jobs_split in jobs_split:
        futures = []
        for one_job in jobs:
            #big_future = client.scatter(args[i])  # good
            if check_done_aux_files:
                abund, atmo = jobs[one_job].abund, jobs[one_job].atmo
                skip_fit = check_same_element_loc_in_two_arrays(done_atmos, done_abunds, atmo, abund, setup.atmos_path)

            if not skip_fit:
                jobs_amount += 1
                big_future = client.scatter(jobs[one_job])
                #big_future_setup = client.scatter(setup, broadcast=True)
                #[big_future_setup] = client.scatter([setup], broadcast=True)

                #[fut_dict] = client.scatter([setup], broadcast=True)
                #score_guide = lambda row: expensive_computation(fut_dict, row)

                future = client.submit(launch_job, big_future)
                futures.append(future)  # prepares to get values

        print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
        futures = client.gather(futures)  # starts the calculations (takes a long time here)
        print("Worker calculation done")  # when done, save values
        all_futures_combined += futures

    #setup.njobs = jobs_amount

    collect_output(setup, all_futures_combined, jobs_amount)
