import sys
import os
from init_run import Setup
from parallel_worker import run_serial_job, collect_output
import shutil
from itertools import islice
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import numpy as np
import socket


def get_dask_client(client_type: str, cluster_name: str, workers_amount_cpus: int, nodes=1, slurm_script_commands=None,
                    slurm_memory_per_core=3.6, time_limit_hours=72, slurm_partition="debug", **kwargs):
    if cluster_name is None:
        cluster_name = "unknown"
    print("Preparing workers")
    if client_type == "local":
        client = get_local_client(workers_amount_cpus)
    elif client_type == "slurm":
        client = get_slurm_cluster(workers_amount_cpus, nodes, slurm_memory_per_core,
                                   script_commands=slurm_script_commands, time_limit_hours=time_limit_hours,
                                   slurm_partition=slurm_partition, **kwargs)
    else:
        raise ValueError("client_type must be either local or slurm")

    print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    print(f"Assuming that the cluster is ran at {cluster_name} (change in config if not the case)")

    print(f"ssh -N -L {port}:{host}:{port} {cluster_name}")
    print(f"Then go to http://localhost:{port}/status to check the status of the workers")

    print("Worker preparation complete")

    return client


def get_local_client(workers_amount, **kwargs):
    if workers_amount >= 1:
        client = Client(threads_per_worker=1, n_workers=workers_amount, **kwargs)
    else:
        client = Client(threads_per_worker=1, **kwargs)
    return client


def get_slurm_cluster(cores_per_job: int, jobs_nodes: int, memory_per_core_gb: int, script_commands=None,
                      time_limit_hours=72, slurm_partition='debug', **kwargs):
    if script_commands is None:
        script_commands = [            # Additional commands to run before starting dask worker
            'module purge',
            'module load basic-path',
            'module load intel',
            'module load anaconda3-py3.10']
    # Create a SLURM cluster object
    # split into days, hours in format: days-hh:mm:ss
    days = time_limit_hours // 24
    hours = time_limit_hours % 24
    if days == 0:
        time_limit_string = f"{int(hours):02d}:00:00"
    else:
        time_limit_string = f"{int(days)}-{int(hours):02d}:00:00"
    print(time_limit_string)
    cluster = SLURMCluster(
        queue=slurm_partition,                      # Which queue/partition to submit jobs to
        cores=cores_per_job,                     # Number of cores per job (so like cores/workers per node)
        memory=f"{memory_per_core_gb * cores_per_job}GB",         # Amount of memory per job (also per node)
        job_script_prologue=script_commands,     # Additional commands to run before starting dask worker
        walltime=time_limit_string                      # Time limit for each job
    )
    cluster.scale(jobs=jobs_nodes)      # How many nodes
    client = Client(cluster)

    return client


def chunks(data, SIZE=1000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

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
    loc1 = np.where(array1 == f"'{elem1.replace(str_to_add_array1, '').replace('.mod', '').replace('/', '')}'")[0]
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
    return run_serial_job(setup, job)


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

    """dask_temp_dir = os.path.join(os.getcwd(), 'tmp_dask_worker_space', '')
    mkdir(dask_temp_dir)
    with dask.config.set({'temporary_directory': dask_temp_dir}):
        pass
    """

    slurm = False
    if slurm:
        client = get_dask_client(client_type='slurm', cluster_name='gemini-login.mpia.de',
                                 workers_amount_cpus=setup.ncpu,
                                 nodes=5, slurm_memory_per_core=3.6, time_limit_hours=(24 * 14 - 1),
                                 slurm_partition='long')
    else:
        client = get_dask_client(client_type='local', cluster_name='gemini-login.mpia.de',
                                 workers_amount_cpus=setup.ncpu,
                                 nodes=1, slurm_memory_per_core=3.6, time_limit_hours=(24 * 14 - 1),
                                 slurm_partition='long')

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

    #jobs_split = np.split(jobs, math.ceil(len(jobs) / 1000))

    MAX_TASKS_PER_CPU_AT_A_TIME = 16000

    all_futures_combined = []

    for one_jobs_split in chunks(jobs, setup.ncpu * MAX_TASKS_PER_CPU_AT_A_TIME):
        futures = []
        for one_job in one_jobs_split:
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
