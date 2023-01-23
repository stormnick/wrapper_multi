import sys
from init_run import setup
from parallel_worker import run_serial_job, collect_output
from multiprocessing import Pool

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = './config.txt'

    """ Read config. file and distribute individual jobs """
    set = setup(file=config_file)

    """ Start individual (serial) jobs in parallel """
    args = [ [set, set.jobs[k]] for  k in set.jobs.keys()]
    with Pool(processes=set.ncpu) as pool:
        jobs_with_result = pool.map( run_serial_job, args )
    """
    Read and organise output from each individual serial job
    into common output files
    NOTE: if the job timed out before the output was produced, 
    some results might be salvaged using routines 
    inside combine_grids module
    """
    collect_output(set, jobs_with_result)

    exit(0)
