import sys
import os
from init_run import setup, serial_job
from parallel_worker import run_serial_job, collect_output
import multiprocessing
from multiprocessing import Pool
import time



if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = './config.txt'

if __name__ == '__main__':

    """ Read config. file and distribute individual jobs """
    set = setup(file=config_file)
    """ Start individual (serial) jobs in parallel """
    # here comes the multiprocessing part


    # """ Start individual jobs """
    workers = []
    for k in set.jobs.keys():
        job = set.jobs[k]
        p = multiprocessing.Process( target=run_serial_job, args=(set, job) )
        workers.append(p)
    # start every process with a delay of 1 second
    for p in workers:
        p.start()
        time.sleep(1)
    # wait until all processes are done before proceeding
    for p in workers:
        p.join()

    for k in set.jobs.keys():
        job = set.jobs[k]
        print(k, set.jobs[k])
    # collect_output(set)
    exit(0)
