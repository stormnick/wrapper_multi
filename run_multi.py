#!/usr/local2/misc/anaconda/anaconda3/bin/python3.7
import sys
import os
from init_run import setup, serial_job
from parallel_worker import run_serial_job, collect_output
import multiprocessing
from multiprocessing import Pool
import time
import numpy as np





if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = './config.txt'
        
    """ Read config. file and distribute individual jobs """
    set = setup(file=config_file)
    """ Start individual (serial) jobs in parallel """

    args = []
    for k in set.jobs.keys():
        args.append([set, set.jobs[k]] )
    with Pool(processes=set.ncpu) as pool:
        jobs_with_result = pool.map( run_serial_job, args )
    """
    Read and organise output from each individual serial job
    into common output files
    """
    collect_output(set, jobs_with_result)

    exit(0)
