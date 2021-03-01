import sys
import os
from init_run import setup
from parallel_worker import run_serial_job, collect_output
import multiprocessing
from multiprocessing import Pool




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
        print(p, p.is_alive())
        workers.append(p)
    for p in workers:
        p.start()
        time.sleep(1)
    for p in workers:
        p.join()
    print("done")
        # p.join()

# collect_output(set)

    exit(0)
