import sys
import os
from init_run import setup
from parallel_worker import *
import multiprocessing



# if __name__ == '__main__':
if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = './config.txt'


""" Read config. file and distribute individual jobs """
set = setup(file=config_file)
""" Start individual (serial) jobs in parallel """
# here comes the multiprocessing part


    # """ Start individual jobs """
workers = []


for k in set.jobs.keys():
    job = set.jobs[k]
    setup_multi_job(set, job)
    p = multiprocessing.Process( target=run_serial_job(set, job) )
    workers.append(p)
for p in workers:
    p.start()

collect_output(set)

exit(0)
