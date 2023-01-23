import dask
from dask.distributed import Client, get_worker
from time import sleep
import numpy as np

def long_task(number):
    worker = get_worker()
    #print(worker.my_personal_state)
    if worker.folder_to_use == 1:
        extra_sleep = 5
    else:
        extra_sleep = 0
    print(f"{worker.folder_to_use} start {number}")
    sleep(0.1 + number / 1000 + extra_sleep)
    print(f"{worker.folder_to_use} end {number}")
    return "hi"

def my_function(number):
    worker = get_worker()
    worker.folder_to_use = number



if __name__ == '__main__':
    folder_numbers = [1,2,3,4]
    tasks = np.arange(0, 100, 1)

    client = Client(threads_per_worker=1, n_workers=4)
    for number in folder_numbers:
        future = client.submit(my_function, number)
    futures = []
    for task in tasks:
        future = client.submit(long_task, task)
        futures.append(future)
    futures = client.gather(futures)
