from sys import argv, exit
import os
from init_run import setup, serial_job
from parallel_worker import run_serial_job, collect_output
from multiprocessing import Pool
import time
import numpy as np
import glob
import datetime

if __name__ ==  '__main__':
    """
    Search for all the grids and auxilarly files and combine into one
    """
    path = str(argv[1])
    auxFiles = glob.glob(path + '/auxData_*.dat') 
    binFiles = glob.glob(path + '/output_*.bin')
    if len(auxFiles) != len(binFiles):
        print(f"# of auxilarly file does not equal # of grids found in {path}")
        exit()

    today = datetime.date.today().strftime("%b-%d-%Y")
    commonBinary = open('./output_NLTEgrid4TS_%s_combined.bin' %(today), 'wb')
    commonAux = open('./auxData_NLTEgrid4TS_%s_combined.dat' %(today), 'w')

    pointer_last = 0
    writtenComment = False

    for i in range(len(auxFiles)):
        if 'combined' not in auxFiles[i]:
            print(f"Reading from {auxFiles[i]} and {binFiles[i]}")
    
            with open(binFiles[i], 'rb') as f:
                commonBinary.write(f.read())
    
            for line in open(auxFiles[i], 'r').readlines():
                if line == '':
                    print(f"found empty line in {auxFiles[i]}")
                    exit()
                elif line.startswith('#'):
                        if not writtenComment:
                            commonAux.write(line)
                            writtenComment = True
                else:
                    line = line.split('#')[0].replace('\n','')
                    pointer = int(line.split()[-1]) + pointer_last
                    commonAux.write(f" {'    '.join(line.split()[0:-1]) } {pointer:.0f}\n")
            pointer_last = pointer
    
        commonBinary.close()
        commonAux.close()
