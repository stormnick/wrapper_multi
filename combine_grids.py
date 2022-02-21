from sys import argv, exit
import os
from init_run import setup, serial_job
from parallel_worker import run_serial_job, collect_output
from multiprocessing import Pool
import time
import numpy as np
import glob
import datetime


def combineOutput_multipleJobs(path):
    auxFiles = glob.glob(path + '/auxData_*.dat')
    binFiles = glob.glob(path + '/output_*.bin')
    if len(auxFiles) != len(binFiles):
        print(f"# of auxilarly file does not equal # of grids found in {path}")
        exit()
    else:
        print(f"Found {'  '.join( str(x)  for x in auxFiles)}")
        print(f"and {'  '.join( str(x) for x in binFiles)}")
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
            pointer_last = pointer - 1
    commonBinary.close()
    commonAux.close()
    print(f"Saved to {commonBinary} and {commonAux.close}")

def combineParallelGrids_timeout(path, description):
    """ In case 99% of the computations are done but output was not organised """
    today = datetime.date.today().strftime("%b-%d-%Y")
    com_f = open(path + f"/output_NLTEgrid4TS_{today}.bin", 'wb')
    com_aux = open(path + f"/auxData_NLTEgrid4TS_{today}.dat", 'w')

    header = "NLTE grid (grid of departure coefficients) in TurboSpectrum format. \nAccompanied by an auxilarly file and model atom. \n" + \
            f"{description} \n" + \
            "Created: %s \nby Ekaterina Magg (emagg at mpia dot de) \n" %(today)
    header = str.encode('%1000s' %(header) )
    com_f.write(header)
    # Fortran starts with 1 while Python starts with 0
    pointer = len(header) + 1

    auxFiles = glob.glob(path + '/auxdata_4TS.txt')
    binFiles = glob.glob(path + '/output_4TS.bin')

    for i in range(len(auxFiles)):
        print(f"Reading from {auxFiles[i]} and {binFiles[i]}")
        # departure coefficients in binary format
        with open(binFiles[i], 'rb') as f:
            com_f.write(f.read())
        for line in open(auxFiles[i], 'r').readlines():
            if not line.startswith('#'):
                rec_len = int(line.split()[-1])
                com_aux.write('\t'.join(line.split()[0:-1]))
                com_aux.write("%10.0f \n" %(pointer))
                pointer = pointer + rec_len
            # simply copy comment lines
            else:
                com_aux.write(line)

    com_f.close()
    com_aux.close()
    datetime1 = datetime.datetime.now()

if __name__ ==  '__main__':
    """
    Search for all the grids and auxilarly files and combine into one
    """
    path = str(argv[1])
    combineOutput_multipleJobs(path)
