import sys
import os
import time
import numpy as np


"""
A script to convert exisiting NLTE grids to a new format
"""

def read_NLTEbin_oldFormat(binFile):
    f = open(binFile, 'rb')
    for i in range(10**9):
        atmosID = f.readline().decode('utf-8', 'ignore').split(' ')[0].strip()
        if len(atmosID) > 0:
            print(atmosID)

            ndep = np.fromfile(f, dtype='i4', count=2)[0]
            nk = np.fromfile(f, dtype='i4', count=2)[0]
            print(ndep, nk)

            tau = np.fromfile(f, dtype='f8', count=ndep)
            print(tau)

            b = np.fromfile(f, dtype='f8', count=ndep*nk).reshape(nk, ndep)
            print(b)
        else:
            f.close()
            break

    return

def write_NLTEbin_newFormat():
    return


if __name__ == '__main__':
