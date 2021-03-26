import sys
import os
import time
import numpy as np
from parallel_worker import addRec_to_NLTEbin


"""
A script to convert exisiting NLTE grids to a new format
"""

if __name__ == '__main__':
    binFile = sys.argv[1]
    print("Reading binary NLTE grid from %s" %(binFile) )

    f = open(binFile, 'rb')
    header = f.readline(1000).decode('utf-8', 'ignore')
    print(header)
    for i in range(10**9):
        atmosID = f.readline(500).decode('utf-8', 'ignore')#.split(' ')[0].strip()
        print(atmosID)
        if len(atmosID) > 0:

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
