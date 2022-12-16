from sys import argv, exit
import os
import time
import numpy as np
import glob
import datetime

def addRec_to_NLTEbin(binFile, atmosID, ndep, nk, tau, depart):
    # transform to match fortran format

    # writes a record into existing NLTE binary
    fbin = open(binFile, 'ab')
    record_len = 0

    record_len = record_len + 500
    fbin.write(str.encode('%500s' %atmosID))

    record_len = record_len + 4
    fbin.write(int(ndep).to_bytes(4, 'little'))

    record_len = record_len + 4
    fbin.write(int(nk).to_bytes(4, 'little'))

    fbin.write(np.array(tau, dtype='f8').tobytes())
    record_len = record_len + ndep * 8
    fbin.write(np.array(depart, dtype='f8').tobytes())
    record_len = record_len + ndep * nk * 8

    fbin.close()

    return record_len

def addDeparturesToExistingGrid(binFilePath, auxFilePath, NLTEdata):
    """
    Append departure coefficients and complimenting data
    to the existing binary NLTE grid and auxilarly file

    Input:

    """
    atmosID, ndep, nk, tau, depart = NLTEdata
    record_len = addRec_to_NLTEbin(binFilePath, atmosID, ndep, nk, tau, depart)

    auxData = open(auxFilePath).readlines()
    pointer_last = int(auxData[-1].split()[-1])

    teff, logg, feh, alpha, mass, vturb, abund = np.random.random(7)
    with open(auxFilePath, 'a') as auxF:
        auxF.write(f" '{atmosID}'  {teff:10.4f} {logg:10.4f} {feh:10.4f}\
                        {alpha:10.4f} {mass:10.4f} {vturb:10.4f} {abund:10.4f} \
                        {pointer_last + record_len:60.0f} \n")



def combineOutput_multipleJobs(path):
    auxFiles = glob.glob(path + '/auxData_*.dat')
    binFiles = glob.glob(path + '/output_*.bin')
    if len(auxFiles) != len(binFiles):
        print(f"# of auxilarly files does not equal # of grids found in {path}")
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
                    commonAux.write(f" {'    '.join(line.split()[0:-1]) }     {pointer:60.0f}\n")
            pointer_last = pointer - 1
    commonBinary.close()
    commonAux.close()
    print(f"Saved to output_NLTEgrid4TS_{today}_combined.bin and ./auxData_NLTEgrid4TS_{today}_combined.dat")

def combineParallelGrids_timeout(path, description):
    """ In case 99% of the computations are done but output was not organised """
    today = datetime.date.today().strftime("%b-%d-%Y")
    com_f = open( f"./output_NLTEgrid4TS_{today}.bin", 'wb')
    com_aux = open(f"./auxData_NLTEgrid4TS_{today}.dat", 'w')

    header = "NLTE grid (grid of departure coefficients) in TurboSpectrum format. \nAccompanied by an auxilarly file and model atom. \n" + \
            f"{description} \n" + \
            "Computed with MULTI 1D (using EkaterinaSe/wrapper_multi (github)), \n" %(today)
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
                com_aux.write(f"{'  '.join(line.split()[0:-1])}   {pointer:35.0f} \n")
                pointer = pointer + rec_len
            # simply copy comment lines
            else:
                com_aux.write(line)

    com_f.close()
    com_aux.close()
    print(f"saved in ./output_NLTEgrid4TS_{today}.bin and ./auxData_NLTEgrid4TS_{today}.dat")

if __name__ ==  '__main__':
    """
    Search for all the grids and auxilarly files and combine into one
    """
    path = str(argv[1])
    combineOutput_multipleJobs(path)
