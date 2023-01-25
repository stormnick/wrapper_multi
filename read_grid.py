import numpy as np
from collections import Counter
import sys

try:
    sys.set_int_max_str_digits(9999999)
except:
    pass


def check_length_of_pointers(path):
    data = np.loadtxt(path, usecols=(8), dtype=int)
    print(Counter(np.diff(data)))
    print(np.where(np.diff(data) == 1000))


def read_grid(aux_file, bin_file):
    pointers = np.loadtxt(aux_file, usecols=(8), unpack=True, dtype=int)
    pointers = np.delete(pointers, 0)

    header = "NLTE grid (grid of departure coefficients) in TurboSpectrum format. \nAccompanied by an auxilarly file and model atom. \n"
    header = str.encode('%1000s' % (header))

    pointer = len(header) + 1

    print(pointer)

    fbin = open(bin_file, 'rb')
    fbin.read(len(header))
    with open("temp_file.txt", 'a+') as f:

        for one_pointer in pointers:
            if one_pointer == 99999999:
                return
            one_str = str(fbin.read(500))#.replace("", "")
            pointer += 500
            """if "x" in one_str:
                print("x")
                one_str = str(fbin.read(500)).replace(" ", "")
                pointer += 500
                one_str = str(fbin.read(500)).replace(" ", "")
                pointer += 500"""
            f.write(one_str)
            int_conv = str(int.from_bytes(fbin.read(4), 'little'))
            f.write(int_conv)

            pointer += 4
            fbin.read(one_pointer - pointer)
            #print(int.from_bytes(fbin.read(191909 - pointer - 1), 'little'))
            pointer += (one_pointer - pointer)
            #print(fbin.read(500))
            f.write(f" {one_pointer}\n")


if __name__ == '__main__':
    read_grid("auxData_NLTEgrid4TS_Jan-11-2023_combined.dat", "output_NLTEgrid4TS_Jan-11-2023_combined.bin")
