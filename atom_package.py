import sys
import os
import numpy as np

def write_atom_noReFormatting(atom_data, file):
    with open(file, 'w') as f:
        for li in atom_data["header"]:
            f.write(li + '\n')
        f.write("%s \n" %(atom_data["element"]))
        f.write("%10.2f %10.3f \n" %(atom_data["abund"], atom_data["atomic_weight"]))
        f.write(f"{atom_data['nk']:.0f} {atom_data['nline']:.0f} {atom_data['ncont']:.0f} {atom_data['nrfix']:.0f} \n")
        for line in atom_data["body"]:
            f.write(line + '\n')

def get_atom_inf(file, comment='') -> dict:
    atom_data = read_atom(file)
    atom_data["abund"] = None
    atom_data["body"]: list = None

    """
    A small comment line from the config file.
    Written to the header of the NLTE binary grid
    """
    if len(comment) > 100:
        print("Please, use a shorter comment for atom_comment. Stopped")
        exit(1)
    else:
        atom_data["info"] = comment
    return atom_data

def read_atom(file):
    atom_data = {}
    data = []
    for line in open(file, 'r'):
        line = line.strip()
        data.append(line)

    """ Read the header """
    c_noncom = 0  # a counter for non-commented lines
    c_all = 0  # a count for all lines, including commented lines
    atom_data["header"] = []
    for li in data:
        c_all += 1
        if not li.startswith('*') and li != '':
            c_noncom += 1
            if c_noncom == 1:
                atom_data["element"] = li
            elif c_noncom == 2:
                atom_data["abund"], atom_data["atomic_weight"] = np.array(li.split()).astype(float)
            elif c_noncom == 3:
                atom_data["nk"], atom_data["nline"], atom_data["ncont"], atom_data["nrfix"] = np.array(li.split()).astype(int)
                break
        else:
            atom_data["header"].append(li)
    atom_data["body"]: list = data[c_all:]
    return atom_data

class ModelAtom_:
    def __init__(self, file, comment=''):
        self.abund = None
        self.body: list = None
        self.read_atom(file)
        """
        A small comment line from the config file.
        Written to the header of the NLTE binary grid
        """
        if len(comment) > 100:
            print("Please, use a shorter comment for atom_comment. Stopped")
            exit(1)
        else:
            self.info = comment

    def read_atom(self, file):
        data = []
        for line in open(file, 'r'):
            line = line.strip()
            data.append(line)

        """ Read the header """
        c_noncom = 0  # a counter for non-commented lines
        c_all = 0  # a count for all lines, including commented lines
        self.header = []
        for li in data:
            c_all += 1
            if not li.startswith('*') and li != '':
                c_noncom += 1
                if c_noncom == 1:
                    self.element = li
                elif c_noncom == 2:
                    self.abund, self.atomic_weight = np.array(li.split()).astype(float)
                elif c_noncom == 3:
                    self.nk, self.nline, self.ncont, self.nrfix = np.array(li.split()).astype(int)
                    break
            else:
                self.header.append(li)
        self.body: list = data[c_all:]
