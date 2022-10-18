import numpy as np
from sys import exit, argv
from matplotlib import pyplot as plt
from scipy import interpolate
import cProfile

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def read_ew_grid(filename='./ewgrid.txt'):
    """
    Read a grid of EW as produced by running wrapper-multi
    organise a dictionary and return
    Input:
    filename (string): path to the EW grid
    Output:
    ew_dict (dictionary)

    """
    data = np.loadtxt(filename, comments='#')

    ew_dict = {}
    i = 0
    for k in ['teff', 'logg', 'feh', 'abund',  'g_stat', 'e_i',  'lam', 'fosc', 'ew_nlte', 'ew_lte', 'vturb']:
        ew_dict.update({ k : data[:, i] })
        i += 1

    with open(filename, 'r') as f:
        atmos_ids = np.array([ s.split()[-1].replace('\n', '') for s in f.readlines() if not s.startswith('#') ])
    ew_dict.update({ 'atmos':atmos_ids })

    return ew_dict


def compute_corrections(ew_dict):
    """
    Compute NLTE corrections from the CoG
    Input:
    ew_dict (dictionary)
    """
    ew_dict.update({ 'nlte_corr' : np.full(len(ew_dict['ew_lte']), np.nan) })
    ar_corr = np.zeros(len(ew_dict['lam']))

    size_total = len(ar_corr)

    printProgressBar(0, size_total, prefix = 'NLTE corrections grid:', suffix = 'Complete', length = 50)
    i_pr = 0

    atm_un = np.unique(ew_dict['atmos'])
    lam_un = np.unique(ew_dict['lam'])
    for atm in atm_un:
        # this is here temporaly, to avoid computing the whole MN grid
        pos = np.where(ew_dict['atmos']==atm)[0]
        temp = ew_dict['teff'][pos][0]
        logg = ew_dict['logg'][pos][0]
        feh = ew_dict['feh'][pos][0]
        for line in lam_un:
            mask = np.logical_and(ew_dict['atmos'] == atm, ew_dict['lam'] == line)
            # interpolated abundance as a function of EW in LTE
            f_interp = interpolate.interp1d(ew_dict['ew_lte'][mask], ew_dict['abund'][mask], kind='linear', fill_value = 'extrapolate')
            max_ab = max(ew_dict['abund'][mask])
            min_ab = min(ew_dict['abund'][mask])
            for i in range(len(ew_dict['ew_nlte'][mask])):
                # EW of the line if it was computed in NLTE
                ew_n = ew_dict['ew_nlte'][mask][i]
                # if line is weaker than 10 mÅ, don't compute correction
                if ew_n < 0.001:
                    corr = np.nan
                else:
                    # to match this EW while modelling in LTE, one needs to use abundance of:
                    ab_n = f_interp(ew_n)
                    # which is *corr* different from the center value
                    corr = ew_dict['abund'][mask][i] - ab_n
                    # then NLTE abundance (which we know here, it's ew_dict['abund'][mask][i]) is ab_n (as measured in LTE) + corr

                ew_dict['nlte_corr'][np.where(mask)[0][i]] = corr
                i_pr += 1
                printProgressBar(i_pr, size_total, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return ew_dict

def write_corrections_grid(out_dict, path='./NLTEgrid'):
    # sort in the following order:
    for k in ['atmos', 'lam', 'abund']:
        sorted_ind = out_dict[k].argsort(kind='mergesort')
        for kk in out_dict.keys():
            out_dict[kk] = out_dict[kk][sorted_ind]

    # write in the text file
    with open(path + '.txt','w') as f:
        header = '# atmosID, teff, logg, feh, abund,  vturb, lam,  g_stat, e_i,  fosc, ew_nlte, ew_lte, nlte_corr \n'
        f.write(header)
        # for formatting of strings
        max_len = max([len(s) for s in out_dict['atmos']])

        for i in range(len(out_dict['atmos'])):
            if out_dict['nlte_corr'][i] != None and np.isfinite(out_dict['nlte_corr'][i]):
                s = f"{out_dict['atmos'][i]:<{max_len}s}  "
                for k in ['teff', 'logg', 'feh', 'abund',  'vturb',  'lam', 'g_stat', 'e_i', 'fosc', 'ew_nlte', 'ew_lte', 'nlte_corr']:
                    s = s + f"{out_dict[k][i] : 3f}  "
                f.write( s + '\n' )

def read_corrections_grid(filename = './NLTEgrid.txt'):
    atmos = np.loadtxt(filename, comments='#', usecols=0, dtype=str)
    # fill the first column (strings) with nans
    data = np.loadtxt(filename, comments='#', converters = {0: lambda s: np.nan})

    corr = {}
    i = 1
    corr.update({'atmos' : atmos })
    for k in ['teff', 'logg', 'feh', 'abund',  'vturb',  'lam', 'g_stat', 'e_i', 'fosc', 'ew_nlte', 'ew_lte', 'nlte_corr']:
        corr.update({ k : data[:, i]})
        i += 1

    return corr

if __name__ == '__main__':
    if len(argv) < 2:
        raise Warning("<Usage>: python3.7 ./compute_nlte_corrections.py './path/EW_grid.txt' ")
    filename = argv[1]

    ew_dict = read_ew_grid(filename)
    out = compute_corrections(ew_dict)
    write_corrections_grid(out)
