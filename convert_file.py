from agilent import agilentImage
from data import AgilentImageReader
import numpy as np
import os


from tkinter import filedialog as fd
from tkinter import Tk


root = Tk()
root.withdraw()
folder = fd.askdirectory(title="Select folder with all *.dat files")
files = os.listdir(folder)

print(folder)
for name in files:
    if name.split('.')[-1] == 'dat':
        try:
            path_dat = str(folder) + '/' + str(name)

            print(path_dat.split('/')[-1], '-- CONVERTING . . . ')

            dat_file = AgilentImageReader(path_dat)
            result = dat_file.read_spectra()

            xs, vals, additional = result

            meta_dados = np.array(additional.metas)

            total = np.concatenate((meta_dados, vals), axis=1)

            information = 'X,' + 'Y,' + \
                str(xs).replace('[', '').replace(']', '').replace(',', ',')
                
            path_save = str(path_dat).replace('.dat', '')
            np.savetxt(f'{path_save}_converted.csv',
                    total,
                    fmt='%.8f',
                    delimiter=',',
                    newline='\n',
                    header=f'{information}',
                    footer='',
                    comments='',
                    encoding='UTF-8')
            
            print(path_dat.split('/')[-1],' DONE')
        except:
            print(path_dat.split('/')[-1],' Fail')

print('Files was converted')