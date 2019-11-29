from agilent import agilentImage
from data import AgilentImageReader
import numpy as np


from tkinter import filedialog as fd
from tkinter import Tk


root = Tk()
root.withdraw()
file = fd.askopenfile(title="File *.dat")
path_dat = file.name
print(path_dat)

# path_dat = '/home/ABTLUS/joao.levandoski/Documents\
# /ic-orange/transfer/ic-orange_D/jupyter/antes-do-login\
# /arquivo[.hdr e .spc]\
# /pos0_x25_9-04mm_1.dat'

arquivo = AgilentImageReader(path_dat)
resultado = arquivo.read_spectra()

xs, vals, additional = resultado
# print(arquivo.data)
# print(arquivo.info.keys())
# print(xs) #Wavenumber
# print(vals.shape) #Resultado (16384 linhas) (1841 colunas)
# print(additional) #Ponto de onde foi captado o spectra 0>127 (x) 0>127(y)
# a = str(xs)
# # print(a)

# f = open('testando.txt', 'w')

# f.write(a)
# f.close()
# y = []
# for i in range(128):
#     y.append(i)

nome = path_dat.split('.')[-2]
nome = nome.split('/')[-1]

# pixeis = 128
# map_x = np.array([], dtype=int)
# number = np.arange(0, pixeis)
# for _ in range(pixeis):
#     map_x = np.concatenate((map_x, number), axis=0)

# map_y = np.array([], dtype=int)
# arr = np.zeros(pixeis, dtype=int)
# for j in range(pixeis):
#     map_y = np.concatenate((map_y, arr), axis=0)
#     arr += 1

# map_x = map_x.reshape((len(map_x), 1))
# map_y = map_y.reshape((len(map_y), 1))

# print(map_x.shape)
# print(map_y.shape)
# coordenadas = np.concatenate((map_x, map_y), axis=1)
# # print(coordenadas)

meta_dados = np.array(additional.metas)

total = np.concatenate((meta_dados, vals), axis=1)

# print(total.shape)
information = 'map_x,' + 'map_y,' + \
    str(xs).replace('[', '').replace(']', '').replace(',', ',')
# print(additional.metas)
# meta_dados = np.array(additional.metas)
# print(meta_dados.shape)

np.savetxt(f'{path_dat}-converted.csv',
           total,
           fmt='%.8f',
           delimiter=',',
           newline='\n',
           header=f'{information}',
           footer='',
           comments='',
           encoding='UTF-8')
