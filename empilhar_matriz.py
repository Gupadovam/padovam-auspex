from framework import file_m2k
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from bisect import bisect
from scipy.sparse.linalg import svds

experiment_root ="/home/gustavopadovam/Varredura_circunferencial_full/sw_full.m2k"
experiment_ref =""
matriz_empilhada = np.zeros([1000, 196, 11])
for i in range(11):
    data_ref = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian', sel_shots=i)[1]
    matriz_aqui = np.sum(data_ref.ascan_data, axis=2)[624:1624,82:278,0]

    matriz_empilhada[:,:,i] = (matriz_aqui)

np.save('/home/gustavopadovam/ENSAIOS/Matriz_empilhada.npy', matriz_empilhada)

