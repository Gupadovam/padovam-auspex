from framework import file_m2k
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from bisect import bisect
from scipy.sparse.linalg import svds

# Função SVD
def svd_denoising(matriz_empilhada, rank):
    # Realizar decomposição de valores singulares (SVD)
    U, S, V = svds(matriz_empilhada, rank)
    S = np.diag(S)

    # Estimar o ruído
    estimated_noise = U @ S @ V

    # Remover o ruído
    denoised = matriz_empilhada - estimated_noise

    return denoised

# Carregar a matriz empilhada
matriz_empilhada = np.load('/home/gustavopadovam/ENSAIOS/Matriz_empilhada.npy')

# Definir o rank para SVD
rank = 1

#Alocar todos os Ângulos
matriz_all_angles = np.zeros([1000, 196, 11])
for i in range(196):
    # Aplicar denoising usando SVD
    matriz_denoised = svd_denoising(matriz_empilhada[:, i, :], rank)
    matriz_all_angles[:,i,:] = matriz_denoised[:,:]

for i in range(11):
    np.save(f'/home/gustavopadovam/ENSAIOS/Matriz_empilhada_denoised_shot{i}.npy', matriz_all_angles)


