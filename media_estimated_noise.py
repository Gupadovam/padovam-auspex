from framework import file_m2k
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from bisect import bisect
from scipy.sparse.linalg import svds

def svd_estimated_noising(matriz_empilhada, rank):
    # Realizar decomposição de valores singulares (SVD)
    U, S, V = svds(matriz_empilhada, rank)
    S = np.diag(S)

    # Estimar o ruído
    estimated_noise = U @ S @ V

    return estimated_noise

# Carregar a matriz empilhada
matriz_empilhada = np.load('/home/gustavopadovam/ENSAIOS/Matriz_empilhada.npy')

# Definir o rank para SVD
rank = 1

#Alocar todos os Ângulos
matriz_all_angles = np.zeros([1000, 196, 11])
for i in range(196):
    # Aplicar denoising usando SVD
    estimated_noise = svd_estimated_noising(matriz_empilhada[:, i, :], rank)
    matriz_all_angles[:,i,:] = estimated_noise[:,:]
    med_estimated_noise = np.mean(matriz_all_angles,axis=2)

plt.figure(figsize=(13, 5))
# Plotar a matriz empilhada
plt.subplot(1, 2, 1)
plt.imshow(np.log10(np.abs(matriz_empilhada[:,:,8])+1e-6), aspect='auto', cmap='magma', interpolation="None")
plt.colorbar()
plt.title(f"Matriz Original - Amostra {8}")

plt.subplot(1, 2, 2)
denoised = matriz_empilhada[:,:,7]- med_estimated_noise
plt.imshow(np.log10(np.abs(denoised)+1e-6), aspect='auto', cmap='magma', interpolation="None")
plt.colorbar()
plt.title(f"Matriz com subtração da média do estimated noise- Shot {8}")

plt.show()