from framework import file_m2k
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from bisect import bisect
from scipy.sparse.linalg import svds
#
# matriz_empilhada = np.zeros([1250, 196, 14])
# for i in range(14):
#     experiment_root = "/home/gustavopadovam/ENSAIOS/"
#     experiment_ref = f"api_full_{i+9}.m2k"
#
#     data_ref = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
#                              bw_transd=0.5, tp_transd='gaussian', sel_shots=0)[0]
#     matriz_aqui = np.sum(data_ref.ascan_data, axis=2)[374:1624,83:279,0]
#
#     matriz_empilhada[:,:,i] = (matriz_aqui)
#
# np.save('/home/gustavopadovam/ENSAIOS/Matriz_empilhada.npy', matriz_empilhada)

# # Função SVD
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

# Plotar a matriz empilhada, a matriz denoise e a matriz final na mesma figura
for i in range(11):
    plt.figure(figsize=(13, 5))
    # Plotar a matriz empilhada
    plt.subplot(1, 2, 1)
    plt.imshow(np.log10(np.abs(matriz_empilhada[:,:,i])+1e-6), aspect='auto', cmap='magma', interpolation="None")
    plt.colorbar()
    plt.title(f"Matriz Original - Shot {i}")

    # Plotar a matriz
    plt.subplot(1, 2, 2)
    plt.imshow(np.log10(np.abs(matriz_all_angles[:,:,i])+1e-6), aspect='auto', cmap='magma', interpolation="None")
    plt.colorbar()
    plt.title(f"Matriz com SVD - Shot {i}")

    plt.tight_layout()
    plt.show()
