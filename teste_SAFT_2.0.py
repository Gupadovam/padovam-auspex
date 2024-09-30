import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import matplotlib
from tqdm import tqdm
import time

dados = np.load("/media/gustavopadovam/USB20FD/dados_tfm.npy", allow_pickle=True).item()
ascans = dados.get('ascans')
speed_m_s = dados.get('speed_m_s') * 1e3
f_sampling_MHz = dados.get('f_sampling_MHz') * 1e6
samples_t_init_microsec = dados.get('samples_t_init_microsec') * 1e-6
elem_positions_mm = dados.get('elem_positions_mm')

g = np.diagonal(ascans, 0, 1, 2)  # B-scan

ts = 1 / f_sampling_MHz  # Periodo de amostragem

roi_ori = np.asarray([10, -20])  # Define origem da ROI
roi_width = 40  # Define largura da ROI
roi_height = 20  # Define altura da ROI
roi_res = (100, 64)  # Define a resolução da ROI

z = np.linspace(roi_ori[0], roi_ori[0] + roi_height, roi_res[0])  # Convertendo para posição
xroi = np.linspace(roi_ori[1], roi_ori[1] + roi_width, roi_res[1])  # Convertendo para posição


def saft_2(g, cl, xt, z, ts, t_init):
    height = np.shape(z)[0]  # Altura
    width = np.shape(xroi)[0]

    [X, Z, E] = np.meshgrid(xroi, z, xt)  # Largura

    dist = np.sqrt((X - E) ** 2 + Z ** 2)
    f = np.zeros((height, width))
    delay = (2 * dist / cl) - t_init
    t = np.round(delay / ts).astype(int)
    t = np.minimum(t, g.shape[0] - 1)  # Cria matriz da imagem

    for elem in range(len(xt)):  # Percorrendo as pos do transdutor
        f += g[t[:, :, elem], elem]

    return f


plt.figure()
plt.imshow(g, aspect="auto")
plt.title('B-Scan')

plt.figure()
start = time.time()
f = saft_2(g, speed_m_s, elem_positions_mm, z, ts, samples_t_init_microsec)
print(time.time() - start)
plt.imshow(np.log10(abs(f) + 0.1), aspect="auto",
           extent=[roi_ori[1], roi_ori[1] + roi_width, roi_ori[0] + roi_height, roi_ori[0]])
plt.title('SAFT 2.0')
plt.show()