import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

path = "/media/gustavopadovam/USB20FD/dados_tfm.npy"
dados = np.load(path, allow_pickle=True).item()
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
roi_res = (500, 64)  # Define a resolução da ROI

z = np.linspace(roi_ori[0], roi_ori[0] + roi_height, roi_res[0])  # Convertendo para posição
xroi = np.linspace(roi_ori[1], roi_ori[1] + roi_width, roi_res[1])  # Convertendo para posição


def saft_2(g, cl, xt, z, ts, t_init):
    height = np.shape(z)[0]  # Define altura da ROI
    width = np.shape(xroi)[0]  # Define a largura da ROI

    [X, Z] = np.meshgrid(xroi, z)  # Cria matrizes de posições
    f = np.zeros((height, width))  # Cria matriz vazia do tamanho da ROI

    for elem in range(len(xt)):  # Loop para percorrer os elementos
        dist = np.sqrt((X - xt[elem]) ** 2 + Z ** 2)  # Calculo da distancia
        delay = (2 * dist / cl) - t_init  # Calculo do delay
        t = np.round(delay / ts).astype(int)  # Calculo da amostra correspondente
        t = np.minimum(t, g.shape[0] - 1)  # Pegar apenas o valor minimo
        f += g[t, elem]  # Soma o correspondente de g em f

    return f


plt.figure()
plt.imshow(g, aspect="auto")
plt.title('B-Scan')

plt.figure()
start = time.time()
f = saft_2(g, speed_m_s, elem_positions_mm, z, ts, samples_t_init_microsec)
plt.imshow(np.log10(abs(f) + 1e-3), aspect="auto",
           extent=[roi_ori[1], roi_ori[1] + roi_width, roi_ori[0] + roi_height, roi_ori[0]])
plt.title('SAFT')
print(f"The script took {time.time() - start:.2f}s to run!!!")
plt.show()