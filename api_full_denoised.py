from framework import file_m2k
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from bisect import bisect
from framework.post_proc import envelope
# Script faz a análise usando API do ensaio deslocando o furo para margem da imagem.

def crop_ascan(ascan, t_span, t0=None, tf=None):
    if t0 is not None and tf is not None:
        t0_idx = np.argmin(np.power(t_span - t0, 2))
        tf_idx = np.argmin(np.power(t_span - tf, 2))
        return ascan[t0_idx:tf_idx, :]


def plot_echoes(t_base, t_echoes, n_echoes=3, color='blue', label='_', xbeg=-40, xend=40, alpha=.3):
    x = np.arange(xbeg, xend, 1e-1)
    for n in range(n_echoes):
        y = np.ones_like(x) * (t_base + t_echoes * (n + 1))
        plt.plot(x, y,  ':', color=color, label=label, alpha=alpha)
    if label != "_":
        plt.legend()

def api_func(img, corners, thresh=.5, drawSquare=True):
    north_east_corner = corners[0]
    south_west_corner = corners[1]
    img_cropped = img[north_east_corner[0]:south_west_corner[0], north_east_corner[1]:south_west_corner[1]]
    local_max = np.max(img_cropped)
    maxLocationCoord = np.where(img_cropped==local_max)
    maxLocation = maxLocationCoord[1] + north_east_corner[1]
    img_cropped_masked = img_cropped > thresh * local_max
    img_masked = np.zeros_like(img)
    img_masked[north_east_corner[0]:south_west_corner[0], north_east_corner[1]:south_west_corner[1]] += img_cropped_masked
    api = np.sum(img_masked * 1.0) / len(img_masked)

    if drawSquare:
        width = 1
        scale_factor = int(img.shape[0]/img.shape[1])
        img_masked[north_east_corner[0] - width * scale_factor : south_west_corner[0] + width * scale_factor,
                   north_east_corner[1] - width : north_east_corner[1] + width] = 1

        img_masked[north_east_corner[0] - width * scale_factor : south_west_corner[0] + width * scale_factor,
                   south_west_corner[1] - width: south_west_corner[1] + width] = 1

        img_masked[north_east_corner[0] - width * scale_factor : north_east_corner[0] + width * scale_factor,
                   north_east_corner[1] - width : south_west_corner[1] + width] = 1

        img_masked[south_west_corner[0] - width * scale_factor : south_west_corner[0] + width * scale_factor,
                   north_east_corner[1] - width: south_west_corner[1] + width] = 1

    return api, maxLocation, img_masked, local_max
# Análise dos ascans:

experiment_root = "/home/gustavopadovam/ENSAIOS/Varredura_circunferencial_full/"
experiment_ref = f"sw_full_ref.m2k"
betas = np.linspace(-90, 90, 161)

data_ref = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)[0]

ang_span_original = np.linspace(-90, 90, 361)
t_span_original = data_ref.time_grid[:, 0]

# Definir mesma colorbar:
vmin_sscan = 0
vmax_sscan = 5.5

# Corta o scan e timegrid para range desejado:
t0 = bisect(t_span_original, 60)
tend = bisect(t_span_original, 70)

# Corta somente para região de inc direta

ang0 = bisect(ang_span_original, -49)
ang_end = bisect(ang_span_original, 49)

# plt.imshow(sscan_exp_log, extent=[-49, 49, t_span[-1], t_span[0]], cmap='magma', aspect='auto',
#                    interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)

# Corta o A-scan para limites definidos:
data_ref.ascan_data = data_ref.ascan_data[t0:tend, ang0:ang_end, :, :]

# New t_span and ang_span
t_span = t_span_original[t0:tend]
ang_span = ang_span_original[ang0:ang_end]
# Operações
log_cte = 1e-6

# Definir mesma colorbar:
vmin_sscan = 0
vmax_sscan = 5.5

# Cantos das caixas que irão conter as falhas:
corners = [
    [(962, 60), (1212, 95)],  # [(Row, column), (Row, column)] and [(North-east), (South-west)]
    [(962, 70), (1212, 120)],  #
    [(962, 80), (1212, 120)],
    [(962, 100), (1212, 140)],
    [(962, 110), (1212, 160)],
    [(962, 110), (1212, 160)]
]

api_vec = np.zeros(32)
maxAng = np.zeros_like(api_vec)
maxPixel = np.zeros_like(api_vec)
m = 0

plt.figure()
for i in range(10,0,-1):
    j = i + 1
    data = np.load(f'/home/gustavopadovam/ENSAIOS/Matriz_empilhada_denoised_shot{i}.npy')

    # Corta o scan e timegrid para range desejado:
    t0 = bisect(t_span_original, 60)
    tend = bisect(t_span_original, 70)

    # Corta somente para região de inc direta

    ang0 = bisect(ang_span_original, -49)
    ang_end = bisect(ang_span_original, 49)

    # New t_span and ang_span
    t_span = t_span_original[t0:tend]
    ang_span = ang_span_original[ang0:ang_end]

    # Faz a operação de somatório + envoltória:
    sscan_exp = envelope(np.sum(data, axis=2), axis=0)
    sscan_exp_log = np.log10(sscan_exp + log_cte)

    # APlicação da API:
    corners = [(314, 90 + (10 - i) * 8), (452, 120 + (10 - i) * 8)]
    api_vec[10 - i], maxAngIdx, img_masked, maxPixel[10 - i] = api_func(sscan_exp, corners, thresh=.5)
    maxAng[10 - i] = ang_span[maxAngIdx]

    if (j % 2 == 0) or (j == 11):
        m += 1
        if m == 5:
            pass
        print(m)
        plt.subplot(2, 8, m)
        plt.title(f"S-scan da posição {j}")
        plt.imshow(sscan_exp_log, extent=[-49, 49, t_span[-1], t_span[0]], cmap='magma', aspect='auto',
                   interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)
        plot_echoes(61.75, 0, n_echoes=1, color='blue', xbeg=-40, xend=40)

        if j == 1:
            plt.ylabel(r"Tempo em $\mu s$")
            plt.xlabel(r"Ângulo de varredura da tubulação")

        plt.subplot(2, 8, m + 8)
        plt.title(f"API={api_vec[22 - i]:.4f}")
        plt.imshow(img_masked, extent=[-49, 49, t_span[-1], t_span[0]], aspect='auto', interpolation="None")

        if i == 1:
            plt.ylabel(r"Tempo em $\mu s$")
            plt.xlabel(r"Ângulo de varredura da tubulação")

plt.figure()
plt.title("API em função do ângulo de varredura.")
plt.plot(maxAng, api_vec, 'o:')
plt.xticks(maxAng)
plt.ylabel("API")
plt.xlabel(r"Ângulo de varredura da tubulação")
plt.grid()

plt.figure()
plt.title("Valor do máximo de intensidade do pixel (não está em escala log).")
plt.plot(maxAng, maxPixel, 'o:', color='r')
plt.xticks(maxAng)
plt.ylabel("Intensidade do pixel")
plt.xlabel(r"Ângulo de varredura da tubulação")
plt.grid()

plt.show()
