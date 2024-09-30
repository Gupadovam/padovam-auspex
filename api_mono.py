from framework import file_m2k
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from bisect import bisect
from framework.post_proc import envelope
from smartwedge import imaging_utils
from smartwedge.smartwedge import Smartwedge
from smartwedge.sw_models import *
from smartwedge.imaging_utils import *
sw = SmartwedgeCompacta()

experiment_root = "/media/gustavopadovam/TOSHIBA EXT/SmartWedge/API_Mono/"
experiment_ref = "Furo_2.m2k"

data = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=None)
vmax= 5.5
t_span = data.time_grid[:,0]
a_scan = data.ascan_data[:,0,0,0]
s_scan = data.ascan_data[:,0,0,:]
s_scan /= np.abs(s_scan).max()
angulo = (25*0.5) / 2
angulos_mono = np.linspace(-angulo, angulo, s_scan.shape[1])

rspan_mono = convert_time2radius(t_span, 14.97, 21.11, sw.cl_steel, sw.coupling_cl)

corners = [(67,-4), (54,4)]
api, maxAngIdx, img_masked, maxPixel= imaging_utils.api_func_polar(s_scan,rspan_mono,angulos_mono,corners)


plt.subplot(1,2,1)
plt.title('S_SCAN do MONO')
plt.ylabel('raio em mm')
plt.xlabel('Ângulo de varredura da tubulação')
plt.imshow(np.log10(envelope(s_scan, axis=0) + 1e-5), aspect='equal', extent= (-angulo, angulo , rspan_mono[-1], rspan_mono[0]))

plt.subplot(1,2,2)
plt.imshow(img_masked, aspect='equal', extent= (-angulo, angulo , rspan_mono[-1], rspan_mono[0]), interpolation= "NONE")
plt.title(f"Api = {api:.2f} mm²")





# experiment_root = "/media/gustavopadovam/USB/API_as_built_imma/"
# experiment_ref = f"pos_1.m2k"
#
# dados = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
#                                       bw_transd=0.5, tp_transd='gaussian', sel_shots=None)
#
# t_span = dados.time_grid[:,0]
# ascans_data = dados.ascan_data[:,:,:,0]
# s_scan = ascans_data.sum(axis=2)
# s_scan /= np.abs(s_scan).max()
# rspan_array = convert_time2radius(t_span, 54.37, 60.22, sw.cl_steel, sw.coupling_cl)
# angulos_array = np.linspace(-40, 40, s_scan.shape[1])
#
# corners = [(64,-1), (58,6)]
# api, maxAngIdx, img_masked, maxPixel= imaging_utils.api_func_polar(s_scan,rspan_array,angulos_array,corners)
#
# plt.subplot(2,2,2)
# plt.title('S_SCAN')
# plt.ylabel('raio(mm)')
# plt.xlabel('Ângulo de varredura da tubulação')
# plt.imshow(np.log10(envelope(s_scan,axis=0)+1e-3), aspect='equal', extent= (-40,40,rspan_array[-1],rspan_array[0]))
#
# plt.subplot(2,2,4)
# plt.imshow(img_masked, aspect='equal', extent= (-40, 40 , rspan_array[-1], rspan_array[0]), interpolation= "NONE")
# plt.title(f"Api = {api:.2f} mm²")
#
# plt.show()
#

experiment_root = "/media/gustavopadovam/USB/API_as_built_imma/"
experiment_ref = "ref.m2k"
betas = np.linspace(-40, 40, 181)

data_ref = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

t_span_original = data_ref.time_grid

# Operações
log_cte = .5

# Imagem pos 0 para ajuste das boxes
img = np.log10(envelope(np.sum(data_ref.ascan_data, axis=2), axis=0) + log_cte)

# Corta o scan e timegrid para range desejado:
t0 = 50
tend = 50 + 15

# Corta o A-scan para limites definidos:
# data_ref.ascan_data = crop_ascan(data_ref.ascan_data, t_span_original, t0, tend)

# New t_span
t_span = crop_ascan(t_span_original, t_span_original, t0, tend)

# Definir mesma colorbar:
vmin_sscan = 0
vmax_sscan = 5.5

api_vec = np.zeros(16)
maxAng = np.zeros_like(api_vec)
maxPixel = np.zeros_like(api_vec)
m = 0

plt.figure(figsize=(10, 5))
for i in range(0, 16):
    print(f"Progresso:{i/16 * 100:.1f} %")
    j = i + 1
    experiment_name = f"pos_{i}.m2k"
    data_experiment = file_m2k.read(experiment_root + experiment_name, type_insp='contact', water_path=0, freq_transd=5,
                                      bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

    # Corta o A-scan para limites definidos:
    # data_experiment.ascan_data = crop_ascan(data_experiment.ascan_data, t_span_original, t0, tend)

    # Faz a operação de somatório + envoltória:
    sscan_exp = envelope(np.sum(data_experiment.ascan_data - data_ref.ascan_data, axis=2), axis=0)
    sscan_exp_log = np.log10(sscan_exp + log_cte)

    t_span = data_experiment.time_grid[:,0]
    ascans_data = data_experiment.ascan_data[:,:,:,0]
    s_scan = ascans_data.sum(axis=2)
    s_scan /= np.abs(s_scan).max()
    rspan_array = convert_time2radius(t_span, 54.37, 60.22, sw.cl_steel, sw.coupling_cl)
    angulos_array = np.linspace(-40, 40, s_scan.shape[1])

    corners = [(64, angulos_array[20]+ i * 3.997), (58, angulos_array[60]+i*3.997)]
    api_vec[i], maxAngIdx, img_masked, maxPixel[i] = imaging_utils.api_func_polar(s_scan,rspan_array,angulos_array,corners)
    maxAng[i] = betas[maxAngIdx]

    if j % 2 == 0:
        m += 1
        if m == 5:
            pass
        print(m)
        plt.subplot(2, 8, m)
        plt.title(f"S-scan da posição {i}")
        plt.imshow(sscan_exp_log, extent=[-40, 40, rspan_array[-1], rspan_array[0]], cmap='magma', aspect='auto',
                   interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)
        plot_echoes(57.8, 0, n_echoes=1, color='blue', xbeg=-40, xend=40)

        if j == 1:
            plt.ylabel(r"raio em mm")
            plt.xlabel(r"Ângulo de varredura da tubulação")

        plt.subplot(2, 8, m + 8)
        plt.title(f"API={api_vec[i]:.4f} mm²")
        plt.imshow(img_masked, extent=[-40, 40, rspan_array[-1], rspan_array[0]], aspect='auto', interpolation="None")

        if i == 1:
            plt.ylabel(r"raio em mm")
            plt.xlabel(r"Ângulo de varredura da tubulação")

plt.figure(figsize=(10, 5))
plt.title("API em função do ângulo de varredura.")
plt.plot(maxAng[:-3], api_vec[:-3], 'o:', label="API SW")
valor = api
vetor = np.full(16, valor)
plt.plot(maxAng[:-3], vetor[:-3], 'o:', label= "API mono = 0,57mm²")
plt.xticks(maxAng[:-3],  rotation=90, )
plt.ylabel("Área em mm²")
plt.xlabel(r"Ângulo de varredura da tubulação")
plt.grid()
plt.legend()

plt.figure(figsize=(10, 5))
plt.title("Valor do máximo de intensidade do pixel (não está em escala log).")
plt.plot(maxAng, maxPixel, 'o:', color='r')
plt.xticks(maxAng)
plt.ylabel("Intensidade do pixel")
plt.xlabel(r"Ângulo de varredura da tubulação")
plt.grid()
plt.show()