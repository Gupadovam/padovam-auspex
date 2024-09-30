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

experiment_root = "/media/gustavopadovam/TOSHIBA EXT/SmartWedge/focal_laws_res_test"
experiment_ref = f"/ref/ref_0_1.m2k"

data_ref = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

t_span_original = data_ref.time_grid

# Operações
log_cte = .5

# Imagem pos 0 para ajuste das boxes
img = np.log10(envelope(np.sum(data_ref.ascan_data, axis=2), axis=0) + log_cte)

# Definir mesma colorbar:
vmin_sscan = 0
vmax_sscan = 5.5

api_vec = np.zeros(16)
maxAng = np.zeros_like(api_vec)
maxPixel = np.zeros_like(api_vec)

m=1
betas_dict = {
    1: np.linspace(-40, 40, 901),
    2: np.linspace(-40, 40, 451),
    3: np.linspace(-40, 40, 301),
    4: np.linspace(-40, 40, 226),
    5: np.linspace(-40, 40, 181)
}

for i in range (1,6):
    experiment_name = f"/0_{i}.m2k"
    data_experiment= file_m2k.read(experiment_root+ experiment_name, type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
    experiment_ref = f"/ref/ref_0_{i}.m2k"
    data_ref = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

    # Faz a operação de somatório + envoltória:
    sscan_exp = envelope(np.sum(data_experiment.ascan_data - data_ref.ascan_data, axis=2), axis=0)
    sscan_exp_log = np.log10(sscan_exp + log_cte)

    t_span = data_experiment.time_grid[:,0]
    ascans_data = data_experiment.ascan_data[:,:,:,0]
    s_scan = ascans_data.sum(axis=2)
    s_scan /= np.abs(s_scan).max()
    rspan_array = convert_time2radius(t_span, 54.37, 60.22, sw.cl_steel, sw.coupling_cl)
    angulos_array = np.linspace(-40, 40, s_scan.shape[1])


    corners = [(63,-1), (58,4)]
    api_vec[i], maxAngIdx, img_masked, maxPixel[i] = imaging_utils.api_func_polar(s_scan, rspan_array, angulos_array,corners)
    betas = betas_dict[i]
    maxAng[i] = betas[maxAngIdx]

    plt.subplot(2, 5, m)
    plt.title(f"S-scan da Lei focal com Step 0.{i}")
    plt.imshow(sscan_exp_log, extent=[-40, 40, rspan_array[-1], rspan_array[0]], cmap='magma', aspect='auto',
               interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)
    plt.ylim(57,68)
    if i == 1:
        plt.ylabel(r"raio em mm")
        plt.xlabel(r"Ângulo de varredura da tubulação")

    plt.subplot(2, 5, m + 5)
    plt.title(f"API={api_vec[i]:.4f} mm²")
    plt.imshow(img_masked, extent=[-40, 40, rspan_array[-1], rspan_array[0]], aspect='auto', interpolation="None")

    if i == 1:
        plt.ylabel(r"raio em mm")
        plt.xlabel(r"Ângulo de varredura da tubulação")
    m+=1
    plt.show()
