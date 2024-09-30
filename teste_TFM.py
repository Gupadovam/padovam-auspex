import numpy as np
from framework import file_m2k
import matplotlib
matplotlib.use('TkAgg')
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import time

experiment_root ="/home/gustavopadovam/ENSAIOS/FMC_furo.m2k"
experiment_ref =""
dados = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                        bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

ascans_data = dados.ascan_data
speed_of_sound = 5900
sampling_frequency_MHz = dados.inspection_params.sample_freq * 1e6
initial_sample_times_microsec =dados.inspection_params.gate_start * 1e-6
element_positions_mm = dados.probe_params.elem_center[:,0]

sampling_time = 1 / sampling_frequency_MHz

roi_origin = np.load('/home/gustavopadovam/ENSAIOS/ROI_FMC/roi_pts_R70_R48_ang90.npy')*1e3

times = np.load('/home/gustavopadovam/ENSAIOS/ROI_FMC/times_R70toR48_ang90_64elem.npy')*1e6

z_roi = roi_origin[1,:].reshape([200,200])
x_roi = roi_origin[0,:].reshape([200,200])

def tfm_calculation(g, cl, t_init, xt, z, x, ts):
    Width = np.shape(xt)[0]  # Largura
    Height = np.shape(z)[0]

    tfm_image = np.zeros((Height, Width))

    for i, roi_x in enumerate(x):
        for j, roi_z in enumerate(z):
            for emi in range(Width):
                for trans in range(Width):
                    dist1 = times[i,j,emi]
                    dist2 =  times[i,j,trans]
                    dist = dist1+ dist2
                    delay = dist / cl - t_init
                    amostra = np.round((delay / ts)).astype(int)  # Calculo da amostra
                    amostra = np.minimum((amostra), g.shape[0] - 1)
                    tfm_image += g[amostra, emi, trans]
    tfm_image = np.abs(hilbert(tfm_image, axis=0))
    return tfm_image

plt.figure()
start_time = time.time()
tfm_image = tfm_calculation(ascans_data, speed_of_sound, initial_sample_times_microsec, element_positions_mm, z_roi, x_roi, sampling_time)
plt.imshow(np.log10(abs(tfm_image) + 1e-3), aspect="auto",
           extent=[roi_origin[1], roi_origin[1] + 200, roi_origin[0] + 200, roi_origin[0]])
plt.title('Total Focusing Method (TFM)')
print(f'A script demorou {time.time() - start_time:.2f}s para terminar!')
plt.show()