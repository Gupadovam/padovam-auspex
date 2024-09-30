import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sci
from framework.data_types import ImagingROI
from framework import file_m2k, file_law
from framework.post_proc import normalize, envelope
from imaging import tfm

plt.close('all')

files_speed = [r'D:\Ecos_laterais\ecos_espessura\pc_1_vel.m2k',
               r'D:\Ecos_laterais\ecos_espessura\pc_2_vel.m2k',
               r'D:\Ecos_laterais\ecos_espessura\pc_3_vel.m2k',
               r'D:\Ecos_laterais\ecos_espessura\pc_4_vel.m2k',
               r'D:\Ecos_laterais\ecos_espessura\pc_5_vel.m2k']
files_thickness = [r'D:\Ecos_laterais\ecos_espessura\pc_1_hor.m2k',
                   r'D:\Ecos_laterais\ecos_espessura\pc_2_hor.m2k',
                   r'D:\Ecos_laterais\ecos_espessura\pc_3_hor.m2k',
                   r'D:\Ecos_laterais\ecos_espessura\pc_4_hor.m2k',
                   r'D:\Ecos_laterais\ecos_espessura\pc_5_hor.m2k']

size = [[80, 60, 24],
        [249, 80, 20.2],
        [200, 200, 20.3],
        [195, 195, 20],
        [200, 120, 30]]

names = ['pc_giovanni', 'bp_solda', 'pc_victor_au', 'pc_victor_aço', 'sapata']

# Lê os dados
for i in range(1):#len(files_speed)):
    i=3
## Calcula a velocidade - menor espessura
    data = file_m2k.read(files_speed[i], freq_transd=5, bw_transd=0.5, tp_transd='gaussian',
                         sel_shots=3, read_ascan=True, type_insp="contact")
    b_scan = envelope(np.diagonal(data[0].ascan_data[..., 0], axis1=1, axis2=2))
    plt.figure()
    plt.imshow(np.log10(b_scan + 1e-10), aspect='auto')
    plt.xlabel('Element')
    plt.ylabel(r'Time [$\mu$s]')
    plt.title(f'{names[i]} - To calculate sound speed')

    if i == 0:
        sum_ax1 = normalize(np.sum(b_scan, axis=1)[500:4800])
        peaks, _ = sci.find_peaks(sum_ax1, height=0.050, distance=800)

    elif i==2:
        sum_ax1 = normalize(np.sum(b_scan, axis=1)[500:6000])
        peaks, _ = sci.find_peaks(sum_ax1, height=0.050, distance=600)

    plt.figure()
    plt.plot(sum_ax1)
    plt.plot(peaks, sum_ax1[peaks], "x")
    plt.plot(np.zeros_like(sum_ax1), "--", color="gray")
    plt.show()

    dist = peaks - np.roll(peaks, 1)
    dist_mean = np.mean(dist[1::])

    c_spc = 2000 * size[i][2] / (dist_mean * data[0].inspection_params.sample_time)
    print('cl=', c_spc)

## Calcula a distância entre ecos múltiplos - maior espessura
    data = file_m2k.read(files_thickness[i], freq_transd=5, bw_transd=0.5, tp_transd='gaussian',
                         sel_shots=3, read_ascan=True, type_insp="contact")

    # b_scan = envelope(np.diagonal(data[0].ascan_data[..., 0], axis1=1, axis2=2))
    #
    # plt.figure()
    # plt.suptitle(f'{names[i]}')
    # plt.subplot(121)
    # x_inf = 0
    # x_sup = 63
    # y_inf = data[0].time_grid[-1, 0]  # *1e-3*c_spc/2
    # y_sup = data[0].time_grid[0, 0]  # *c_spc*2
    # plt.imshow(np.log10(b_scan + 1e-10), aspect='auto', extent=[x_inf, x_sup, y_inf, y_sup])

    # Define a ROI
    # corner_roi = np.array([-10.0, 0.0, size[i][0]-10])[np.newaxis, :]  # [x0, y0, z0]
    corner_roi = np.array([-10.0, 0.0, 0])[np.newaxis, :]  # [x0, y0, z0]

    roi = ImagingROI(corner_roi, height=2*size[i][0], width=20.0, h_len=4 * 120, w_len=4 * 20)
    x_inf = corner_roi.min()
    y_sup = corner_roi.max()
    x_sup = x_inf + roi.width
    y_inf = y_sup + roi.height

    chave = tfm.tfm_kernel(data[1], roi=roi, sel_shot=0, c=c_spc)
    tfm = data[1].imaging_results[chave].image
    tfm_log = np.log10(normalize(envelope(tfm)) + 1e-10)
    # plt.subplot(121)
    plt.title('TFM')
    plt.imshow(tfm_log, aspect='auto', extent=[x_inf, x_sup, y_inf, y_sup])

    sum_tfm = normalize(envelope(np.sum(tfm, axis=1), axis=0))

    if i==0:
        peaks_tfm, _ = sci.find_peaks(sum_tfm, height=0.02, distance=45)
    elif i==2:
        peaks_tfm, _ = sci.find_peaks(sum_tfm, height=0.02, distance=18)
    step = (y_inf - y_sup) / tfm.shape[0]
    dist_tfm = np.arange(y_sup, y_inf, step)

    plt.subplot(122)
    plt.title(f'Distance')
    plt.plot(dist_tfm, sum_tfm)
    for k in range(len(peaks_tfm)):
        plt.text(dist_tfm[peaks_tfm[k]] + 0.5, sum_tfm[peaks_tfm[k]] + 0.02, f'{dist_tfm[peaks_tfm[k]]:.2f}')
        if k > 0:
            x_mean = (dist_tfm[peaks_tfm[k]] + dist_tfm[peaks_tfm[k - 1]]) / 2
            diff = dist_tfm[peaks_tfm[k]] - dist_tfm[peaks_tfm[k - 1]]
            plt.text(x_mean, -0.1, f'{diff:.2f}')
    plt.ylim([-0.2, 1.1])
    plt.plot(dist_tfm[peaks_tfm], sum_tfm[peaks_tfm], "x")
    plt.xlabel('Distance [mm]')
    plt.ylabel('Amplitude')
    plt.show()

    time = 2e3 * dist_tfm/ c_spc

    plt.figure()
    plt.title(f'{names[i]} - dist')

    plt.plot(time, sum_tfm)
    for k in range(len(peaks_tfm)):
        plt.text(time[peaks_tfm[k]]+0.5, sum_tfm[peaks_tfm[k]]+0.02, f'{time[peaks_tfm[k]]:.2f}')
        if k>0:
            x_mean = (time[peaks_tfm[k]]+time[peaks_tfm[k-1]])/2
            diff = time[peaks_tfm[k]]-time[peaks_tfm[k-1]]
            plt.text(x_mean,-0.1, f'{diff:.2f}')
    plt.ylim([-0.2,1.1])
    plt.plot(time[peaks_tfm], sum_tfm[peaks_tfm], "x")


    dist_art = (time - np.roll(time, 1))[1:4]

    # time_fundo = 2e3 * size[0][0] / c_spc
    # time_1art = 4e3 * np.sqrt((size[0][0] / 2) ** 2 + (size[0][2] / 2) ** 2) / c_spc
    # time_canto_inf = 2e3 * np.sqrt((size[0][0]) ** 2 + (size[0][2]) ** 2) / c_spc
    # time_2art = 2e3 * (2 * np.sqrt((size[0][0] / 4) ** 2 + (size[0][2] / 2) ** 2) + np.sqrt(
    #     (size[0][0] * 2 / 4) ** 2 + (size[0][2]) ** 2)) / c_spc
    # time_3art = 2e3 * (2 * np.sqrt((size[0][0] / 6) ** 2 + (size[0][2] / 2) ** 2) + 2 * np.sqrt(
    #     (size[0][0] * 2 / 6) ** 2 + (size[0][2]) ** 2)) / c_spc