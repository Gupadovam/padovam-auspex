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

experiment_root = "/home/gustavopadovam/Área de Trabalho/Ensaio_XL"
experiment_name = "/RASGOS-XL/pos360.m2k"
data_experiment = file_m2k.read(experiment_root + experiment_name, type_insp='contact', water_path=0, freq_transd=5,
                                bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
experiment_ref = "/ref_XL.m2k"
data_ref = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                         bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

log_cte = 1e-6

vmin_sscan = 0
vmax_sscan = 5.5

t_span= data_experiment.time_grid[:,0]

# Faz a operação de somatório + envoltória:
sscan_exp = envelope(np.sum(data_experiment.ascan_data - data_ref.ascan_data, axis=2), axis=0)
sscan_exp_log = np.log10(sscan_exp + log_cte)
plt.ylabel('tempo(µs)')
plt.xlabel('Ângulo')
plt.imshow(sscan_exp_log, extent=[-45, 45, t_span[-1], t_span[0]], cmap='magma', aspect='auto',
           interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)

plt.show()