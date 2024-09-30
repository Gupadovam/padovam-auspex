import numpy as np
from framework import file_m2k
import matplotlib
from tqdm import tqdm
from bisect import bisect
matplotlib.use('TkAgg')
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from framework.post_proc import envelope


experiment_root = "/home/gustavopadovam/ENSAIOS/Varredura_circunferencial_full/40db.m2k"
experiment_ref = ""
dados = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                      bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

t_span = dados.time_grid[:,0]
ascans_data = dados.ascan_data
s_scan = ascans_data.sum(axis=2)
s_scan /= np.abs(s_scan).max()

plt.figure()
plt.title('S_SCAN')
plt.xlim(81.7,277.5)
plt.ylim(3.0e03, 1.8e03)
plt.imshow(np.log10(envelope(s_scan,axis=0)+1e-3), aspect='auto')
plt.show()