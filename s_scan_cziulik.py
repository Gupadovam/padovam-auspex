import numpy as np
from framework import file_m2k
import matplotlib
from tqdm import tqdm
from bisect import bisect
matplotlib.use('TkAgg')
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from framework.post_proc import envelope


experiment_root = "/media/gustavopadovam/TOSHIBA EXT/SmartWedge/Meia Cana Cziulik/posiçao_10.m2k"
experiment_ref = ""
dados = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                      bw_transd=0.5, tp_transd='gaussian', sel_shots = 0)
t_span= dados.time_grid[:,0]
ascan_data = dados.ascan_data
s_scan = dados.ascan_data_sum
vmax = 5.5
angulo = 26.601611916788222
t_inicial = 46.73
gate = 19.872
plt.figure()
plt.title('S_SCAN do Multiplos Elem')
plt.xlim([-angulo,angulo])
plt.ylim([t_inicial+gate, t_inicial])
plt.ylabel('tempo(µs)')
plt.xlabel('Ângulo de varredura da tubulação')
plt.imshow(np.log10(envelope(s_scan,axis=0)+.5), aspect='equal',extent = (-40,40, t_span[-1], t_span[0]), vmax=vmax)
plt.show()