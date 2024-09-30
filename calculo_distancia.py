from framework import file_m2k
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from bisect import bisect
from framework.post_proc import envelope

experiment_root = "/home/gustavopadovam/ENSAIOS/mono_elem/"
experiment_ref = "mono_a110S_5_.m2k"
# experiment_ref = "mono_a110S_5_furo.m2k"
# experiment_ref = "mono_V110_5_.m2k"
# experiment_ref = "mono_V110_5_furo.m2k"

data = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=None)

t_span = data.time_grid[:,0]


# Defina os limites do intervalo y
intervalo_x_inicio01 = bisect(t_span, 10)
intervalo_x_fim01 = bisect(t_span, 30)
intervalo_x_inicio02 = bisect(t_span, 30)
intervalo_x_fim02 = bisect(t_span, 60)


ascan1 = data.ascan_data[intervalo_x_inicio01:intervalo_x_fim01, 0, 0, 3]
ascan2 = data.ascan_data[intervalo_x_inicio02:intervalo_x_fim02, 0, 0, 3]
# Encontre o índice do máximo valor dentro do intervalo y
maximo1 = np.max(ascan1)
maximo2 = np.max(ascan2)
indices1 = np.where(ascan1 == maximo1)[0][0]
indices2= np.where(ascan2 == maximo2)[0][0]

t_max1 = t_span[intervalo_x_inicio01+indices1]
t_max2 = t_span[intervalo_x_inicio02+indices2]

# velocidade = 2dist / t
vel = 1480
t = (t_max2-t_max1)
dist = (vel * t) / 2 *1e-3
print(dist)
print(t)

plt.figure()
plt.plot(t_span, data.ascan_data[:,0,0,3])
plt.plot(t_max1, maximo1, 'ro')
plt.plot(t_max2, maximo2, 'ro')

plt.figure()
plt.plot(t_span[2000:6000], data.ascan_data[2000:6000, 0, 0, 3])
plt.plot(t_max1, maximo1, 'ro')
plt.plot(t_max2, maximo2, 'ro')
plt.text(t_max1, maximo1, t_max1, fontsize=10, color='b', verticalalignment='bottom', horizontalalignment='right')
plt.text(t_max2, maximo2, t_max2, fontsize=10, color='b', verticalalignment='bottom', horizontalalignment='right')
plt.text(49, 4500, f'Diferença dos tempos: {t:.3f}', fontsize=10, color='black', verticalalignment='bottom', horizontalalignment='right')
plt.text(49, 4000, f'Distância calculada: {dist:.3f}mm', fontsize=10, color='black', verticalalignment='bottom', horizontalalignment='right')


plt.show()


