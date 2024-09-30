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


plt.subplot(2,2,1)
plt.title('S_SCAN do MONO')
plt.ylabel('raio(mm)')
plt.xlabel('Ângulo de varredura da tubulação')
plt.imshow(np.log10(envelope(s_scan, axis=0) + 1e-5), aspect='equal', extent= (-angulo, angulo , rspan_mono[-1], rspan_mono[0]))

plt.subplot(2,2,3)
plt.imshow(img_masked, aspect='equal', extent= (-angulo, angulo , rspan_mono[-1], rspan_mono[0]), interpolation= "NONE")
plt.title(f"Api = {api:.2f} mm²")















experiment_root = "/media/gustavopadovam/USB/API_as_built_imma/"
experiment_ref = f"pos_6.m2k"

dados = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                                      bw_transd=0.5, tp_transd='gaussian', sel_shots=None)

t_span = dados.time_grid[:,0]
ascans_data = dados.ascan_data[:,:,:,0]


s_scan = ascans_data.sum(axis=2)
s_scan /= np.abs(s_scan).max()

plt.subplot(2,2,2)
plt.title('S_SCAN')

plt.imshow(np.log10(envelope(s_scan,axis=0)+1e-3), aspect='equal', extent= (-49,49,t_span[-1],t_span[0]))
# plt.xlim([-angulo*1.2,angulo*1.2])
plt.show()