import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

################ PARÂMETROS ################
# Parâmetros da cunha inteligente
cL = 6400                   # Velocidade das ondas longitudinais (L)
cT = cL / 2                 # Velocidade das ondas transversais (T)
r_circ = 0.07                # Raio da superfície externa do tubo
d = 0.21                     # Distância do centro do tubo (origem) ao transdutor
ang_ref = 90 * np.pi / 180  # Ângulo no qual z é forçado
z_ref = 1.2 * r_circ        # Valor forçado de z em ang_ref

############################################

x_ref = z_ref * np.sin(ang_ref)
y_ref = z_ref * np.cos(ang_ref)
r_ref = np.sqrt(x_ref**2 + (d - y_ref)**2)
t_TL = r_ref / cT + z_ref / cL
t_LL = r_ref / cL + z_ref / cL

################ DEFINIÇÃO DE TODAS AS FUNÇÕES ÚTEIS ################
def raizes_bhaskara(a, b, c):
  '''Calcula as raízes do polinômio ax^2 + bx + c = 0'''
  if a == 0:
    return -c/b, -c/b

  delta = np.sqrt(b**2 - 4*a*c)
  x1 = (-b + delta) / (2*a)
  x2 = (-b - delta) / (2*a)
  return x1, x2

def z_r_de_alpha(alpha, c1, c2, d, t):
  '''Calcula os valores de z e r para um dado alpha'''
  a = 1 -c1**2 / c2**2
  b = - 2 * d * np.cos(alpha) + 2 * t * c1**2 / c2
  c = d**2 - c1**2 * t**2
  z = raizes_bhaskara(a, b, c)[0]
  r = np.sqrt(z**2 + d**2 - 2*z*d*np.cos(alpha))
  return z, r

def x_y_de_alpha(alpha, c1, c2, d, t):
  '''Calcula as coordenadas (x,y) da lente para um dado alpha'''
  z, r = z_r_de_alpha(alpha, c1, c2, d, t)
  y = z * np.cos(alpha)
  x = z * np.sin(alpha)
  return x, y

# Espelho TL
alpha = np.linspace(0, np.pi/2, 101)
z, r = z_r_de_alpha(alpha, cT, cL, d, t_TL)
x_mirror, y_mirror = x_y_de_alpha(alpha, cT, cL, d, t_TL)
alpha_circ = np.linspace(-np.pi/2, np.pi/2, 201)
x_circ = r_circ * np.sin(alpha_circ)
y_circ = r_circ * np.cos(alpha_circ)
plt.plot(x_mirror[y_mirror<=d], y_mirror[y_mirror<=d], '--k', label='Espelho TL')
plt.plot(-x_mirror[y_mirror<=d], y_mirror[y_mirror<=d], '--k')
plt.plot(x_circ, y_circ, '-C0')
plt.plot([0], [0], 'or')
plt.plot([0], [d], 'sk')
plt.axis('equal')
angulo_exemplo = np.pi*.4
z_exemplo, r_exemplo = z_r_de_alpha(angulo_exemplo, cT, cL, d, t_TL)
x_exemplo, y_exemplo = x_y_de_alpha(angulo_exemplo, cT, cL, d, t_TL)
plt.plot([0, x_exemplo, 0], [0, y_exemplo, d], 'green', alpha=.5)

# Espelho LL
z, r = z_r_de_alpha(alpha, cL, cL, d, t_LL)
x_mirror, y_mirror = x_y_de_alpha(alpha, cL, cL, d, t_LL)
alpha_circ = np.linspace(-np.pi/2, np.pi/2, 201)
x_circ = r_circ * np.sin(alpha_circ)
y_circ = r_circ * np.cos(alpha_circ)
plt.plot(x_mirror[y_mirror<=d], y_mirror[y_mirror<=d], ':k', label='Espelho LL')
plt.plot(-x_mirror[y_mirror<=d], y_mirror[y_mirror<=d], ':k')
plt.plot(x_circ, y_circ, '-C0')
x_exemplo, y_exemplo = x_y_de_alpha(angulo_exemplo, cL, cL, d, t_LL)
plt.plot([0, x_exemplo, 0], [0, y_exemplo, d], 'green', alpha=.5)
plt.legend()
plt.show()
