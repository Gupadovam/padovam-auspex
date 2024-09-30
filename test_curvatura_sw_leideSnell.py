import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

################ PARÂMETROS ################
# Parâmetros da cunha inteligente
cL = 6400  # Velocidade das ondas longitudinais (L)
cT = cL * 0.4920  # Velocidade das ondas transversais (T)
r_circ = 0.07  # Raio da superfície externa do tubo
d = 0.21  # Distância do centro do tubo (origem) ao transdutor
ang_ref = 90 * np.pi / 180  # Ângulo no qual z é forçado
z_ref = 1.2 * r_circ  # Valor forçado de z em ang_ref

############################################

x_ref = z_ref * np.sin(ang_ref)
y_ref = z_ref * np.cos(ang_ref)
r_ref = np.sqrt(x_ref ** 2 + (d - y_ref) ** 2)
t_TL = r_ref / cT + z_ref / cL
t_LL = r_ref / cL + z_ref / cL


################ DEFINIÇÃO DE TODAS AS FUNÇÕES ÚTEIS ################
def raizes_bhaskara(a, b, c):
    '''Calcula as raízes do polinômio ax^2 + bx + c = 0'''
    if a == 0:
        return -c / b, -c / b

    delta = np.sqrt(b ** 2 - 4 * a * c)
    x1 = (-b + delta) / (2 * a)
    x2 = (-b - delta) / (2 * a)
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


def calcular_vetor_normal(x, y):
    '''Calcula as normais da curvatura'''
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    ds_dt = np.sqrt(dx_dt ** 2 + dy_dt ** 2)

    dx_ds = dx_dt / ds_dt
    dy_ds = dy_dt / ds_dt

    vetor_normal_x = -dy_ds / np.sqrt(dx_ds ** 2 + dy_ds ** 2)
    vetor_normal_y = dx_ds / np.sqrt(dx_ds ** 2 + dy_ds ** 2)

    return vetor_normal_x, vetor_normal_y


def calcular_ang(vetor1_x, vetor1_y, vetor2_x, vetor2_y):
    produto_escalar = vetor1_x * vetor2_x + vetor1_y * vetor2_y
    norma_vetor1 = np.sqrt(vetor1_x ** 2 + vetor1_y ** 2)
    norma_vetor2 = np.sqrt(vetor2_x ** 2 + vetor2_y ** 2)
    coseno = produto_escalar / (norma_vetor1 * norma_vetor2)
    if all(0 <= cos <= 1 for cos in coseno):
        ang = np.arccos(coseno)
        return ang


# Espelho TL
plt.figure()
alpha = np.linspace(0, np.pi / 2, 301)
z, r = z_r_de_alpha(alpha, cT, cL, d, t_TL)
x_mirror, y_mirror = x_y_de_alpha(alpha, cT, cL, d, t_TL)
alpha_circ = np.linspace(-np.pi / 2, np.pi / 2, 601)
x_circ = r_circ * np.sin(alpha_circ)
y_circ = r_circ * np.cos(alpha_circ)


# Plotagem dos vetores normais usando quiver
vetor_normal_x, vetor_normal_y = calcular_vetor_normal(x_mirror[y_mirror <= d], y_mirror[y_mirror <= d])
plt.quiver(x_mirror[180], y_mirror[180], -vetor_normal_x[65], -vetor_normal_y[65], color='r', scale=15,
           width=0.003,label='Normais a Curvatura')
# plt.quiver(-x_mirror[y_mirror <= d], y_mirror[y_mirror <= d], vetor_normal_x, -vetor_normal_y, color='r', scale=15,
#            width=0.003, label='Normais a Curvatura')

# Definindo direções dos feixes incidente e refletido
direcao_feixe_inc_x = x_mirror[y_mirror <= d]
direcao_feixe_inc_y = np.abs(d - y_mirror[y_mirror <= d])
direcao_feixe_ref_x = -x_mirror[y_mirror <= d]
direcao_feixe_ref_y = -y_mirror[y_mirror <= d]

# Cálculo dos ângulos de incidência e reflexão e suas plotagens
theta_inc = calcular_ang(vetor_normal_x, vetor_normal_y, direcao_feixe_inc_x, -direcao_feixe_inc_y)
plt.quiver(x_mirror[180], y_mirror[180], -direcao_feixe_inc_x[65], direcao_feixe_inc_y[65], color='b',
           scale=2, width=0.003, label='Feixe de Incidência')
# plt.quiver(-x_mirror[y_mirror <= d], y_mirror[y_mirror <= d], direcao_feixe_inc_x, direcao_feixe_inc_y, color='b',
#            scale=2, width=0.003, label='Feixe de Incidência')

theta_ref = calcular_ang(-vetor_normal_x, -vetor_normal_y, direcao_feixe_ref_x, direcao_feixe_ref_y)
plt.quiver(x_mirror[180], y_mirror[180], direcao_feixe_ref_x[65], direcao_feixe_ref_y[65], color='g',
           scale=2, width=0.003, label='Feixe de Reflexão')
# plt.quiver(-x_mirror[y_mirror <= d], y_mirror[y_mirror <= d], -direcao_feixe_ref_x, direcao_feixe_ref_y, color='g',
#            scale=2, width=0.003, label='Feixe de Reflexão')


# Calculo erro medio utilizando a lei de snell
razao = np.sin(theta_inc) / np.sin(theta_ref)
razao_med = np.mean(razao)
diferencas = np.abs(razao - 0.4920)
erro_medio = np.mean(diferencas)
porcentagem_erro_medio = (erro_medio / 0.4920) * 100
print(f"Razão média entre os senos dos angulos de incidencia e refração: {razao_med:.4f}")
print(f"Porcentagem do erro médio: {porcentagem_erro_medio:.2f}%")


# Calcule as normais em cada ponto do semicírculo
vetor_normal_x_cano, vetor_normal_y_cano = calcular_vetor_normal(x_circ, y_circ)
vetor_normal_x_cano_utilizado = vetor_normal_x_cano[480]
vetor_normal_y_cano_utilizado = vetor_normal_y_cano[480]
angulo_feixeRef_normalCano = calcular_ang(-direcao_feixe_ref_x, direcao_feixe_ref_y, -vetor_normal_x_cano[186::-1],-vetor_normal_y_cano[186::-1])
# angulo_feixeRef_normalCano = np.mean(angulo_feixeRef_normalCano == 1)
# plt.quiver(np.hstack((x_circ[0:63], x_circ[139:201])), np.hstack((y_circ[0:63], y_circ[139:201])), -vetor_normal_x_cano_utilizado,
#            -vetor_normal_y_cano_utilizado, scale=15, color='r', width=0.003, label='Normais')
plt.quiver( x_circ[480],y_circ[480], -vetor_normal_x_cano_utilizado,-vetor_normal_y_cano_utilizado, scale=15, color='r', width=0.003)

# if math.isclose(angulo_feixeRef_normalCano, 0, abs_tol=0.01):
#     text = "O feixe de refração bate com o vetor normal do cano."
# else:
#     text = "O feixe de refração não está alinhado com o vetor normal do cano."


# Plotagem
plt.plot(x_mirror[y_mirror <= d], y_mirror[y_mirror <= d], '--k', label='Espelho TL')
plt.plot(-x_mirror[y_mirror <= d], y_mirror[y_mirror <= d], '--k')
plt.plot(x_circ, y_circ, '-C0')
plt.plot([0], [0], 'or')
plt.plot([0], [d], 'sk')
angulo_exemplo = np.pi*.3
z_exemplo, r_exemplo = z_r_de_alpha(angulo_exemplo, cT, cL, d, t_TL)
x_exemplo, y_exemplo = x_y_de_alpha(angulo_exemplo, cT, cL, d, t_TL)
plt.plot([0, x_exemplo, 0], [0, y_exemplo, d], 'green', alpha=.5)
plt.text(0.03, 0.2, f'Razão média entre os \nsenos dos angulos de\nincidencia e reflexão: {razao_med:.4f}', fontsize=12, ha='left', va='center')
# plt.text(0.01, -0.005, text, fontsize=12, ha='left', va='center')
plt.gca().set_aspect("equal")
plt.legend()

#plotagem erro medio
plt.figure()
referencia = [0.4920] * len(razao)

# Plotar os valores de razao e a referência
plt.plot(razao, label='Valores de razao')
plt.plot(referencia, linestyle='--', label='Valor de referência (0.4920)')
plt.xlabel('Índice')
plt.ylabel('Valores')
plt.title('Comparação dos valores simulados e teórico')
plt.ylim(0.48, 0.5)
plt.yticks([i/1000 for i in range(480, 500, 2)])
plt.legend()
# plt.gca().set_aspect("equal")
plt.show()

