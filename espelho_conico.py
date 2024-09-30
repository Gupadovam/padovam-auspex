import autograd.numpy as np
from autograd import grad, hessian
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Definindo constantes
theta_cone = np.radians(45)  # Ângulo de 45 graus do cone
y_cone_offset = 3e-3  # Deslocamento do centro do cone em y
z_cone_offset = -3e-3  # Deslocamento do centro do cone em z
x_cone_offset = 0e-3  # Deslocamento do cone no eixo x
L = 6e-3  # Altura do cone


def cone_radius(y):
    """Função para calcular o raio do cone em função da posição y"""
    return np.abs(y) * np.tan(theta_cone)


def plot_initial_setup(ax):
    ax.plot([0], [0], [0], 'sk')
    X = np.linspace(-20e-3, 20e-3, 100)
    Y = np.linspace(-20e-3, 20e-3, 100)
    X, Y = np.meshgrid(X, Y)
    Z1 = 10e-3 * np.ones_like(X)
    Z2 = 20e-3 * np.ones_like(X)
    ax.plot_surface(X, Y, Z1, alpha=0.5, color='gray')
    ax.plot_surface(X, Y, Z2, alpha=0.5, color='gray')

    # Ajustando a posição do cone
    y_cone = np.linspace(0, L, 100) + y_cone_offset
    theta = np.linspace(0, 2 * np.pi, 100)
    theta, y_cone = np.meshgrid(theta, y_cone)

    # Calculando o raio do cone com o deslocamento no eixo y
    r_cone = cone_radius(y_cone - y_cone_offset)
    z_cone = r_cone * np.sin(theta) + z_cone_offset
    x_cone = r_cone * np.cos(theta) + x_cone_offset

    ax.plot_surface(x_cone, y_cone, z_cone, color='blue', alpha=0.5)

def plot_focus(ax, f):
    ax.plot([f[0]], [f[1]], [f[2]], 'or')


def travel_time(vars, f):
    x1, y1, x2, y2, theta, y_cone = vars
    y_interface1 = 10e-3
    y_interface2 = 20e-3

    # Cálculo correto do raio do cone sem aplicar o offset de y_cone aqui
    r_cone = cone_radius(y_cone)

    # Calculando as coordenadas do cone com o deslocamento aplicado apenas na visualização
    z_cone = r_cone * np.sin(theta) + z_cone_offset
    x_cone = r_cone * np.cos(theta) + x_cone_offset


    # Cálculo do tempo de viagem em cada meio
    t1 = np.sqrt((x_cone) ** 2 + (y_cone) ** 2 + z_cone ** 2) / c0
    t2 = np.sqrt((x1 - x_cone) ** 2 + (y1 - y_cone) ** 2 + (y_interface1 - z_cone) ** 2) / c0
    t3 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (y_interface2 - y_interface1) ** 2) / c1
    t4 = np.sqrt((f[0] - x2) ** 2 + (f[1] - y2) ** 2 + (f[2] - y_interface2) ** 2) / c2

    return (t1 + t2 + t3 + t4) ** 2


# Derivatives using autograd
grad_travel_time = grad(travel_time)
hess_travel_time = hessian(travel_time)


def optimize_travel_time(f, x1_init, x2_init, theta_init, y_cone_init, max_iter=50, reg_factor=.3e-9):
    x1, y1 = x1_init
    x2, y2 = x2_init
    theta = theta_init
    y_cone = y_cone_init
    vars = np.array([x1, y1, x2, y2, theta, y_cone])

    travel_times = []
    gradients = [[] for _ in range(6)]

    for i in range(max_iter):
        travel_times.append(travel_time(vars, f))
        grad = grad_travel_time(vars, f)
        for j in range(6):
            gradients[j].append(grad[j])
        hess = hess_travel_time(vars, f)
        hess += reg_factor * np.eye(6)
        delta = np.linalg.solve(hess, -grad)
        vars += delta

    return vars, travel_times, gradients



def plot_optimized_path(ax, vars, f):
    x1, y1, x2, y2, theta, y_cone = vars
    y_interface1 = 10e-3
    y_interface2 = 20e-3

    # Calcular raio e coordenadas do cone com o deslocamento
    r_cone_opt = cone_radius(y_cone)
    z_cone_opt = r_cone_opt * np.sin(theta) + z_cone_offset
    x_cone_opt = r_cone_opt * np.cos(theta) + x_cone_offset

    # Ajustar y_cone com o deslocamento
    y_cone_adjusted = y_cone + y_cone_offset

    # Plotando os caminhos otimizados considerando os offsets na visualização
    ax.plot([0, x_cone_opt], [0, y_cone_adjusted], [0, z_cone_opt], 'b-')  # Primeiro meio até reflexão no cone
    ax.plot([x_cone_opt, x1], [y_cone_adjusted, y1+y_cone_offset], [z_cone_opt, y_interface1], 'c-')  # Reflexão até a primeira interface
    ax.plot([x1, x2], [y1+y_cone_offset, y2], [y_interface1, y_interface2], 'y-')  # Segundo meio
    ax.plot([x2, f[0]], [y2, f[1]], [y_interface2, f[2]], 'g-')  # Terceiro meio até o foco


def plot_travel_times(travel_times):
    plt.figure()
    plt.semilogy(travel_times)
    plt.xlabel('Iterações')
    plt.ylabel('Tempo de Viagem')
    plt.title('Tempo de Viagem por Iteração')
    plt.grid()
    plt.show()


def plot_gradients(gradients):
    labels = ['grad_x1', 'grad_y1', 'grad_x2', 'grad_y2', 'grad_theta', 'grad_y_cone']
    for i, grad in enumerate(gradients):
        plt.figure()
        plt.semilogy(grad)
        plt.xlabel('Iterações')
        plt.ylabel('Magnitude do Gradiente')
        plt.title(f'Magnitude {labels[i]} por Iteração')
        plt.grid()
        plt.show()


# Configurações iniciais
c0, c1, c2 = 1485., 6320., 1485.
f = (0e-3, 6e-3, 25e-3)
x1_init = [1e-3, 1e-3]
x2_init = [1e-3, 1e-3]
theta_init = np.pi / 4  # Um valor inicial razoável para o ângulo
y_cone_init = 1e-3  # Valor inicial para o y do cone


# Otimização
vars_opt, travel_times, gradients = optimize_travel_time(f, x1_init, x2_init, theta_init, y_cone_init)
x1_opt, y1_opt, x2_opt, y2_opt, theta_opt, y_cone_opt = vars_opt
print(f'Otimizados: {vars_opt}')

# Plotando resultados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_initial_setup(ax)
plot_focus(ax, f)
plot_optimized_path(ax, vars_opt, f)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Plotar tempos de viagem e gradientes
plot_travel_times(travel_times)
# plot_gradients(gradients)
