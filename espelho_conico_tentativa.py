import autograd.numpy as np
from autograd import grad, hessian
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Definindo constantes
x_cone_offset = 3e-3  # Deslocamento do centro do cone em x
y_cyl_offset = 0e-3  # Deslocamento do centro do cone em y
z_cyl_offset = -3e-3  # Deslocamento do centro do cone em z
L = 6e-3  # Length of the cone
angle_cone = np.radians(45)  # Ângulo do cone


# Função para plotar a configuração inicial
def plot_initial_setup(ax):
    ax.plot([0], [0], [0], 'sk')
    X = np.linspace(-20e-3, 20e-3, 100)
    Y = np.linspace(-20e-3, 20e-3, 100)
    X, Y = np.meshgrid(X, Y)
    Z1 = 10e-3 * np.ones_like(X)
    Z2 = 20e-3 * np.ones_like(X)
    ax.plot_surface(X, Y, Z1, alpha=0.5, color='gray')
    ax.plot_surface(X, Y, Z2, alpha=0.5, color='gray')

    # Plotting a horizontal cone
    x_cone = np.linspace(0, L, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta, x_cone = np.meshgrid(theta, x_cone)
    R = np.abs(x_cone) * np.tan(angle_cone)  # Raio varia com x
    y_cone = R * np.cos(theta) + y_cyl_offset
    z_cone = R * np.sin(theta) + z_cyl_offset  # Posição do cone abaixo da primeira interface

    # Aplicando o deslocamento em x
    ax.plot_surface(x_cone + x_cone_offset, y_cone, z_cone, color='blue', alpha=0.5)


def plot_focus(ax, f):
    ax.plot([f[0]], [f[1]], [f[2]], 'or')


def travel_time(vars, f):
    x1, y1, x2, y2, theta, x_cone = vars
    y_interface1 = 10e-3
    y_interface2 = 20e-3
    R = np.abs(x_cone) * np.tan(angle_cone)  # Raio varia com x
    z_cone = R * np.sin(theta) + z_cyl_offset
    y_cone = R * np.cos(theta) + y_cyl_offset

    # Incluindo x_cyl_offset nas posições do cone
    x_cone = x_cone + x_cone_offset

    t1 = np.sqrt((x_cone) ** 2 + (y_cone) ** 2 + z_cone ** 2) / c0
    t2 = np.sqrt((x1 - x_cone) ** 2 + (y1 - y_cone) ** 2 + (y_interface1 - z_cone) ** 2) / c0
    t3 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (y_interface2 - y_interface1) ** 2) / c1
    t4 = np.sqrt((f[0] - x2) ** 2 + (f[1] - y2) ** 2 + (f[2] - y_interface2) ** 2) / c2

    return (t1 + t2 + t3 + t4) **2


# Derivatives using autograd
grad_travel_time = grad(travel_time)
hess_travel_time = hessian(travel_time)


def optimize_travel_time(f, x1_init, x2_init, theta_init, x_cone_init, max_iter=100, reg_factor=10e-10):
    x1, y1 = x1_init
    x2, y2 = x2_init
    theta = theta_init
    x_cone = x_cone_init
    vars = np.array([x1, y1, x2, y2, theta, x_cone])

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
    x1, y1, x2, y2, theta, x_cone = vars
    y_interface1 = 10e-3
    y_interface2 = 20e-3
    R = np.abs(x_cone) * np.tan(angle_cone)  # Raio varia com x
    z_cone_opt = R * np.sin(theta) + z_cyl_offset
    y_cone_opt = R * np.cos(theta) + y_cyl_offset

    # Incluindo x_cyl_offset
    x_cone_opt = x_cone + x_cone_offset

    ax.plot([0, x_cone_opt], [0, y_cone_opt], [0, z_cone_opt], 'b-')  # Primeiro meio até reflexão no cone
    ax.plot([x_cone_opt, x1], [y_cone_opt, y1], [z_cone_opt, y_interface1], 'c-')  # Reflexão até primeira interface
    ax.plot([x1, x2], [y1, y2], [y_interface1, y_interface2], 'y-')  # Segundo meio
    ax.plot([x2, f[0]], [y2, f[1]], [y_interface2, f[2]], 'g-')  # Terceiro meio até o foco


def plot_travel_times(travel_times):
    plt.figure()
    plt.semilogy(travel_times)
    plt.xlabel('Iterations')
    plt.ylabel('Travel Time')
    plt.title('Travel Time over Iterations')
    plt.grid()
    plt.show()


def plot_gradients(gradients):
    labels = ['grad_x1', 'grad_y1', 'grad_x2', 'grad_y2', 'grad_theta', 'grad_x_cone']
    for i, grad in enumerate(gradients):
        plt.figure()
        plt.semilogy(grad)
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Magnitude')
        plt.title(f'{labels[i]} Magnitude over Iterations')
        plt.grid()
        plt.show()


# Initial configurations
c0, c1, c2 = 1485., 6320., 1485.
f = (2e-3, 0e-3, 25e-3)
x1_init = [0.0, 0.0]
x2_init = [0.015, 0.015]
theta_init = 0.0
x_cone_init = 0.0

# Optimization
vars_opt, travel_times, gradients = optimize_travel_time(f, x1_init, x2_init, theta_init, x_cone_init)
x1_opt, y1_opt, x2_opt, y2_opt, theta_opt, x_cone_opt = vars_opt

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_initial_setup(ax)
plot_focus(ax, f)
plot_optimized_path(ax, vars_opt, f)
plt.show()

print(f"Minimum travel time: {travel_time(vars_opt, f):.6e} seconds")
print(f"Optimal intersection inteface points: ({x1_opt:.6e}, {y1_opt:.6e}, 10.000000e-03), ({x2_opt:.6e}, {y2_opt:.6e}, 20.000000e-03)")
z_cone_opt = np.abs(x_cone_opt) * np.tan(angle_cone) * np.sin(theta_opt) + z_cyl_offset
y_cone_opt = np.abs(x_cone_opt) * np.tan(angle_cone) * np.cos(theta_opt) + y_cyl_offset
print(f"Optimal reflection point on cone: ({x_cone_opt:.6e}, {y_cone_opt:.6e}, {z_cone_opt:.6e})")
print()

y_interface1 = 10e-3
y_interface2 = 20e-3

def compute_incidence_reflection_angles(vars):
    x1, y1, x2, y2, theta, x_cone = vars

    # Coordenadas do ponto de reflexão no cone
    R = np.abs(x_cone) * np.tan(angle_cone)
    z_cone = R * np.sin(theta) + z_cyl_offset
    y_cone = R * np.cos(theta) + y_cyl_offset
    x_cone = x_cone + x_cone_offset

    r_cone = np.sqrt(x_cone ** 2 + y_cone ** 2)
    normal = np.array([x_cone, y_cone, -r_cone * np.tan(angle_cone)])

    # Normalizando o vetor normal
    normal /= np.linalg.norm(normal)

    # Vetor incidente (da origem até o ponto de reflexão)
    incident_vector = np.array([x_cone, y_cone, z_cone])
    incident_vector_norm = incident_vector / np.linalg.norm(incident_vector)

    # Vetor refletido (do ponto de reflexão até a primeira interface)
    reflected_vector = np.array([x1 - x_cone, y1 - y_cone, y_interface1 - z_cone])
    reflected_vector_norm = reflected_vector / np.linalg.norm(reflected_vector)

    # Cálculo do ângulo de incidência e reflexão
    cos_incidence_angle = np.dot(incident_vector_norm, normal)
    incidence_angle = np.arccos(np.clip(cos_incidence_angle, -1.0, 1.0))

    cos_reflection_angle = np.dot(reflected_vector_norm, normal)
    reflection_angle = np.arccos(np.clip(-cos_reflection_angle, -1.0, 1.0))

    snell_cone_valid = np.isclose(
        np.sin(incidence_angle) / c0,
        np.sin(reflection_angle) / c0,
        atol=1e-6
    )

    return np.degrees(incidence_angle), np.degrees(reflection_angle), snell_cone_valid


# Cálculo dos ângulos de incidência e reflexão
incidence_angle, reflection_angle, snell_cone_valid = compute_incidence_reflection_angles(vars_opt)

# Exibindo os resultados
print(f"Ângulo de incidência no cone: {incidence_angle:.2f} graus")
print(f"Ângulo de reflexão no cone: {reflection_angle:.2f} graus")
print(f"Lei de Snell válida para o ponto no cone: {'Sim' if snell_cone_valid else 'Não'}")


# Função para calcular o ângulo de incidência e refração nas interfaces e validar a Lei de Snell
def compute_and_validate_interface_angles(vars):
    x1, y1, x2, y2, theta, x_cone = vars

    # Coordenadas da primeira e segunda interface
    y_interface1 = 10e-3
    y_interface2 = 20e-3

    # Coordenadas do ponto de reflexão no cone
    R = np.abs(x_cone) * np.tan(angle_cone)
    z_cone = R * np.sin(theta) + z_cyl_offset
    y_cone = R * np.cos(theta) + y_cyl_offset
    x_cone = x_cone + x_cone_offset

    ### Cálculo do ângulo de incidência e refração para a primeira interface ###

    # Vetor incidente na primeira interface (trajetória entre o cone e o ponto x1, y1)
    incident_vector_1 = np.array([x1 - x_cone, y1 - y_cone, y_interface1 - z_cone])
    incident_vector_1_norm = incident_vector_1 / np.linalg.norm(incident_vector_1)

    # Vetor refratado na primeira interface (trajetória entre x1, y1 e x2, y2)
    refracted_vector_1 = np.array([x2 - x1, y2 - y1, y_interface2 - y_interface1])
    refracted_vector_1_norm = refracted_vector_1 / np.linalg.norm(refracted_vector_1)

    # Vetor normal à primeira interface (normal é vertical para a interface plana)
    normal_interface_1 = np.array([0, 0, 1])

    # Ângulo de incidência na primeira interface
    cos_incidence_angle_1 = np.dot(incident_vector_1_norm, normal_interface_1)
    incidence_angle_1 = np.arccos(np.clip(cos_incidence_angle_1, -1.0, 1.0))

    # Ângulo de refração na primeira interface
    cos_refraction_angle_1 = np.dot(refracted_vector_1_norm, normal_interface_1)
    refraction_angle_1 = np.arccos(np.clip(cos_refraction_angle_1, -1.0, 1.0))

    ### Cálculo do ângulo de incidência e refração para a segunda interface ###

    # Vetor incidente na segunda interface (trajetória entre x1, y1 e x2, y2)
    incident_vector_2 = np.array([x2 - x1, y2 - y1, y_interface2 - y_interface1])
    incident_vector_2_norm = incident_vector_2 / np.linalg.norm(incident_vector_2)

    # Vetor refratado na segunda interface (trajetória entre x2, y2 e o foco f)
    refracted_vector_2 = np.array([f[0] - x2, f[1] - y2, f[2] - y_interface2])
    refracted_vector_2_norm = refracted_vector_2 / np.linalg.norm(refracted_vector_2)

    # Vetor normal à segunda interface (normal é vertical para a interface plana)
    normal_interface_2 = np.array([0, 0, 1])

    # Ângulo de incidência na segunda interface
    cos_incidence_angle_2 = np.dot(incident_vector_2_norm, normal_interface_2)
    incidence_angle_2 = np.arccos(np.clip(cos_incidence_angle_2, -1.0, 1.0))

    # Ângulo de refração na segunda interface
    cos_refraction_angle_2 = np.dot(refracted_vector_2_norm, normal_interface_2)
    refraction_angle_2 = np.arccos(np.clip(cos_refraction_angle_2, -1.0, 1.0))

    ### Validando a Lei de Snell para cada interface ###
    snell_1_valid = np.isclose(
        np.sin(incidence_angle_1) / c0,
        np.sin(refraction_angle_1) / c1,
        atol=1e-6
    )

    snell_2_valid = np.isclose(
        np.sin(incidence_angle_2) / c1,
        np.sin(refraction_angle_2) / c2,
        atol=1e-6
    )

    return (np.degrees(incidence_angle_1), np.degrees(refraction_angle_1), snell_1_valid,
            np.degrees(incidence_angle_2), np.degrees(refraction_angle_2), snell_2_valid)


# Cálculo e validação dos ângulos de incidência e refração
(incidence_angle_1, refraction_angle_1, snell_1_valid,
 incidence_angle_2, refraction_angle_2, snell_2_valid) = compute_and_validate_interface_angles(vars_opt)

# Exibindo os resultados
print(f"Ângulo de incidência na primeira interface: {incidence_angle_1:.2f} graus")
print(f"Ângulo de refração na primeira interface: {refraction_angle_1:.2f} graus")
print(f"Lei de Snell válida para a primeira interface: {'Sim' if snell_1_valid else 'Não'}")

print(f"Ângulo de incidência na segunda interface: {incidence_angle_2:.2f} graus")
print(f"Ângulo de refração na segunda interface: {refraction_angle_2:.2f} graus")
print(f"Lei de Snell válida para a segunda interface: {'Sim' if snell_2_valid else 'Não'}")


plot_travel_times(travel_times)
