import autograd.numpy as np
from autograd import grad, hessian
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Definindo constantes
R = 2e-3  # Raio do cilindro
y_cyl_offset = 10e-3  # Deslocamento do centro do cilindro em y
z_cyl_offset = 0e-3  # Deslocamento do centro do cilindro em z


def plot_initial_setup(ax):
    ax.plot([0], [0], [0], 'sk')
    X = np.linspace(-20e-3, 20e-3, 100)
    Y = np.linspace(-20e-3, 20e-3, 100)
    X, Y = np.meshgrid(X, Y)
    Z1 = 10e-3 * np.ones_like(X)
    Z2 = 20e-3 * np.ones_like(X)
    ax.plot_surface(X, Y, Z1, alpha=0.5, color='gray')
    ax.plot_surface(X, Y, Z2, alpha=0.5, color='gray')

    # Plotting a horizontal cylinder
    L = 80e-3  # Length of the cylinder
    x_cylinder = np.linspace(-L / 2, L / 2, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta, x_cylinder = np.meshgrid(theta, x_cylinder)
    y_cylinder = R * np.cos(theta) + y_cyl_offset
    z_cylinder = R * np.sin(theta) + z_cyl_offset  # Position the cylinder below the first interface

    ax.plot_surface(x_cylinder, y_cylinder, z_cylinder, color='blue', alpha=0.5)


def plot_focus(ax, f):
    ax.plot([f[0]], [f[1]], [f[2]], 'or')


def travel_time(vars, f):
    x1, y1, x2, y2, theta, x_cyl = vars
    y_interface1 = 10e-3
    y_interface2 = 20e-3
    z_cyl = R * np.sin(theta) + z_cyl_offset
    y_cyl = R * np.cos(theta) + y_cyl_offset

    t1 = np.sqrt((x_cyl) ** 2 + (y_cyl) ** 2 + z_cyl ** 2) / c0
    t2 = np.sqrt((x1 - x_cyl) ** 2 + (y1 - y_cyl) ** 2 + (y_interface1 - z_cyl) ** 2) / c0
    t3 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (y_interface2 - y_interface1) ** 2) / c1
    t4 = np.sqrt((f[0] - x2) ** 2 + (f[1] - y2) ** 2 + (f[2] - y_interface2) ** 2) / c2

    return (t1 + t2 + t3 + t4) ** 2


# Derivatives using autograd
grad_travel_time = grad(travel_time)
hess_travel_time = hessian(travel_time)


def optimize_travel_time(f, x1_init, x2_init, theta_init, x_cyl_init, max_iter=50, reg_factor=.3e-9):
    x1, y1 = x1_init
    x2, y2 = x2_init
    theta = theta_init
    x_cyl = x_cyl_init
    vars = np.array([x1, y1, x2, y2, theta, x_cyl])

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
    x1, y1, x2, y2, theta, x_cyl = vars
    y_interface1 = 10e-3
    y_interface2 = 20e-3
    z_cyl_opt = R * np.sin(theta) + z_cyl_offset
    y_cyl_opt = R * np.cos(theta) + y_cyl_offset

    ax.plot([0, x_cyl], [0, y_cyl_opt], [0, z_cyl_opt], 'b-')  # First medium to cylinder reflection
    ax.plot([x_cyl, x1], [y_cyl_opt, y1], [z_cyl_opt, y_interface1],
            'c-')  # Reflection to first interface
    ax.plot([x1, x2], [y1, y2], [y_interface1, y_interface2], 'y-')  # Second medium
    ax.plot([x2, f[0]], [y2, f[1]], [y_interface2, f[2]], 'g-')  # Third medium to focus


def plot_travel_times(travel_times):
    plt.figure()
    plt.semilogy(travel_times)
    plt.xlabel('Iterations')
    plt.ylabel('Travel Time')
    plt.title('Travel Time over Iterations')
    plt.grid()
    plt.show()


def plot_gradients(gradients):
    labels = ['grad_x1', 'grad_y1', 'grad_x2', 'grad_y2', 'grad_theta', 'grad_x_cyl']
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
f = (10e-3, -5e-3, 25e-3)
x1_init = [0.015, 0.015]
x2_init = [0.015, 0.015]
theta_init = 0.0
x_cyl_init = 0.0

# Optimization
vars_opt, travel_times, gradients = optimize_travel_time(f, x1_init, x2_init, theta_init, x_cyl_init)
x1_opt, y1_opt, x2_opt, y2_opt, theta_opt, x_cyl_opt = vars_opt

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_initial_setup(ax)
plot_focus(ax, f)
plot_optimized_path(ax, vars_opt, f)
plt.show()

# Display results
print(f"Minimum travel time: {travel_time(vars_opt, f):.6e} seconds")
print(
    f"Optimal intersection points: ({x1_opt:.6e}, {y1_opt:.6e}, 10.000000e-03), ({x2_opt:.6e}, {y2_opt:.6e}, 20.000000e-03)")
z_cyl_opt = R * np.sin(theta_opt) + z_cyl_offset
y_cyl_opt = R * np.cos(theta_opt) + y_cyl_offset
print(f"Optimal reflection point on cylinder: ({x_cyl_opt:.6e}, {y_cyl_opt:.6e}, {z_cyl_opt:.6e})")

# Normal of Cylinder
normal_cylinder = np.array([0, y_cyl_opt - y_cyl_offset, z_cyl_opt - z_cyl_offset]) / R

# Incident vector in cylinder
incident_vector = -np.array([x_cyl_opt, y_cyl_opt, z_cyl_opt])
incident_angle = np.arccos(np.dot(incident_vector, normal_cylinder) / np.linalg.norm(incident_vector))

y_interface1 = 10e-3
# Reflected vector in cylinder
reflected_vector = np.array([x1_opt - x_cyl_opt, y1_opt - y_cyl_opt, y_interface1 - z_cyl_opt])
reflected_angle = np.arccos(np.dot(reflected_vector, normal_cylinder) / np.linalg.norm(reflected_vector))

print()
print("Cylinder:")
print(f"Incident Cylinder angle: {np.degrees(incident_angle):.2f} ")
print(f"Reflected Cylinder angle: {np.degrees(reflected_angle):.2f} ")
if abs(incident_angle-reflected_angle) < 1e-3:
    print("Law of Snell respected")
else:
    print("Law of Snell not respected")


# Normal na interface 1
normal_interface1 = np.array([0, 1, 0])

# Vetor incidente na interface 1
incident_vector_interface1 = np.array([x1_opt - x_cyl_opt, y1_opt - y_cyl_opt, y_interface1 - z_cyl_opt])
incident_angle_interface1 = np.pi/2 - np.arccos(np.dot(incident_vector_interface1, normal_interface1) / np.linalg.norm(incident_vector_interface1))

# Vetor refletido na interface 1
y_interface2 = 20e-3
reflected_vector_interface1 = np.array([x2_opt - x1_opt, y2_opt - y1_opt, y_interface2 - y_interface1])
reflected_angle_interface1 = np.pi/2 - np.arccos(np.dot(reflected_vector_interface1, normal_interface1) / np.linalg.norm(reflected_vector_interface1))

r1 = c0/c1
r2 = np.sin(incident_angle_interface1)/np.sin(reflected_angle_interface1)
print()
print("Interface1:")
print(f"Incident Interface 1 angle: {np.degrees(incident_angle_interface1):.2f} ")
print(f"Refrected Interface 1 angle: {np.degrees(reflected_angle_interface1):.2f} ")
print(f"Law of Snell: Ration velocity {r1} and Ration sin {r2} - ", end='')
if abs(r1-r2) < 0.1:
    print("Law of Snell respected")
else:
    print("Law of Snell not respected")


# Normal na interface 2
normal_interface2 = np.array([0, 1, 0])

# Vetor incidente na interface 2
incident_vector_interface2 = np.array([x2_opt - x1_opt, y2_opt - y1_opt, y_interface2 - y_interface1])
incident_angle_interface2 = np.pi/2 - np.arccos(np.dot(incident_vector_interface2, normal_interface2) / np.linalg.norm(incident_vector_interface2))

# Vetor refletido na interface 2 (direção do foco)
reflected_vector_interface2 = np.array([f[0] - x2_opt, f[1] - y2_opt, f[2] - y_interface2])
reflected_angle_interface2 = np.pi/2 - np.arccos(np.dot(reflected_vector_interface2, normal_interface2) / np.linalg.norm(reflected_vector_interface2))

r2 = c1/c2
r3 = np.sin(incident_angle_interface2)/np.sin(reflected_angle_interface2)
print()
print("Interface2:")
print(f"Incident Interface 2 angle: {np.degrees(incident_angle_interface2):.2f} ")
print(f"Refrected Interface 2 angle: {np.degrees(reflected_angle_interface2):.2f} ")
print(f"Law of Snell: Ratio velocity {r2} and Ratio sin {r3} - ", end='')
if abs(r2-r3) < 0.1:
    print("Law of Snell respected")
else:
    print("Law of Snell not respected")



plot_travel_times(travel_times)
