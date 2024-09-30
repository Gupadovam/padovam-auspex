import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.figure()
plt.plot([0],[0], 'sk')
plt.plot([-20e-3, 20e-3], [10e-3, 10e-3], 'k')
c0 = 1485.
c1 = 6000
alpha = 0#.7*np.pi/13
plt.axis('equal')

# Focus
f = (10e-3, 15e-3)
plt.plot([f[0]], [f[1]], 'or')


# Distance from point to line
def sqdist(alpha, f):
  sin_beta = c1 * np.sin(alpha) / c0
  beta = np.arcsin(sin_beta)
  r0 = 10e-3 / np.cos(alpha)
  r1 = 2 * r0
  e = (r0 * np.sin(alpha), r0 * np.cos(alpha))
  final = (e[0] + r1 * np.sin(beta), e[1] + r1 * np.cos(beta))
  (x1, y1) = e
  (x2, y2) = final
  (x0, y0) = f
  d = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
  return d**2

def deriv_alpha(alpha):
  eps = 1e-4
  return sqdist(alpha + eps, f) - sqdist(alpha - eps, f) / (2 * eps)

def deriv2_alpha(alpha):
  eps = 1e-4
  return deriv_alpha(alpha + eps) - deriv_alpha(alpha - eps) / (2 * eps)

N = 1100
alphas = np.zeros(N)
dists = np.zeros(N)
for i in range(N):
  #print('iteration ' + str(i))
  #print('alpha: ' + str(alpha))
  #print('sqdist: ' + str(sqdist(alpha, f)))
  alpha = alpha - deriv_alpha(alpha) / deriv2_alpha(alpha)
  alphas[i] = alpha
  dists[i] = sqdist(alpha, f)


# First medium
r0 = 10e-3 / np.cos(alpha)
e = (r0 * np.sin(alpha), r0 * np.cos(alpha))
plt.plot([0, e[0]], [0, e[1]])

# Snell's Law
sin_beta = c1 * np.sin(alpha) / c0
beta = np.arcsin(sin_beta)
r1 = 2 * r0
final = (e[0] + r1 * np.sin(beta), e[1] + r1 * np.cos(beta))
plt.plot([e[0], final[0]], [e[1], final[1]])
print(alpha)

plt.figure()
plt.plot(dists)
plt.grid()