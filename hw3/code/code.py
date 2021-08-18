import numpy as np
import matplotlib.pyplot as plt

L = np.matrix('2 -1 -1 0 0 0; -1 2 -1 0 0 0; -1 -1 3 -1 0 0; 0 0 -1 3 -1 -1; 0 0 0 -1 2 -1; 0 0 0 -1 -1 2')
D = np.matrix('2 0 0 0 0 0; 0 2 0 0 0 0; 0 0 3 0 0 0; 0 0 0 3 0 0; 0 0 0 0 2 0; 0 0 0 0 0 2')

# D_inv_sqrt = np.reciprocal(np.sqrt(D), where=np.sqrt(D)!=0)
# L_s = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)
Ls = np.matrix('2 -0.5 -0.3333 0 0 0; -0.5 2 -0.3333 0 0 0; -0.5 -0.5 3 -0.3333 0 0; 0 0 -0.3333 3 -0.5 -0.5; 0 0 0 -0.3333 2 -0.5; 0 0 0 -0.3333 -0.5 2')

v, u = np.linalg.eig(L)
L_min = sorted(np.asarray(u[:,2]).reshape(-1))

v, u = np.linalg.eig(Ls)
Ls_min = sorted(np.asarray(u[:,3]).reshape(-1))

plt.plot(L_min)
plt.title('u')
plt.show()

plt.plot(Ls_min)
plt.title('u_s')
plt.show()