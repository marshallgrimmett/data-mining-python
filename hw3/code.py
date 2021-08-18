import numpy as np
import matplotlib.pyplot as plt

# L = np.matrix('2 -1 -1 0 0 0; -1 2 -1 0 0 0; -1 -1 3 -1 0 0; 0 0 -1 3 -1 -1; 0 0 0 -1 2 -1; 0 0 0 -1 -1 2')
# D = np.matrix('2 0 0 0 0 0; 0 2 0 0 0 0; 0 0 3 0 0 0; 0 0 0 3 0 0; 0 0 0 0 2 0; 0 0 0 0 0 2')

# D_inv_sqrt = np.reciprocal(np.sqrt(D), where=np.sqrt(D)!=0)
# Ls = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)
# Ls = np.matrix('2 -0.5 -0.3333 0 0 0; -0.5 2 -0.3333 0 0 0; -0.5 -0.5 3 -0.3333 0 0; 0 0 -0.3333 3 -0.5 -0.5; 0 0 0 -0.3333 2 -0.5; 0 0 0 -0.3333 -0.5 2')

# v, u = np.linalg.eig(L)
# L_min = sorted(np.asarray(u[:,2]).reshape(-1))
# print(u[:,2])

# v, u = np.linalg.eig(Ls)
# Ls_min = sorted(np.asarray(u[:,3]).reshape(-1))
# print(u[:,3])


# plt.plot(L_min)
# plt.title('u')
# plt.show()

# plt.plot(Ls_min)
# plt.title('u_s')
# plt.show()

A = np.matrix([
    [0,1,1,0,1,0,0],
    [1,0,1,0,0,0,0],
    [1,1,0,2,0,0,0],
    [0,0,2,0,0,1,0],
    [1,0,0,0,0,2,0],
    [0,0,0,1,2,0,0],
    [0,0,0,0,0,2,0]
    ])

D = np.matrix([
    [3,0,0,0,0,0,0],
    [0,2,0,0,0,0,0],
    [0,0,4,0,0,0,0],
    [0,0,0,3,0,0,0],
    [0,0,0,0,3,0,0],
    [0,0,0,0,0,3,0],
    [0,0,0,0,0,0,2]
    ])

L = D - A

D_inv_sqrt = np.reciprocal(np.sqrt(D), where=np.sqrt(D)!=0)
Ls = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)

# avg weight cut
v, u = np.linalg.eig(A)
# print(v)
A_max = sorted(np.asarray(u[:,1]).reshape(-1))
print(u[:,1])

# ratio cut
v, u = np.linalg.eig(L)
# print(v)
L_min = sorted(np.asarray(u[:,2]).reshape(-1))
print(u[:,2])

# normalized cut
v, u = np.linalg.eig(Ls)
# print(v)
Ls_min = sorted(np.asarray(u[:,2]).reshape(-1))
print(u[:,2])


plt.plot(A_max)
plt.title('u_a')
plt.show()

plt.plot(L_min)
plt.title('u')
plt.show()

plt.plot(Ls_min)
plt.title('u_s')
plt.show()