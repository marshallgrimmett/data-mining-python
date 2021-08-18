import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

###############################################################################
# A
D = pd.read_csv('hw1Data.csv', header=None).values
D = D - np.mean(D, axis=0)

###############################################################################
# B
def cov1(D):
    return np.dot(D.T, D)/len(D)
print('cov1')
start = time.time()
cov = cov1(D)
end = time.time()
print('Time: ' + str(end - start))
print(cov)
print()

def cov2(D):
    cov2 = np.zeros((6,6))
    for row in D:
        cov2 = cov2 + np.outer(row, row.T)
    cov2 = cov2/len(D)
    return cov2
print('cov2')
start = time.time()
cov = cov2(D)
end = time.time()
print('Time: ' + str(end - start))
print(cov)
print()

def cov3(D):
    cov3 = np.zeros((6,6))
    for i, row in enumerate(cov3):
        for j in range(len(row)):
            temp = 0
            for k in range(len(D)):
                temp += (D[k,i]-np.mean(D[:,i]))*(D[k,j]-np.mean(D[:,j]))
            cov3[i,j] = temp/len(D)
    return cov3
print('cov3')
start = time.time()
cov = cov3(D)
end = time.time()
print('Time: ' + str(end - start))
print(cov)
print()

###############################################################################
# C
eig = np.linalg.eigh(cov1(D))
print('Eigenvalues:')
print(eig[0])
print()
print('Eigenvectors:')
print(eig[1])
print()

###############################################################################
# D
def find_r(eigvals, a):
    eigsum = np.sum(eigvals)
    for i in range(len(eigvals)):
        temp = np.sum(eigvals[-(i+1):])
        if temp/eigsum >= a:
            return i+1
print("New Dimension, r: " + str(find_r(eig[0], 0.9)))

###############################################################################
# E
Ur = np.zeros((6,2))
Ur[:,0] = eig[1][:, 5]
Ur[:,1] = eig[1][:, 4]
np.savetxt("Components.txt", Ur.T, delimiter=",")
plt.plot(Ur[:,0], label = "PC 1")
plt.plot(Ur[:,1], label = "PC 2")
plt.xlabel('Dimensions')
plt.ylabel('Magnitude')
plt.title('First 2 Principal Components')
plt.legend()
plt.show()

###############################################################################
# F
A = np.zeros((len(D), 2))
for i in range(len(D)):
    A[i] = np.dot(Ur.T, np.transpose([D[i]])).T
plt.scatter(A[:,0], A[:,1])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Reduced Dimension Data Matrix')
plt.show()

retainedVar = eig[0][-2:]
retainedVar = retainedVar[0] + retainedVar[1]
print('Retained Variance: ' + str(retainedVar))
