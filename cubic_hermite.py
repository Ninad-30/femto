import numpy as np
from scipy.misc import derivative

functions = [
    lambda x,y:1,
    lambda x,y:x,
    lambda x,y:y,
    lambda x,y:x*y,
    lambda x,y:x**2,
    lambda x,y:y**2,
    lambda x,y:x**2 * y,
    lambda x,y:x * y**2,
    lambda x,y:x**3,
    lambda x,y:y**3
]

coordinates = [
    [0,0],
    [0,1],
    [1,0],
    [1/3,1/3],
    [0,0],
    [0,1],
    [1,0],
    [0,0],
    [0,1],
    [1,0],
]

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

psi = [
    lambda f:f(0,0),
    lambda f:f(1,0),
    lambda f:f(1,0),
    lambda f:f(1/3,1/3),
    lambda f:partial_derivative(f,0,[0,0]),
    lambda f:partial_derivative(f,0,[0,1]),
    lambda f:partial_derivative(f,0,[1,0]),
    lambda f:partial_derivative(f,1,[0,0]),
    lambda f:partial_derivative(f,1,[0,1]),
    lambda f:partial_derivative(f,1,[1,0]),
]

matrix = []

for ps in psi:
    lis = []
    for f in functions:
        lis.append(ps(f))
    matrix.append(lis)


print(np.linalg.det(np.array(matrix)))
# final = np.linalg.inv(np.array(matrix))

# for i in range(10):
#     for j in range(10):
#         print(final[i,j], end=' ')
#     print()