import numpy as np

# Задание Операции над одномерными массивами

array1 = np.linspace(start=np.pi * -4, stop=np.pi * 4, num=100)
array2 = np.square(np.cos(array1)) + np.square(np.sin(array1))
result = np.all(a=array2,where=1)
print(result)

# Задание Двумерный массив

diag1 = np.arange(start=3,stop=25,step=3)
diag2 = np.linspace(start=-1,stop=-1,num=8)

matrix1 = np.diag(diag1)
matrix2 = np.flip(np.diag(diag2),1)

result = matrix1 + matrix2
print(result)

# Задание 1

matrix1 = np.random.sample((5,5))
matrix2 = np.random.sample((5,5))

print(matrix1.ndim)
print(matrix1.shape)
print(matrix2.ndim)
print(matrix2.shape)
    
def np_mult(a,b):
    return a @ b

def mult(a, b):
    length = len(a) 
    result_matrix = [[0 for i in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length):
            for k in range(length):
                result_matrix[i][j] += a[i][k] * b[k][j]
    return result_matrix


a = matrix1.tolist()
b = matrix2.tolist()

M1 = mult(a,b)

M2 = np_mult(matrix1,matrix2)

print(np.allclose(np.array(M1), M2))

# Задание 2

a = np.random.sample((1, 3))[0]
b = np.random.sample((1, 3))[0]

def scalar_product(a, b):
    scalar_sum = 0

    for i in range(len(a)):
        scalar_sum += a[i] * b[i]

    return scalar_sum

def np_scalar_product(a, b):
    return np.dot(a,b)

M1 = scalar_product(a,b)
M2 = np_scalar_product(a,b)

print(np.allclose(M1, M2))

# Задание 3

A = np.array([
    [1, 2, 3, 4, 5],
    [3, 2, 0,-1, 1],
    [0, 0,-1,-2,-3],
    [0, 1,-1, 1,-1]
])

def cumsum(A):
    cumsum = []
    cumulative_sum = 0
    for row in A:
        for element in row:
            cumulative_sum += element
            cumsum.append(cumulative_sum)
    return cumsum

S1 = cumsum(A)
S2 = np.cumsum(A)

print(np.allclose(S1, S2))