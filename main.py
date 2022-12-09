import numpy as np
from scipy import linalg
from scipy.sparse.linalg import cg

np.random.seed(1)


a = -1  # min элемент в матрице
b = 1  # max элемент в матрице

n = 3  # размер матрицы
m = n + 2

A = a + (b - a) * np.random.rand(n, n)
print('COND:', np.linalg.cond(A))
print('Det:', np.linalg.det(A))
u = a + (b - a) * np.random.rand(n, 1)
f = np.dot(A, u)

A = np.append(A, [A[0] + A[n - 1]], axis=0)
A = np.append(A, [A[1] + A[n - 2]], axis=0)
f = np.append(f, [f[0] + f[n - 1]], axis=0)
f = np.append(f, [f[1] + f[n - 2]], axis=0)

e_A = 0.01
rand_val = -1 + 2 * np.random.rand()  # [-1, 1)
print(rand_val)
print((1 + rand_val * e_A))
# A_ = np.eye(3) * (1 + rand_val * e_A)
# A_ = A * np.random.rand()  # НУЖНО ИСПРАВИЬЬ ГЕНЕРАЦИЮ РАНДОМНОЙ МАТРИЦЫ
A_ = np.eye(3) * np.random.rand()  # НУЖНО ИСПРАВИЬЬ ГЕНЕРАЦИЮ РАНДОМНОЙ МАТРИЦЫ


print(A_)

e_h = 0.01
f_ = f * (1 + rand_val * e_h)
print(np.linalg.norm(np.dot(A_, u) - f_, ord=2))
h = np.linalg.norm(A - A_, ord=2)
print(A)
print(A_)
sigma = np.linalg.norm(f - f_, ord=2)
psi = h * np.linalg.norm(u, ord=2) + sigma


def jacobi(A, b, x_0, eps):
    b = b.flatten()
    D = np.diag(A)
    R = A - np.diagflat(D)
    x_prev = x_0
    x = x_prev
    N = 25  # число итераций
    while N >= 0:
        x = (b - np.dot(R, x_prev)) / D
        x_prev = x
        N -= 1
    return x


def gradient_descent_with_fractional_step(f, x_0, eps, M):
    k = 0
    x_prev = x_0
    x = x_prev
    t_k = 10
    while True:
        k += 1
        x = x_prev - t_k * np.squeeze(F(x_prev))

        if (f(m, n, A_, x, f_, h, sigma) - f(m, n, A_, x_prev, f_, h, sigma)) < 0:
            if np.linalg.norm(x - x_prev, ord=2) <= eps or k + 1 > M:
                return x
        else:
            t_k /= 2.
        x_prev = x


def kron(q, i):
    if q == i:
        return 1
    else:
        return 0


def L_(m, n, a, v, f, h, sigma):
    first = 0
    for i in range(m):
        buff = 0
        for j in range(n):
            buff += a[i][j]*v[j]-f[i]
        first += np.power(buff, 2)
    first = np.squeeze(np.sqrt(first))

    second = 0
    for i in range(n):
        second += np.power(v[i], 2)
    second = np.sqrt(second) * h

    return first + second + sigma


def diff1(m, n, k, h, a, v, f):
    buff = np.zeros(5, float)

    for i in range(m):
        buff[0] = 0
        for j in range(n):
            buff[0] += a[i][j] * v[j] - f[i]
        buff[1] += a[i][k] * buff[0]
        buff[2] += np.power(buff[0], 2)

    buff[2] = np.sqrt(buff[2])
    buff[3] = h * v[k]
    for j in range(n):
        buff[4] += np.power(v[j], 2)
    buff[4] = np.sqrt(buff[4])

    return buff[1] / buff[2] + buff[3] / buff[4]


def diff2(m, n, k, q, h, a, v, f):
    buff = np.zeros(8, float)

    for i in range(m):
        buff[0] += a[i][k]*a[i][q]

    for i in range(m):
        buff[1] = 0
        for j in range(n):
            buff[1] += a[i][j]*v[j]-f[i]
        buff[2] += np.power(buff[1], 2)
        buff[3] += a[i][q] * buff[1]
        buff[4] += a[i][k] * buff[1]

    buff[5] = buff[2]
    buff[2] = np.sqrt(buff[2])

    for j in range(n):
        buff[6] += np.power(v[j], 2)

    buff[7] = np.sqrt(buff[6])

    return (buff[0] * buff[2] - (buff[3]*buff[4])/buff[2]) / buff[5] + (h/buff[6]) * (kron(k, q)*buff[7] - v[q]*v[k]/buff[7])


def F(x):
    return np.array([[diff1(m, n, d, h, A_, x, f_)] for d in range(n)])


def dF(x):
    buff = []
    for i in range(n):
        buff2 = []
        for j in range(n):
            buff2.append([diff2(m, n, i, j, h, A_, x, f_)])
        buff.append(buff2)
    return np.squeeze(np.array(buff))


def rho(m, n, a, v, f, h, sigma, lambda_delta):
    first = 0

    for i in range(m):
        buff = 0
        for j in range(n):
            buff += a[i][j]*v[j]-f[i]
        first += np.power(buff, 2)
    first = np.squeeze(np.sqrt(first))
    second = 0
    for i in range(n):
        second += np.power(v[i], 2)
    second = np.sqrt(second) * h

    return first - np.power(lambda_delta + sigma + second, 2)


x_0 = np.squeeze(u)
print(gradient_descent_with_fractional_step(L_, x_0, 1e-2, 1000))

# -------- second part -------
v_grad = gradient_descent_with_fractional_step(L_, x_0, 1e-2, 1000)
print(np.linalg.norm(v_grad, ord=2))
lambda_delta = L_(m, n, A_, v_grad, f_, h, sigma)
print(lambda_delta, sigma, h)
A_T = np.transpose(A_)
alpha = 1
matrix = np.dot(A_T, A_) + alpha * np.eye(n)
vector = np.dot(A_T, f_)
print(f)
x = cg(matrix, vector)[0]
val = 1


while val > 1e-2 and alpha > 0:
    matrix = np.dot(A_T, A_) + alpha * np.eye(n)
    vector = np.dot(A_T, f_)
    answ = cg(matrix, vector)[0]
    val = rho(m, n, A_, answ, f_, h, sigma, lambda_delta)
    alpha -= 0.0001
    print(val, alpha)
