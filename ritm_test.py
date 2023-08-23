import numpy as np
import matplotlib.pyplot as plt


def M():
    '''
        Возвращает матрицу М: M du/dt = f(u)
    '''
    M = np.zeros((5, 5))
    for i in range(0, 4):
        M[i, i] = 1
    return M


def f(u, g, l, m):
    '''
        Правая часть СДАУ
    '''
    x = u[0]
    y = u[1]
    v_x = u[2]
    v_y = u[3]
    T = u[4]
    res = np.zeros(5)
    res[0] = v_x
    res[1] = v_y
    res[2] = -x*T/(m*l)
    res[3] = -y*T/(m*l) - g
    res[4] = x**2 + y**2-l**2
    return res


def df(u, g, m, l):
    '''
        Возвращает матрицу Якоби Fu[i,j] = df[i]/du[j]
    '''
    x = u[0]
    y = u[1]
    v_x = u[2]
    v_y = u[3]
    T = u[4]
    res = np.zeros((5, 5))
    res[0, 2] = 1.
    res[1, 3] = 1.
    res[2, 0] = -T/(m*l)
    res[2, 4] = -x/(m*l)
    res[3, 1] = -T/(m*l)
    res[3, 4] = -y/(m*l)
    res[4, 0] = 2*x
    res[4, 1] = 2*y
    return res


def rosenbrok_solver(f, df, u_0, alpha=1, N_MAX=500):
    '''
        Решение уравнения вида M du/dt = f(u) схемой Розенброка
        f - функция правых частей
        df - функция, возвращающая матрицу Якоби функции f
        u_0 - массив с начальными условиями 
        alpha - коэффициент, определяющий метод
        N_MAX - количество разбиений на сетке
    '''
    t = np.linspace(
        t0, T, N_MAX + 1)  # массив для хранения значений t в узлах сетки
    # массив для хранения численного значения координат u
    u = np.zeros((N_MAX + 1, 5))
    u[0, :] = u_0  # строка с начальными условиями

    h = t[1] - t[0]  # шаг

    # Реализация алгоримта
    for i in range(1, N_MAX + 1):
        g = g0 + 0.05 * np.sin(2*np.pi*t[i-1])
        w = np.linalg.solve(
            M() - alpha * h * df(u[i-1], g, m, l), f(u[i-1], g, l, m))
        u[i] = u[i-1] + h * w.real
    return t, u


# Параметры задачи
t0 = 0
T = 100
x_0 = 3.
y_0 = -4.
v_x_0 = 0.
v_y_0 = 0.
l = 5.0
m = 1.0
g0 = 9.81

# Определим T0
T_0 = 2*y_0*m*l/x_0**2
u_0 = np.array([x_0, y_0, v_x_0, v_y_0, T_0])

# Вызов решателя с alpha = 1
N_MAX = 500
alpha = 1
t1, u1 = rosenbrok_solver(f, df, u_0, alpha, N_MAX)

# Вызов решателя с alpha = (1 + i)/2
N_MAX = 500
alpha = (1 + 1j)/2
t2, u2 = rosenbrok_solver(f, df, u_0, alpha, N_MAX)

# Формирование sqrt(x(t)^2 + y(t)^2)
res1 = np.zeros(N_MAX + 1)
res2 = np.zeros(N_MAX + 1)
for i in range(N_MAX + 1):
    res1[i] = np.sqrt(u1[i][0]**2 + u1[i][1]**2)
    res2[i] = np.sqrt(u2[i][0]**2 + u2[i][1]**2)

plt.plot(t1, res1)
plt.plot(t2, res2)
plt.legend(('alpha = 1', 'alpha = (1 + i)/2'))
plt.show()
