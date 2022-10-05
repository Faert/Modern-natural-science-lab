#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import double, linalg as LA
import itertools

def f1(a): #a - влияние трения a*x'
    def rhs(t, X): #Значение (x', y') в точке (x, y) 
        x, y = X
        return [y, -x**4 + 5*(x**2) - 4 - a*y]#x' = ..., y' = ...
    return rhs


def eq_quiver(rhs, limits, N=25): #векторное поле в NxN равномерно распределённых точках
    xlims, ylims = limits
    xs = np.linspace(xlims[0], xlims[1], N)
    ys = np.linspace(ylims[0], ylims[1], N)
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            vfield = rhs(0.0, [x, y])
            u, v = vfield
            U[i][j] = u
            V[i][j] = v
    return xs, ys, U, V


def plotonPlane(rhs, limits): #построение векторного поля
    #plt.close()
    xlims, ylims = limits
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    xs, ys, U, V = eq_quiver(rhs, limits)
    plt.quiver(xs, ys, U, V, alpha = 0.8)

a = double(input("Введите a (0 - нет трения):"))
states_of_eq = [[-2, 0], [-1, 0], [1, 0], [2, 0]] #не зависят от a
limits_x = (-2.5,2.5)
limits_y = (-2.5,2.5)
#start_dot = (-1., 1.) #Начальная точка траектории

rhs = f1(a)


def lin_syst(a, X): 
    x, y = X
    return (np.array([[0, 1], [-4*(x**3) + 10*x, -a]]))#ввести линеализованную матрицу в общем виде

def RK_line(N, limits_x, limits_y, marker = 'c-', x_t = False): #Числено строит траекторию
    xlims, ylims = [limits_x, limits_y]
    xs = np.linspace(xlims[0], xlims[1], N)
    ys = np.linspace(ylims[0], ylims[1], N)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            start_dot = (x, y)

            sol1 = solve_ivp(rhs, [0., 100.], start_dot, method = 'RK45', rtol=1e-12)#t->+inf
            x1, y1 = sol1.y
            plt.plot(x1, y1, marker)
            sol2 = solve_ivp(rhs, [0., -100.], start_dot, method = 'RK45', rtol=1e-12)#t->-inf
            x2, y2 = sol2.y
            plt.plot(x2, y2, marker)
            
            if(x_t and ((i == 0 and j == 1) or N == 1)): #графики x от t, убрать если не нужно
                x_t1 = x1
                t = np.linspace(0, 100, len(x_t1))
                plt.figure(2)
                plt.plot(t, x_t1, marker)
                plt.figure(1)


plotonPlane(rhs, [limits_x, limits_y])

#убрать, если без графиков x от t
x_t_ = input("Введите 1, если необходимы графики x от t (0, если нет):")
if(x_t_ == '1'):
    x_t_ = True
    plt.figure(2)
    plt.ylim(limits_x[0], limits_x[1])
    plt.figure(1)
else:
    x_t_ = False

for i, X in enumerate(states_of_eq):
    lin = lin_syst(a, X)
    w, v = LA.eig(lin)
    if(w[0].imag == 0 and w[1].imag == 0 and (w[0]*w[1]) < 0):
        print(X, " - седло\n")

        for i, wi in enumerate(w):
            x_sep = np.linspace(X[0]-0.005, X[0]+0.005)
            y_sep = -wi*(x_sep-X[0])
            RK_line(2, (x_sep[0], x_sep[49]), (y_sep[0], y_sep[49]), 'y-', x_t_)
            plt.plot(x_sep, y_sep, 'y--')

        plt.plot(X[0], X[1], 'rx')
    elif(w[0].imag == 0 and w[1].imag == 0 and (w[0]*w[1]) > 0):
        if(w[0].real > 0 or w[1].real > 0):
            print(X, " - неустойчивый узел\n")

            RK_line(2, (X[0]-0.2, X[0]+0.2), (X[1]-0.2, X[1]+0.2), 'r-', x_t_)

            plt.plot(X[0], X[1], 'rx')
        else:
            print(X, " - устойчивый узел\n")

            RK_line(2, (X[0]-0.2, X[0]+0.2), (X[1]-0.2, X[1]+0.2), 'b-', x_t_)

            plt.plot(X[0], X[1], 'bo')
        
    elif(w[0].real == 0 and w[1].real == 0):
        print(X, " - центр\n")

        RK_line(2, (X[0]-0.2, X[0]+0.2), (X[1]-0.2, X[1]+0.2), 'g-', x_t_)

        plt.plot(X[0], X[1], 'bo')
    else:
        if(w[0].real > 0 or w[1].real > 0):
            print(X, " - неустойчивый фокус\n")

            RK_line(2, (X[0]-0.2, X[0]+0.2), (X[1]-0.2, X[1]+0.2), 'r-', x_t_)

            plt.plot(X[0], X[1], 'rx')
        else:
            print(X, " - устойчивый фокус\n")

            RK_line(2, (X[0]-0.2, X[0]+0.2), (X[1]-0.2, X[1]+0.2), 'b-', x_t_)

            plt.plot(X[0], X[1], 'bo')

#RK_line(1, (x, x), (y, y)) можно построить допольнительные траектории проходящии через точку (x, y)
print("Некоторые траектории проходящии вблизи состояний равновесия:")

RK_line(1, (-2.2, -2.2), (0, 0), 'r-', x_t_)#Пример уходящих в бесконечность
if(a <= 0):
    RK_line(1, (-2, -2), (1, 1), 'r-', x_t_)#
    RK_line(1, (1, 1), (1, 1), 'r-', x_t_)#
else:
    RK_line(1, (-2, -2), (-1, -1), 'r-', x_t_)#
    RK_line(1, (1, 1), (1, 1), 'b-', x_t_)#


if(x_t_):
    plt.show(block=False)
    input("Нажмите любую кнопку, чтобы закрыть графики:")
    plt.close('all')
else:
    plt.show()