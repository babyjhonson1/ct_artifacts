import numpy as np
import pandas as pd

from coordinate_transform_functions import *
import constants



# Функция по заданному лучу возвращает точки пересечения луча с сеткой
def ray_tracing(theta, x, y):
    q = np.zeros(3)           # используется только внутри цикла, не трогать
    qmin = np.zeros(3) - 1.  # наименьшие координаты в сетке
    qmax = np.zeros(3) + constants.N_xy + 1 # наибольшие координаты в сетке
    dq = np.ones(3)           # шаг сетки
    rho = np.zeros(3)

    q0 = np.array([xy_to_int(x), xy_to_int(y), 0.]) # координата начала луча
    pq = np.array([-np.sin(theta), np.cos(theta), 0]) # направление движения луча
    pq = pq + 1e-9
    pq = pq / np.sqrt((pq**2).sum())

    # initial conditions
    q = q0
    b = (np.sign(pq) + 1)
    points = []
    ppqq = pq*1e-3
    while (qmin < q).all() and (q < qmax).all():
        qq = q + 1e-9*np.sign(q)
        iq = ((qq - qmin) / dq).astype(np.int32)
        a = iq*dq - (qq-qmin)
        rho = a + b*dq/2 + (1 - np.sign(a)**2)
        dt = np.abs(rho / pq).min()
        # rho = np.sqrt(np.abs(a*(a-b*dq) + b*dq2/2)).min()
        points.append(q.copy())
        q += (pq+ppqq)*dt
    _ = pd.DataFrame(points, columns=['x','y', 'z'])
    return np.array([_['x'].values, _['y'].values])





# Функция определяет координату входа луча в изображение
def ray_origin_coordinates(theta, ksi):
    # по сути нам нужно найти пересечение линии ksi = x*cos(theta) + y*sin(theta) и границы снимка,
    # чтобы начать отслеживать луч именно от нее и избежать лишних вычислений

    x_line = constants.xy_max*np.sign(np.sin(theta)) 
    y_line = -constants.xy_max*np.sign(np.cos(theta))
    if not x_line: x_line = -constants.xy_max
    # Эта запись позволяет определить, с какими из границ снимка нужно искать пересечение луча
    # это зависит исключительно от угла падения луча
    # в случае theta = 0, pi/2 и т.д. будет деление на бесконечность. Код отрабатывает корректно, но лучше что-то с этим сделать

    y = y_line
    x = (ksi-y*np.sin(theta))/np.cos(theta)
    if -constants.xy_max <= x <= constants.xy_max: # ищем пересечение с одной границей, если пересечение вне области определения, то проверяем следующую границу
        return x, y
    
    x = x_line
    y = (ksi-x*np.cos(theta))/np.sin(theta)
    if -constants.xy_max <= y <= constants.xy_max: #Если пересечения нет и здесь, тогда возвращаем nan, чтобы показать, что луч не задевает изображение и можно не производить лишних вычислений
        return x, y
    return np.nan, np.nan


# Функция считает итеграл 1.7 по заданным точкам пересечения луча с сеткой
def ray_projection(points, mu):
    x = points[0,:]
    y = points[1,:]
    d_dzeta = np.sqrt(np.power(x[1:]-x[:-1], 2) + np.power(y[1:]-y[:-1], 2)) # длина отрезка с постоянным mu

    # определение точек, в которых следует взять значение mu для интеграла
    # По сути это координаты клетки в которую попадает отрезок постоянного mu
    # не обязательно использовать среднее арифметическое, просто это очень удобно
    ix = ((x[1:]+x[:-1])/2).astype(np.int32)
    iy = ((y[1:]+y[:-1])/2).astype(np.int32)
    
    correct_index = (ix<constants.N_xy)&(iy<constants.N_xy)# это костыль для работы рей трейсинга, который выдает точки пересечения вне снимка на +-1
    ix = ix[correct_index] # значения с слишком большими индексами вырезаются
    iy = iy[correct_index]
    d_dzeta = d_dzeta[correct_index]

    return (mu[ix,iy]*d_dzeta).sum()# итеграл 1.7 


def ForwarTask(mu, _N_angels, _N_rays, _xy_max):
    constants.N_angels, constants.N_rays, constants.xy_max = _N_angels, _N_rays, _xy_max
    constants.N_xy = mu.shape[0]
    constants.rays_d = np.sqrt(2)*2*constants.xy_max/constants.N_rays

    p = np.zeros((constants.N_rays,constants.N_angels))
    # цикл, перебирающий лучи со всеми theta и ksi
    for int_theta in range(constants.N_angels):
        for int_ksi in range(constants.N_rays):
            x, y = ray_origin_coordinates(int_to_theta(int_theta), int_to_ksi(int_ksi)) # Определение координаты пересечения луча и снимка
            if x != x: #проверка на наличие пересечения
              pass
            else:
                net_points = ray_tracing(int_to_theta(int_theta), x, y) # Определение точек пересечения луча с сеткой mu 
                p[int_ksi,int_theta] = ray_projection(net_points, mu) # Расчет проекции луча
    return p
