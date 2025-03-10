import numpy as np

from coordinate_transform_functions import *
import constants


# Функция возращает значения mu(x,y) выбранной функции mu, для которой известно аналитическое решение p(ksi,theta)
def mu_func(x,y):
    if constants.func_name == 0: #кольцо
        r2 = (x- constants.x0)**2+(y-constants.y0)**2
        return constants.A*r2*np.exp(-r2/constants.a**2)

    if constants.func_name == 2: # точка
        return constants.A*np.exp(-((x-constants.x0)**2 + (y-constants.y0)**2)/constants.a**2)

    if constants.func_name == 3: # диск
        ro = np.sqrt((x - constants.x0)**2 + (y - constants.y0)**2)
        if constants.a - ro >=0.:
            return constants.A
        else:
            return 0.
        
    if constants.func_name == 4: # вогнутый парабаллоид с четкими границами
        ro = np.sqrt((x - constants.x0)**2 + (y - constants.y0)**2)
        if constants.a - ro >=0.:
            return constants.A*ro**2
        else:
            return 0.

# Функция формирует массив mu заданного размера по известной аналитической функции
def mu_func_array(_func_name, A, a, x0, y0, _N_xy, _xy_max, _N_rays):
    constants.func_name, constants.A, constants.a, constants.x0, constants.y0, constants.N_xy, constants.xy_max = _func_name, A, a, x0, y0, _N_xy, _xy_max
    constants.rays_d = np.sqrt(2)*2*_xy_max/_N_rays
    mu = np.zeros((constants.N_xy,constants.N_xy))
    for int_x in range(constants.N_xy):
        for int_y in range(constants.N_xy):
            mu[int_x, int_y] = mu_func(int_to_xy(int_x),int_to_xy(int_y))

    return mu

# Функция возращает значения p(ksi,theta) выбранной функции p, для которой известно аналитическое решение mu(x,y)
def p_func(ksi, theta):
    ksi_ = ksi - constants.x0*np.cos(theta) - constants.y0*np.sin(theta)
    if constants.func_name == 0:
        if ksi_ == 0:
            return 0.8862269254527578
        return np.sqrt(np.pi)*constants.A*constants.a*ksi_**2*np.exp(-ksi_**2/constants.a**2)*(constants.a**2/2/ksi_**2+1)
    if constants.func_name == 2:
        return np.sqrt(np.pi)*constants.A*constants.a*np.exp(-ksi_**2/constants.a**2)

    if constants.func_name == 3:
        if constants.a >= np.abs(ksi_):
            return 2*constants.A*np.abs(ksi_)*np.sqrt((constants.a/ksi_)**2-1)
        else:
            return 0.
        
    if constants.func_name == 4: # эта функция пока не решена аналитически
        if constants.a >= np.abs(ksi_):
            return 2*constants.A*np.abs(ksi_)*np.sqrt((constants.a/ksi_)**2-1)
        else:
            return 0.


# Функция формирует массив p заданного размера по известной аналитической функции
def p_func_array(_func_name, A, a, x0, y0, _N_rays, _N_angels, _xy_max):
    constants.func_name, constants.A, constants.a, constants.x0, constants.y0, constants.N_rays, constants.N_angels, constants.xy_max = _func_name, A, a, x0, y0, _N_rays, _N_angels, _xy_max
    constants.rays_d = np.sqrt(2)*2*constants.xy_max/constants.N_rays
    p = np.zeros((constants.N_rays,constants.N_angels))
    for int_theta in range(constants.N_angels):
        for int_ksi in range(constants.N_rays):
            p[int_ksi, int_theta] = p_func(int_to_ksi(int_ksi),int_to_theta(int_theta))

    return p
