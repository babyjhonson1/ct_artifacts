import numpy as np

import constants
from coordinate_transform_functions import *

def sinc(x):
    if x == 0:
        return 1
    return np.sin(x)/x

def h1(ksi):
    return constants.xi0**2/np.pi/2*(sinc(constants.xi0*ksi) - (sinc(constants.xi0*ksi/2))**2/2.)

def p_convolution(p):
    f = np.zeros((constants.N_rays,constants.N_angels))
    for int_ksi in range(constants.N_rays):
        for int_theta in range(constants.N_angels):
            integral_1_47 = 0
            for int_ksi0 in range(constants.N_rays):
                ksi_minus_ksi0 = int_to_ksi(int_ksi-int_ksi0)
                integral_1_47+=p[int_ksi0,int_theta]*h1(ksi_minus_ksi0)
            f[int_ksi,int_theta] = integral_1_47*constants.rays_d
    # return p
    return f

def filterd_back_projection(p, _N_xy, _xy_max, _xi0):
    constants.N_xy, constants.xy_max, constants.xi0= _N_xy, _xy_max, _xi0
    constants.N_rays = p.shape[0]
    constants.N_angels = p.shape[1]
    constants.rays_d = np.sqrt(2)*2*constants.xy_max/constants.N_rays

    mu = np.zeros((constants.N_xy, constants.N_xy))

    f = p_convolution(p)

    for int_x in range(constants.N_xy):
        for int_y in range(constants.N_xy):
            integral_1_49 = 0
            for int_theta in range(constants.N_angels):
                x = int_to_xy(int_x)
                y = int_to_xy(int_y)
                theta = int_to_theta(int_theta)
                ksi = x*np.cos(theta) + y*np.sin(theta)
                # int_ksi = ksi_to_int(ksi)
                # if -N_rays <= int_ksi < N_rays:
                integral_1_49+=f[ksi_to_int(ksi),int_theta]
            mu[int_x, int_y] = integral_1_49/constants.N_angels # вообще должно быть *dtheta/2/np.pi, но dtheta = 2*np.pi/N_angels

    return mu