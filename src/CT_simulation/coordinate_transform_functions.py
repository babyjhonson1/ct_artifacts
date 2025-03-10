import numpy as np
import constants

# Набор функций для перевода значений переменных из реальных значений в индексы и обратно
def theta_to_int(theta):
    return int(theta/2/np.pi*constants.N_angels)

def ksi_to_int(ksi):
    return int((ksi + np.sqrt(2)*constants.xy_max)/constants.rays_d)

def xy_to_int(xy):
    return int((xy + constants.xy_max)/2/constants.xy_max*constants.N_xy)

def int_to_theta(i):
    return i*2.*np.pi/constants.N_angels

def int_to_ksi(i):
    return i*constants.rays_d - np.sqrt(2)*constants.xy_max

def int_to_xy(i):
    return 2*constants.xy_max/constants.N_xy*i - constants.xy_max
