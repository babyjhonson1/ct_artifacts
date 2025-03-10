import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import time

from Analitical_MU_P import  p_func_array, mu_func_array
from ForwardTask import ForwarTask
from ReverseTask import ReverseTask

# from skimage.transform import radon,iradon,resize
# from skimage.util import random_noise
# from pydicom.data import get_testdata_file
# import pydicom

start_time = time.time()


N_angels = 50 # количество углов вращения
N_rays = 50 # количество лучей в пучке (картинку будет адевать только около половины лучей в среднем за один угол)
N_xy = 100 # размер сетки в единицах имерения количества штук клеток. Оптимально - чтобы совпадало с рарешением пикселей картинки
xy_max = 100 # размер половины сетки в сантиметрах. Стоит учесть, что в сантиметрах система координат в центре, а в пиксилях - в углу
# rays_d = np.sqrt(2)*2*xy_max/N_rays # расстояние между лучами





func_name = 0 # выбор конкретной функции для mu
A = 1 #амплитуда фигуры
a = 30 #радиус фигуры
x0 = 35 #координаты центра фигуры
y0 = 35

p = p_func_array(func_name, A, a, x0, y0, N_rays, N_angels, xy_max)
mu = mu_func_array(func_name, A, a, x0, y0, N_xy, xy_max, N_rays)





# filename = get_testdata_file("CT_small.dcm")
# dsct = pydicom.dcmread(filename)

# mu = dsct.pixel_array#resize(data.shepp_logan_phantom(),(n,n))
# n = 128
# theta = np.linspace(0., 180.,n, endpoint=False)
# sigma = 1e-1
# p = radon(mu, theta=theta, circle = False)
# # mu_reconstructed = iradon(sinogram, theta=theta, circle = False)







p_reconstructed = ForwarTask(mu, N_angels, N_rays, xy_max)


print("Process finished --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()


fig, ax = plt.subplots(1,3, figsize = (20,20))

ax[0].imshow(mu)
ax[0].set_title('Ground truth mu')

ax[1].imshow(p)
ax[1].set_title('Ground truth CT raw data')

ax[2].imshow(p_reconstructed)
ax[2].set_title('CT raw data')

plt.show()









xi0 = 0.1#np.pi/rays_d
mu_reconstructed = ReverseTask(p_reconstructed, N_xy, xy_max, Method_name="__filterd_back_projection__", xi0= xi0)

print("Process finished --- %s seconds ---" % (time.time() - start_time))


fig, ax = plt.subplots(1,3, figsize = (20,20))

ax[0].imshow(mu)
ax[0].set_title('Ground truth mu')

ax[1].imshow(p_reconstructed)
ax[1].set_title('Ground truth CT raw data')

ax[2].imshow(mu_reconstructed)
ax[2].set_title('mu')

plt.show()