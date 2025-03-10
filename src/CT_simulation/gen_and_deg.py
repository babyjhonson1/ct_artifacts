# GENERATION AND DEGRADATION
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from skimage.transform import radon, iradon
from skimage.util import random_noise

def generate(image_size=(512, 512), radius=(50, 90), num_vertices=8, mu=4000):
    image_array = np.zeros(image_size)  # Все пиксели черные 
    # Генерируем случайные вершины для выпуклого многоугольника
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)  # Углы для вершин
    radii = np.random.randint(radius[0], radius[1], num_vertices)  # Случайные радиусы
    x_center = np.random.randint(max(radii), image_size[0] - max(radii))
    y_center = np.random.randint(max(radii), image_size[1] - max(radii))
    x_vertices = radii * np.cos(angles) + x_center  # X-координаты вершин
    y_vertices = radii * np.sin(angles) + y_center  # Y-координаты вершин

    # Создаем Path для многоугольника
    vertices = np.column_stack((x_vertices, y_vertices))
    path = Path(vertices, closed=True)

    # Заполняем многоугольник 
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if path.contains_point((j, i)):  # Проверяем, находится ли точка внутри фигуры
                image_array[i, j] = mu
    return image_array

def generate2(image_size=(512, 512), radius=(50, 90), num_vertices=8, mu=4000):
    num_arrays = np.random.randint(1, 4)  # Случайное количество массивов (от 1 до 3)
    final_array = np.zeros(image_size)  # Итоговый массив
    used_pixels = np.zeros(image_size, dtype=bool)  # Массив для отслеживания занятых пикселей

    for _ in range(num_arrays):
        while True:
            # Генерируем новый массив
            vertices = np.random.randint(5, num_vertices+1)
            shift = np.random.randint(0, (radius[1] - radius[0]) // 3)
            rad = (radius[0] + shift, radius[1] - shift)
            new_array = generate(image_size=image_size, radius=rad, num_vertices=vertices, mu=mu)
            # Проверяем, пересекаются ли ненулевые пиксели с уже занятыми
            overlap = np.logical_and(new_array > 0, used_pixels)
            if not np.any(overlap):  # Если пересечений нет
                break  # Выходим из цикла
        # Добавляем ненулевые пиксели в used_pixels
        used_pixels = np.logical_or(used_pixels, new_array > 0)
        # Складываем массивы
        final_array += new_array

    return final_array

def degradation(image, theta, max_p=1e5, sigma=1e-1):
    sinogram = radon(image, theta=theta, circle = False)
    sinogram = random_noise(sinogram, mode='gaussian', var=sigma, clip=False)
    sinogram_real = np.clip(sinogram, None, max_p)
    image_real = iradon(sinogram_real, theta=theta, circle=False)
    
    return image_real


