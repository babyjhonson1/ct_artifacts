# GENERATION AND DEGRADATION
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from skimage.transform import radon, iradon
from skimage.util import random_noise
import torch

def generate(image_size=(512, 512), radius=(50, 90), num_vertices=8, mu=4000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_array = torch.zeros(image_size, dtype=torch.float32, device=device)
    
    angles = torch.linspace(0, 2 * torch.pi, num_vertices, device=device)
    radii = torch.randint(radius[0], radius[1], (num_vertices,), device=device)
    x_center = torch.randint(max(radii), image_size[0] - max(radii), (1,), device=device)
    y_center = torch.randint(max(radii), image_size[1] - max(radii), (1,), device=device)
    x_vertices = radii * torch.cos(angles) + x_center
    y_vertices = radii * torch.sin(angles) + y_center
    
    y, x = torch.meshgrid(torch.arange(image_size[0], device=device),
                         torch.arange(image_size[1], device=device),
                         indexing='ij')
    px = x.reshape(-1)
    py = y.reshape(-1)
    
    cross_products = []
    for i in range(num_vertices):
        a_x = x_vertices[i]
        a_y = y_vertices[i]
        b_x = x_vertices[(i+1) % num_vertices]
        b_y = y_vertices[(i+1) % num_vertices]
        cp = (b_x - a_x) * (py - a_y) - (b_y - a_y) * (px - a_x)
        cross_products.append(cp)
    cross_products = torch.stack(cross_products, dim=0)
    min_cp = cross_products.min(dim=0).values
    mask_flat = min_cp >= 0
    mask = mask_flat.view(image_size[0], image_size[1])
    
    image_array[mask] = mu
    return image_array

def generate2(image_size=(512, 512), radius=(50, 90), num_vertices=8, mu=4000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_arrays = torch.randint(1, 4, (1,), device=device).item()
    final_array = torch.zeros(image_size, dtype=torch.float32, device=device)
    used_pixels = torch.zeros(image_size, dtype=torch.bool, device=device)

    for _ in range(num_arrays):
        while True:
            vertices = torch.randint(5, num_vertices+1, (1,), device=device).item()
            shift = torch.randint(0, (radius[1] - radius[0]) // 3, (1,), device=device).item()
            rad = (radius[0] + shift, radius[1] - shift)
            new_array = generate(image_size=image_size, radius=rad, num_vertices=vertices, mu=mu)
            overlap = torch.logical_and(new_array > 0, used_pixels)
            if not torch.any(overlap):
                break
        used_pixels = torch.logical_or(used_pixels, new_array > 0)
        final_array += new_array

    return final_array

def degradation(tensor, theta=np.linspace(0., 180., 128, endpoint=False), max_p=1e5, sigma=1e-1):
    # Получаем размеры батча
    batch_size, _, height, width = tensor.shape
    device = tensor.device  # Сохраняем устройство (CPU или GPU) для возврата результата
    
    # Преобразуем тензор в NumPy массив [batch_size, H, W], убирая канал
    tensor_np = tensor.cpu().numpy()[:, 0, :, :]
    
    # Список для хранения обработанных изображений
    degraded_images = []
    
    # Обрабатываем каждое изображение в батче
    for i in range(batch_size):
        image = tensor_np[i]
        
        sinogram = radon(image, theta=theta, circle=False)
        sinogram_noisy = random_noise(sinogram, mode='gaussian', var=sigma, clip=False)
        sinogram_clipped = np.clip(sinogram_noisy, None, max_p)
        image_real = iradon(sinogram_clipped, theta=theta, circle=False)
        
        degraded_images.append(image_real)
    
    # Преобразуем список в NumPy массив [batch_size, H, W]
    degraded_np = np.stack(degraded_images, axis=0)
    
    # Преобразуем NumPy массив обратно в тензор [batch_size, 1, H, W]
    degraded_tensor = torch.from_numpy(degraded_np).unsqueeze(1).float().to(device)
    
    return degraded_tensor


