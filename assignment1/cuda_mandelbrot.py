from numba import cuda
from numba import *
import numpy as np
import matplotlib.pyplot as pyplot
from timeit import default_timer as timer


def mandel(x, y, max_iters):
  """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4:
      return i

  return max_iters

mandel_gpu = cuda.jit(restype=uint32, argtypes=[f8, f8, uint32], device=True)(mandel)

@cuda.jit(argtypes=[f8, f8, f8, f8, uint8[:,:], uint32])
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  startX, startY = cuda.grid(2)
  gridX = cuda.gridDim.x * cuda.blockDim.x
  gridY = cuda.gridDim.y * cuda.blockDim.y

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y 
      image[y, x] = mandel_gpu(real, imag, iters)
for size in [10, 100, 1000, 10000, 20000, 30000, 40000]:
  runs = []
  for run in range(0, 10):
    gimage = np.zeros((size, size), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32,16)
    
    start = timer()
    d_image = cuda.to_device(gimage)
    mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20) 
    d_image.to_host()
    dt = timer() - start
    runs.append(dt)
    # print(f"Size: {size}, Run: {run}, Time: {dt}s")
  average = sum(runs) / len(runs)
  print(f"Average runtime for size {size}: {average:.5f}s")