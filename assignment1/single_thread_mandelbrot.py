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

def create_fractal(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height
    
  for x in range(width):
    real = min_x + x * pixel_size_x
    if x % 10 == 0:
      print(x)
    for y in range(height):
      imag = min_y + y * pixel_size_y
      color = mandel(real, imag, iters)
      image[y, x] = color

for size in [10, 100, 1000, 10000, 20000, 30000, 40000]:
  runs = []
  for run in range(0, 1):
    image = np.zeros((size, size), dtype = np.uint8)
    start = timer()
    create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20) 
    dt = timer() - start
    runs.append(dt)
  average = sum(runs) / len(runs)
  print(f"Average runtime for size {size}: {average:.5f}s")
