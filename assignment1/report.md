# Assignment 1
Matthew Manley

I followed the python madelbrot example here http://nbviewer.ipython.org/gist/harrism/f5707335f40af9463c43

I have a laptop with a discrete graphics card (Nvidia GTX 960M) so I was able to run the sample code and compare the results locally. I ran the code at multiple resolutions to really test how well each implementation scales. I ran each of the cuda calcuations 10 times and took the average of all 10 runs. For the single threaded calculations, I did them 10 times for the 100x100 and the 1k x 1k, but only did one run of the 10k x 10k and didn't even attempt to do a single threaded run at larger resolutions

| Resolution | Single Time (seconds) | Cuda Time (seconds) | Times Faster |
|------------|-----------------------|---------------------|--------------|
| 100 x 100  | 0.04670               | 0.00192             | 15.77        |
| 1k x 1k    | 4.52389               | 0.00680             | 591.24       |
| 10k x 10k  | 471.88547             | 0.44887             | 1051.28      |
| 20k x 20k  |                       | 1.78282             |              |
| 30k x 30k  |                       | 4.02415             |              |    
| 40k x 40k  |                       | 7.10316             |              |

It is obvious that parallel computation of mandelbrot sets is required at higher resolutions. For a 40k x 40k resolution mandelbrot set, there are 1 billion 600 million pixels that need to be calculated. A rough estimate shows that calculating a 40k x 40k mandelbrot set with a single thread would take almost 9 days on my computer, making the cuda version of the application run roughly 106,000 times faster than the single threaded version.

(40000\*40000)/(1000\*1000) * 471 = 753600 seconds7.10316

While increasing resolution of a mandelbrot set is not a practical use of multithreading unless you crank up the iterations (which I had set to 20 for all tests), it does demonstrate the capability of single vs multi-threaded computing.