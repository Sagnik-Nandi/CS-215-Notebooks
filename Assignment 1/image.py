import numpy as np
import matplotlib.pyplot as plt

img=plt.imread("Mona_Lisa.jpg")

plt.imshow( img )
plt.gca().set_xlabel( 'x' ) # set the x-label of the current Axes (returned by the gca)
plt.gca().set_ylabel( 'y' ) # set the y-label of the current Axes
