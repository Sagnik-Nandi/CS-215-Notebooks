

import numpy as np
import matplotlib.pyplot as plt

# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        # """Fit the KDE model with the given data."""
        self.data = np.array(data)  # store data points as a NumPy array 
    
    def epanechnikov_kernel(self, x, xi): # predicting kernel density estimate at x for the data given (xi)
        """Epanechnikov kernel function."""
        # Distance calculation for all xi
        difference = (x[:, np.newaxis, :] - xi[np.newaxis, :, :])  # (M, N, 2) where M = No of grid points, N =  No of data points
        distSquare = np.sum((difference / self.bandwidth) ** 2, axis=-1)  # Shape (M, N)

        # Epanechnikov kernel condition
        ans = np.where(distSquare <= 1, (2/np.pi) * (1 - distSquare), 0)  # Shape (M, N) # 0 when diistance square >1 else 0.75 * (1 - distSquare)
        return ans  

    def evaluate(self, x):
        kerVal = self.epanechnikov_kernel(x, self.data)
        ans = np.sum(kerVal, axis=1) / (self.data.shape[0] * self.bandwidth*self.bandwidth)
        return ans

# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

#Initialize the EpanechnikovKDE class
EpanKDE=EpanechnikovKDE(1)

#Fit the data
EpanKDE.fit(data)

xg, yg = np.mgrid[-5:5:40j, -5:5:40j]
grid_points = np.array([xg.ravel(), yg.ravel()]).T
zg = EpanKDE.evaluate(grid_points).reshape(xg.shape)


fig = plt.figure(figsize=(10,7))
ax= fig.add_subplot(111,projection = '3d')
ax.plot_surface(xg, yg, zg, cmap='viridis', alpha=0.7)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Density')

#Save the plot 
plt.savefig('epanechnikov_density_plot.png')
plt.show()
