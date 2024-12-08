import numpy as np
import matplotlib.pyplot as plt

# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        # """Fit the KDE model with the given data."""
        # TODO
        self.data = np.array(data) 

    def epanechnikov_kernel(self, x, xi):
        # """Epanechnikov kernel function."""
        # print ("hii")
        x1,y1=x
        x2,y2=xi
        distance = (((x1-x2)**2 + (y1-y2)**2)/(self.bandwidth*self.bandwidth))
        if(distance <= 1):
            # print (0.75*(1-distance))
            return 0.75*(1-distance)
        # print (0)
        return 0

    def evaluate(self, x):
        # """Evaluate the KDE at point x."""
        # # TODO
        n = len (self.data)
        print(n)
        h = self.bandwidth
        sum =0
        for i in range(n):
            # print(i)
            sum += (self.epanechnikov_kernel ( self.data[i], x)/h)
        return sum / n
        # sum=self.epanechnikov_kernel(self.data, x)/h

# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# TODO: Initialize the EpanechnikovKDE class
EpanKDE=EpanechnikovKDE(2)

# TODO: Fit the data
EpanKDE.fit(data)

# TODO: Plot the estimated density in a 3D plot
# xg,yg =np.mgrid[-5:5:20j,-5:5:20j]
# zg=np.zeros_like(xg)
# for i in range (xg.shape[0]):
#     for j in range (yg.shape[0]):
#         zg[i,j] =EpanKDE.evaluate(np.array([xg,yg]))
xg, yg = np.mgrid[-5:5:20j, -5:5:20j]
grid_points = np.array([xg.ravel(), yg.ravel()]).T
zg = EpanKDE.evaluate(grid_points).reshape(xg.shape)


fig = plt.figure(figsize=(10,7))
ax= fig.add_subplot(111,projection = '3d')
ax.plot_surface(xg, yg, zg, cmap='viridis', alpha=0.7)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Density')

# TODO: Save the plot 
plt.savefig('epanechnikov_density_plot.png')
plt.show()
