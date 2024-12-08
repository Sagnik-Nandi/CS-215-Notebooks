import numpy as np # math package for N-dimensional arrays and other stuff
import matplotlib.pyplot as plt # image input/output and graph drawing 

def calc_corr(img1,img2):#calculates the correlation coefficient between pixel intensities of
                         #2 images,img1 and img2
    img1=img1-np.mean(img1)
    img2=img2-np.mean(img2)
    return np.mean(img1*img2)/np.sqrt(np.mean(img1*img1)*np.mean(img2*img2))

def shift_image(img,tx):#shift img by tx pixels along x axis

    img_size=np.shape(img)[1]-20
    newimg=np.zeros_like(img)
    newimg=newimg.astype(img.dtype)
    newimg[:,10+tx:10+tx+img_size]=img[:,10:10+img_size]
    return newimg

#reading the image into a grayscale
img = plt.imread( 'Mona_Lisa.jpg' )
img_gray_withoutpadding = img[:,:,0] * 0.2989 + img[:,:,1] * 0.5870 + img[:,:,2] * 0.1140  # result is 2D array (matrix)
padding = ((0, 0), (10, 10))
#pdding the image to allow shifting
img_gray = np.pad(img_gray_withoutpadding, pad_width=padding, mode='constant', constant_values=0)
plt.imshow( img_gray, cmap='gray' )
plt.title("Original image") 
plt.show()

#examples for shfiting the image
img_new=shift_image(img_gray,5)
plt.imshow(img_new,cmap="gray")
plt.title("Original image shifted by 5 units to right")
plt.show()

img_new=shift_image(img_gray,-5)
plt.imshow(img_new,cmap="gray")
plt.title("Original image shifted by 5 units to left")
plt.show()

#calculating the 20 correlation coefficients
X=np.array([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])
Y=np.zeros_like(X)
Y=Y.astype(float)
for i in range(21):
    Y[i]=calc_corr(img_gray,shift_image(img_gray,X[i]))

#plotting the correlation coefficients as a function of x
plt.plot(X,Y)
plt.xlabel("distance shifted")
plt.ylabel("correlation coefficient")
plt.title("Correlation coefficients as a function of tx")
plt.show()      

#creating a normalized histogram
intensity=np.ndarray.flatten(img_gray_withoutpadding)
plt.hist(intensity,density=True)
plt.xlabel('Grayscale Intensities')
plt.ylabel('Normalized frequency')
plt.title('Normalized Histogram')
plt.show()

