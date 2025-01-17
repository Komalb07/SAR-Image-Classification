from osgeo import gdal
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance
from astropy.convolution import convolve as ap_convolve
from astropy.convolution import Box2DKernel
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from skimage.metrics import mean_squared_error

def refined_lee_filter(img, size):
	img_mean = uniform_filter(img, (size, size))
	img_sqr_mean = uniform_filter(img**2, (size, size))
	img_variance = img_sqr_mean - img_mean**2
	overall_variance = variance(img)
	img_weights = img_variance/ (img_variance + overall_variance)
	img_output = img_mean + img_weights * (img - img_mean)
	return img_output

def boxcar_filter(img, kernel):
	box_kernel = Box2DKernel(kernel)
	output = ap_convolve(img, box_kernel, normalize_kernel=True)
	return output

def median_filter(data, kernel_size):
	indexer = kernel_size//2
	padding = np.zeros((data.shape[0]+indexer,data.shape[1]+indexer))
	padding[indexer:padding.shape[0]-indexer+1, indexer:padding.shape[1]-indexer+1] = data
	for i in range(indexer, padding.shape[0] - indexer+1):
		for j in range(indexer, padding.shape[1] - indexer+1):
			padding[i,j] = np.median(padding[i - indexer:i+indexer+1, j-indexer:j+indexer+1])
	return padding

def mean_filter(data, kernel_size):
	indexer = kernel_size//2
	padding = np.zeros((data.shape[0]+indexer,data.shape[1]+indexer))
	padding[indexer:padding.shape[0]-indexer+1, indexer:padding.shape[1]-indexer+1] = data
	for i in range(indexer, padding.shape[0] - indexer+1):
		for j in range(indexer, padding.shape[1] - indexer+1):
			padding[i,j] = np.mean(padding[i - indexer:i+indexer+1, j-indexer:j+indexer+1])
	return padding

def convolution(oldimage, kernel):
	image_h = oldimage.shape[0]
	image_w = oldimage.shape[1]
	kernel_h = kernel.shape[0]
	kernel_w = kernel.shape[1]
	image_pad = np.pad(oldimage, pad_width=((kernel_h//2, kernel_h//2), (kernel_w//2, kernel_w//2)), mode= 'constant', constant_values=0).astype(np.float32)
	h = kernel_h//2
	w = kernel_w//2
	image_conv = np.zeros(image_pad.shape)
	for i in range(h, image_pad.shape[0]-h):
		for j in range(w, image_pad.shape[1]-w):
			x = image_pad[i-h:i-h+kernel_h, j-w:j-w+kernel_w]
			x = x.flatten()*kernel.flatten()
			image_conv[i][j] = x.sum()
	h_end = -h
	w_end = -w
	if h==0:
		return image_conv[h:,w:w_end]
	if w==0:
		return image_conv[h:h_end,w:]
	return image_conv[h:h_end,w:w_end]

def GaussianBlurImage(image, sigma):
	image = Image.fromarray(image)
	image = np.asarray(image)
	filter_size = int(4*sigma+0.5)+1
	gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
	m = filter_size//2
	n = filter_size//2
	for x in range(-m, m+1):
		for y in range(-n, n+1):
			x1 = 2*np.pi*(sigma**2)
			x2 = np.exp(-(x**2 + y**2)/(2*sigma**2))
			gaussian_filter[x+m, y+n] = (1/x1)*x2
		im_filtered = np.zeros_like(image, dtype= np.float32)
		im_filtered = convolution(image, gaussian_filter)
	return im_filtered.astype(np.float32)

palsar = gdal.Open('ALPSRP255220340-H1.1__A/VOL-ALPSRP255220340-H1.1__A')

cols = palsar.RasterXSize
rows = palsar.RasterYSize
bands = palsar.RasterCount

HH = palsar.GetRasterBand(1)
HV = palsar.GetRasterBand(2)

HH_arr = HH.ReadAsArray()
HV_arr = HV.ReadAsArray()

HH_intensity = np.absolute(HH_arr)**2
HV_intensity = np.absolute(HV_arr)**2

fig,ax = plt.subplots(1,1,figsize = (12,8))
plt.imshow(np.angle(HH_arr,deg = True), cmap='gray')
ax.title.set_text("SLC Image")
plt.colorbar()

HH_intensity = np.where(HH_intensity==0.,0.01, HH_intensity)
HV_intensity = np.where(HV_intensity==0.,0.01, HV_intensity)

fig,ax = plt.subplots(1,1,figsize = (12,8))
plt.imshow(np.log(np.flipud(HH_intensity)), cmap='gray')
ax.title.set_text("Enhanced SLC Image")
plt.colorbar()


#Multilook
numlooks = 5
ind = np.array(range(0,numlooks*math.floor((HH_arr.shape[0])/numlooks), numlooks))
lookedHH = HH_arr[ind,:]
lookedHV = HV_arr[ind,:]
for i in range(numlooks+1):
    for look in range(numlooks-1):
        lookedHH+=HH_arr[ind+look,:]
        lookedHV+=HV_arr[ind+look,:]

sigmaHH = 10*np.log10(np.absolute(lookedHH)**2)-83-32
image = sigmaHH[500:1600, 1850:3500]

fig,ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.title.set_text("Multilooked image, by factor {}".format(numlooks))



#BoxCar Filter
Box = boxcar_filter(image, 3)
fig, ax = plt.subplots()
ax.imshow(Box, cmap = 'gray')
ax.title.set_text("Boxcar Filter")

#Refined_lee Filter
Refined = refined_lee_filter(image, 7)
fig, ax = plt.subplots()
ax.imshow(Refined, cmap = 'gray')
ax.title.set_text("Refined lee Filter")


#Mean Filter
Mean = mean_filter(image, 3)[0:1100, 0:1650]
fig, ax = plt.subplots()
ax.imshow(Mean, cmap = 'gray')
ax.title.set_text("Mean Filter")


#Median Filter
Median = median_filter(image, 3)[0:1100, 0:1650]
fig, ax = plt.subplots()
ax.imshow(Median, cmap = 'gray')
ax.title.set_text("Median Filter")


#Gaussian Filter
Gaussian = GaussianBlurImage(image, 3)
fig, ax = plt.subplots()
ax.imshow(Gaussian, cmap = 'gray')
ax.title.set_text("Gaussian Filter")

mse_mean = mean_squared_error(image, Mean)
mse_median = mean_squared_error(image, Median)
mse_gaussian = mean_squared_error(image, Gaussian)
mse_refined = mean_squared_error(image, Refined)
mse_box = mean_squared_error(image, Box)

#Computing Mean Squared error values of each filter to find which filter performed well - The lower the better
print("MSE value of mean filter")
print(mse_mean)
print("MSE value of median filter")
print(mse_median)
print("MSE value of gaussian filter")
print(mse_gaussian)
print("MSE value of refined lee filter")
print(mse_refined)
print("MSE value of boxcar filter")
print(mse_box)

#Classification
r,c = Box.shape
classes = {'urban': 0, 'vegetation': 1, 'water': 2}
n_classes = len(classes)
palette = np.uint8([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
X = Box.reshape(r*c, 1)

#Kmeans
kmeans = KMeans(n_clusters=n_classes, random_state=3).fit(X)
unsupervised = kmeans.labels_.reshape(r,c)
k_m =  Image.fromarray(palette[unsupervised])
k_m.show()


#Supervised
supervised = n_classes*np.ones(shape=(r,c), dtype=int)
supervised[400:430, 1570:1600] = classes['water']
supervised[990:1010, 1380:1410] = classes['vegetation']
supervised[390:410, 50:70] = classes['water']
supervised[390:420, 840:860] = classes['urban']
y = supervised.ravel()
train = np.flatnonzero(supervised < n_classes)
test = np.flatnonzero(supervised == n_classes)

#SVM
sv = SVC(gamma='auto')
sv.fit(X[train], y[train])
y[test] = sv.predict(X[test])
supervised_sv = y.reshape(r, c)
svm = Image.fromarray(palette[supervised_sv])
svm.show()
print("/***Accuracy score of SVM***\\")
print("{:.2f}".format(sv.score(X[train], y[train]) * 100))

#Random_Forest
random_forest = RandomForestClassifier()
random_forest.fit(X[train], y[train])
y[test] = random_forest.predict(X[test])
supervised_rf = y.reshape(r, c)
rf = Image.fromarray(palette[supervised_rf])
rf.show()
print("/***Accuracy score of Random forest***\\")
print("{:.2f}".format(random_forest.score(X[train], y[train])*100))

#Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X[train], y[train])
y[test] = decision_tree.predict(X[test])
supervised_dt = y.reshape(r, c)
dt = Image.fromarray(palette[supervised_dt])
dt.show()
print("/***Accuracy score of Decision Tree***\\")
print("{:.2f}".format(decision_tree.score(X[train], y[train])*100))


plt.show()