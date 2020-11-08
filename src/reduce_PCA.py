# IMPORT NECESSARY LIBRARIES
import cv2
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# IMPORTING IMAGE USING SCIPY AND TAKING R,G,B COMPONENTS

a = Image.open('00000013.jpg')
original = plt.imread('00000013.jpg')

a_np = np.array(a)
a_r = a_np[:,:,0]
a_g = a_np[:,:,1]
a_b = a_np[:,:,2]

def comp_2d(image_2d): # FUNCTION FOR RECONSTRUCTING 2D MATRIX USING PCA
	r,w = image_2d.shape
	cov_mat = image_2d - np.mean(image_2d , axis = 1)[:,None]
	eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat)) # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
	p = np.size(eig_vec, axis =1)
	idx = np.argsort(eig_val)
	idx = idx[::-1]
	eig_vec = eig_vec[:,idx]
	eig_val = eig_val[idx]
	numpc = 10 # THIS IS NUMBER OF PRINCIPAL COMPONENTS, YOU CAN CHANGE IT AND SEE RESULTS
	if numpc <p or numpc >0:
		eig_vec = eig_vec[:, range(numpc)]
	score = np.dot(eig_vec.T, cov_mat)
	recon = np.dot(eig_vec, score) + np.mean(image_2d, axis = 1).T[:,None] # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
	recon_img_mat = np.uint8(np.absolute(recon)) # TO CONTROL COMPLEX EIGENVALUES
	return recon_img_mat

a_r_recon, a_g_recon, a_b_recon = comp_2d(a_r), comp_2d(a_g), comp_2d(a_b) # RECONSTRUCTING R,G,B COMPONENTS SEPARATELY
recon_color_img = np.dstack((a_r_recon, a_g_recon, a_b_recon)) # COMBINING R.G,B COMPONENTS TO PRODUCE COLOR IMAGE

size = 600
original = cv2.resize(original, (size, size))
recon_color_img = cv2.resize(recon_color_img, (size, size))

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original)
f.add_subplot(1, 2, 2)

plt.title("PCA image with " + str(100) + "components")
plt.imshow(recon_color_img)
plt.savefig('Images\\Image_' + str(1) + '.png')
image_ID = +1


