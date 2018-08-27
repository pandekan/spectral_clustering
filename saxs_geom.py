import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys


def polar_img(cart_img, cy=None, cx=None):
	
	if cy is None:
		cy = cart_img.shape[0]//2
	
	if cx is None:
		cx = cart_img.shape[1]//2
		

def radial_avg(data, cy=None, cx=None):
	
	y, x = np.indices(data.shape)
	if cy is None:
		cy = data.shape[0]//2
	
	if cx is None:
		cx = data.shape[1]//2
		
	r = np.hypot((y-cy), (x-cx))
	r = r.astype(int)
	
	data_bin = np.bincount(r.ravel(), data.ravel())
	nr = np.bincount(r.ravel())
	
	rad_avg = data_bin/nr
	
	return rad_avg


fname = sys.argv[1]

f = h5py.File(fname, 'r')
quad = f['data/00'][()]
q_ny, q_nx = quad.shape

del_cy, del_cx = np.meshgrid(np.arange(-5, 6), np.meshgrid(-5, 6), indexing='ij')
for i in range(10):
	new_img = np.zeros((2*(q_ny+del_cy.flatten()[i]), 2*(q_nx+del_cx.flatten()[i])))
	new_img[:512, :512] = quad.copy()
	rad_avg = radial_avg(new_img, del_cy.flatten()[i], del_cx.flatten()[i])
	plt.plot(np.log10(rad_avg[100:-100] + np.finfo(float).eps))
	
plt.show()
