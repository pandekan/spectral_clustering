import numpy as np
import graph_utilities
import h5py
import time
import sys
import matplotlib.pyplot as plt
import clustering
import pyqtgraph as pg
from functools import partial

try:
	from PyQt5 import QtCore, QtGui
	from PyQt5.uic import loadUiType
	from PyQt5.QtGui import *
	from PyQt5.QtCore import *
	from PyQt5.QtWidgets import *
	from PyQt5.QtCore import pyqtSignal as Signal
except ImportError:
	from PyQt4 import QtCore, QtGui
	from PyQt4.uic import loadUiType
	from PyQt4.QtGui import *
	from PyQt4.QtCore import *
	from PyQt4.QtCore import pyqtSignal as Signal

lastclicked = []


class ZoomScatterItem(pg.ScatterPlotItem):
	sigShowCart = Signal(int)
	
	def __init__(self, *args, **kwargs):
		super(ZoomScatterItem, self).__init__(*args, **kwargs)
		self.sigClicked.connect(self.clicked)
	
	def clicked(self, _, points):
		global lastclicked
		for p in lastclicked:
			p.resetPen()
		for point in points:
			point.setPen('b', width=2)
			index = point.data()
			self.sigShowCart.emit(index)
		lastclicked = points
	
	def setImageIndices(self, indices):
		
		self.imageindices = indices
		
		# self.mousePoint = self.viewBox.mapSceneToView(position)
		# print(self.mousePoint.toTuple())


class PlotEigVectors(pg.ScatterPlotItem):
	sigShowEig = Signal(int)
	
	def __init__(self, eigvec, nclusters, labels, *args, **kwargs):
		
		super(PlotEigVectors, self).__init__(*args, **kwargs)
		self.eigvec = eigvec
		self.labels = labels
		self.nclusters = nclusters
		self.sigClicked.connect(self.clicked)
	
	def plot_eigvec(self, eignum):
		
		eig = self.eigvec[:, -eignum]
		s = 0
		all_spots = []
		
		for i in range(self.nclusters):
			
			ind = np.where(self.labels == i)
			# brush color
			b = np.array([pg.intColor(i, 10)] * np.size(ind))
			
			y = eig[ind].flatten()
			x = np.arange(s, s + np.size(ind))
			spots = [{'x': xi, 'y': yi, 'brush': bi} for xi, yi, bi in zip(x, y, b)]
			
			for ii in range(np.size(ind)):
				all_spots.append(spots[ii])
			
			s += np.size(ind)
			
			self.setData(all_spots)
	
	def clicked(self, _, points):
		global lastclicked
		for p in lastclicked:
			p.resetPen()
		for point in points:
			point.setPen('b', width=2)
			index = point.data()
			self.sigShowEig.emit(index)
		lastclicked = points
	
	def setImageIndices(self, indices):
		
		self.imageindices = indices


class ExportROI(pg.RectROI):
	sigSetImage = Signal(object)
	
	def __init__(self, bounditem, *args, **kwargs):
		super(ExportROI, self).__init__(*args, **kwargs)
		self.sigRegionChangeFinished.connect(self.export)
		self.bounditem = bounditem  # type: pg.ScatterPlotItem
	
	def export(self):
		exportpoints = []
		for point in self.bounditem.points():
			if (self.pos().x() < point.pos().x() < self.pos().x() + self.size().x()) and \
					(self.pos().y() < point.pos().y() < self.pos().y() + self.size().y()):
				exportpoints.append(point.data())
		
		return exportpoints
	
	def saveROI(self, fh):
		
		indices = sorted(self.export())
		
		dialog = QFileDialog(caption='Export to h5')  # default 'save_0.h5'
		dialog.setFileMode(QFileDialog.AnyFile)
		dialog.setAcceptMode(QFileDialog.AcceptSave)
		
		if dialog.exec_():
			
			path = dialog.selectedFiles()
			print('path:', str(path[0]))
			
			fout = h5py.File(str(path[0]), 'w')
			npatt = np.size(indices)
			ny, nx = fh['data/cart_data'][indices[0]].shape
			nq = fh['data/saxs'][indices[0]].shape[0]
			dset_cart = fout.create_dataset('data/cart_data', (npatt, ny, nx),
			                                dtype=float)
			dset_saxs = fout.create_dataset('data/saxs', (npatt, nq), dtype=float)
			
			for i in range(npatt):
				dset_cart[i, :, :] = fh['data/cart_data'][indices[i], :, :]
				dset_saxs[i, :] = fh['data/saxs'][indices[i], :]
			fout.create_dataset('data/timestamp', data=fh['data/timestamp'][indices])
			
			fout.close()
	
	def avgROI(self, fh):
		
		indices = sorted(self.export())
		
		avg = np.average(fh['data/cart_data'][indices, :, :], axis=0)
		self.sigSetImage.emit(avg)
	
	def acROI(self, fh):
		
		indices = sorted(self.export())
		n_indices = np.size(indices)
		pol_img = fh['data/polar_data']
		pol_msk = fh['data/polar_mask']
		
		_, nq, nphi = pol_img.shape
		ac_img = np.zeros((nq, nphi))
		ac_msk = np.zeros((nq, nphi))
		
		for i in indices:
			fft_img = np.fft.fft(pol_img[i, :, :], axis=1)
			fft_msk = np.fft.fft(pol_msk[i, :, :], axis=1)
			ac_img += np.fft.ifft(fft_img * fft_img.conj(), axis=1).real / float(
				n_indices)
			ac_msk += np.fft.ifft(fft_msk * fft_msk.conj(), axis=1).real / float(
				n_indices)


def cluster_data_knn(f, **kwargs):
	# saxs = f['data/saxs']
	cart_data = f['data/cart_data']
	ntotal = cart_data.shape[0]
	feature = 'cart'
	
	keys = kwargs.keys()
	if 'knn' in keys:
		knn = kwargs['knn']
	else:
		knn = ntotal - 1
	
	if 'mutual' in keys:
		mutual = kwargs['mutual']
	else:
		mutual = False
	
	if 'nclusters' in keys:
		nclusters = kwargs['nclusters']
	else:
		nclusters = 5  # default
	
	if 'nsamples' in keys:
		nsamples = kwargs['nsamples']
	else:
		nsamples = ntotal
	
	if nsamples == ntotal:
		select_ind = np.arange(0, nsamples)
	else:
		select_ind = np.unique(np.random.randint(0, ntotal, nsamples))
	
	print nsamples, nclusters, knn, mutual
	
	select_ind = np.arange(2000, 2000 + nsamples)
	nsamples = select_ind.size
	
	t0 = time.time()
	if feature == 'saxs':
		d = saxs[select_ind, 20:80]
	else:
		d = cart_data[select_ind, 400:700, 400:600]
		d = d / (np.sum(np.sum(d, axis=-1), axis=-1) +
		         np.finfo(np.float32).eps)[:, None, None]
	t_load = time.time() - t0
	print('data loaded in %4.5f seconds' % t_load)
	print d.shape
	
	# Do clustering
	spec_graph = graph_utilities.SpectralGraph(d, 'correlation')
	spec_graph.gaussian_similarity(sigma_ref=7)
	affinity, _ = spec_graph.knn_graph(knn=knn, mutual=mutual)
	eigval, eigvec = graph_utilities.get_lap_eig(affinity)
	
	t0 = time.time()
	cluster_labels = clustering.spectral_clustering(eigvec, nclusters)
	t_cluster = time.time() - t0
	print('time taken for clustering = %4.5f' % t_cluster)
	
	return eigvec, cluster_labels, select_ind


def cluster_data(f, **kwargs):
	# saxs = f['data/saxs']
	cart_data = f['data/cart_data']
	ntotal = cart_data.shape[0]
	feature = 'cart'
	
	keys = kwargs.keys()
	if 'nclusters' in keys:
		nclusters = kwargs['nclusters']
	else:
		nclusters = 5  # default
	
	if 'nsamples' in keys:
		nsamples = kwargs['nsamples']
	else:
		nsamples = ntotal
	
	if nsamples == ntotal:
		select_ind = np.arange(0, nsamples)
	else:
		select_ind = np.unique(np.random.randint(0, ntotal, nsamples))
	
	select_ind = np.arange(2000, 2000 + nsamples)
	nsamples = select_ind.size
	
	t0 = time.time()
	if feature == 'saxs':
		d = saxs[select_ind, 20:80]
	else:
		d = cart_data[select_ind, 400:700, 400:600]
		d = d / (np.sum(np.sum(d, axis=-1), axis=-1) + np.finfo(np.float32).eps)[:, None,
		        None]
	t_load = time.time() - t0
	print('data loaded in %4.5f seconds' % t_load)
	print d.shape
	
	# Do clustering
	eigval, eigvec = clustering.data_to_eig(d, sigma_ref=7)
	t0 = time.time()
	cluster_labels = clustering.spectral_clustering(eigvec, nclusters)
	t_cluster = time.time() - t0
	print('time taken for clustering = %4.5f' % t_cluster)
	
	return eigvec, cluster_labels, select_ind


class ClusterPlotWidget(QWidget):
	def __init__(self, *args, **kwargs):
		super(ClusterPlotWidget, self).__init__(*args, **kwargs)
		layout = QVBoxLayout()
		self.setLayout(layout)
		
		# Comboboxes
		hbox = QHBoxLayout()
		layout.addLayout(hbox)
		self.index1 = QComboBox()
		self.index2 = QComboBox()
		hbox.addWidget(self.index1)
		hbox.addWidget(self.index2)
		
		# Plot widget
		self.plotwidget = pg.PlotWidget()
		layout.addWidget(self.plotwidget)
	
	def __getattr__(self, attr):  ## implicitly wrap methods from plotItem
		if hasattr(self.plotwidget.plotItem, attr):
			m = getattr(self.plotwidget.plotItem, attr)
			if hasattr(m, '__call__'):
				return m
		raise NameError(attr)


if __name__ == "__main__":
	
	app = QApplication([])
	mainwindow = QMainWindow()
	splitter = QSplitter()
	mainwindow.setCentralWidget(splitter)
	grid = QGridLayout()
	grid.setContentsMargins(0, 0, 0, 0)
	grid.setSpacing(0)
	leftwidget = QWidget()
	leftwidget.setLayout(grid)
	splitter.addWidget(leftwidget)
	
	# Main Menu
	menubar = QMenuBar()
	mainwindow.setMenuBar(menubar)
	exportmenu = menubar.addMenu('&Export')
	eigmenu = menubar.addMenu('&Eigenvectors')
	
	# Right splitter
	rightsplitter = QSplitter()
	rightsplitter.setOrientation(Qt.Vertical)
	splitter.addWidget(rightsplitter)
	
	eigen12 = ClusterPlotWidget()
	eigen13 = ClusterPlotWidget()
	eigen23 = ClusterPlotWidget()
	eigen14 = ClusterPlotWidget()
	cart_view = pg.ImageView()
	eig_view = pg.PlotWidget()
	grid.addWidget(eigen12, 0, 0, 1, 1)
	grid.addWidget(eigen13, 0, 1, 1, 1)
	grid.addWidget(eigen23, 1, 0, 1, 1)
	grid.addWidget(eigen14, 1, 1, 1, 1)
	rightsplitter.addWidget(cart_view)
	
	# Second imageview
	cart_view2 = pg.ImageView()
	rightsplitter.addWidget(cart_view2)
	
	
	def showframe(index):
		cart_view.setImage(f['data/cart_data'][index], autoLevels=False)
	
	
	eigen12item = ZoomScatterItem()
	eigen12item.sigShowCart.connect(showframe)
	
	eigen13item = ZoomScatterItem()
	eigen13item.sigShowCart.connect(showframe)
	
	eigen23item = ZoomScatterItem()
	eigen23item.sigShowCart.connect(showframe)
	
	eigen14item = ZoomScatterItem()
	eigen14item.sigShowCart.connect(showframe)
	
	eigen12.addItem(eigen12item)
	eigen13.addItem(eigen13item)
	eigen23.addItem(eigen23item)
	eigen14.addItem(eigen14item)
	
	roi12 = ExportROI(eigen12item, (0, 0), (1, 1))
	eigen12.addItem(roi12)
	roi12.sigSetImage.connect(cart_view.setImage)
	
	roi13 = ExportROI(eigen13item, (0, 1), (1, 1))
	eigen13.addItem(roi13)
	
	roi23 = ExportROI(eigen23item, (1, 0), (1, 1))
	eigen23.addItem(roi23)
	
	roi14 = ExportROI(eigen14item, (1, 1), (1, 1))
	eigen14.addItem(roi14)
	
	# Get arguments
	fname = sys.argv[1]
	nclusters = int(sys.argv[2])
	nsamples = int(sys.argv[3])
	knn = int(sys.argv[4])
	mutual = bool(int(sys.argv[5]))
	
	f = h5py.File(fname, 'r')
	
	# ROI Menu bindings
	exportmenu.addAction('&Average', partial(roi12.avgROI, f))
	exportmenu.addAction('&Export Hits', partial(roi12.saveROI, f))
	
	arguments = {'nclusters': nclusters, 'nsamples': nsamples, 'knn': knn,
	             'mutual': mutual}
	eigvec, cluster_labels, select_ind = cluster_data_knn(f, **arguments)
	# eigvec, cluster_labels, select_ind = cluster_data(f, **arguments)
	color = iter(plt.cm.hsv(np.linspace(0, 1, nclusters)))
	
	all_spots12 = []
	all_spots13 = []
	all_spots23 = []
	all_spots14 = []
	all_img_ind = []
	
	for i in range(nclusters):
		c = color.next()
		ind = np.where(cluster_labels == i)
		img_ind = select_ind[ind]
		
		# brush color
		b = np.array([pg.intColor(i, 10)] * np.size(ind))
		print i, np.size(ind)
		
		x = eigvec[ind, -1].flatten()
		y = eigvec[ind, -2].flatten()
		spots12 = [{'x': xi, 'y': yi, 'data': img_indi, 'brush': bi} \
		           for xi, yi, img_indi, bi in zip(x, y, img_ind, b)]
		
		roi12.setPos((x.min(), y.min()))
		roi12.setSize(((x.max() - x.min()) / 10., (y.max() - y.min()) / 10.))
		
		x = eigvec[ind, -1].flatten()
		y = eigvec[ind, -3].flatten()
		spots13 = [{'x': xi, 'y': yi, 'data': img_indi, 'brush': bi} \
		           for xi, yi, img_indi, bi in zip(x, y, img_ind, b)]
		
		roi13.setPos((x.min(), y.min()))
		roi13.setSize(((x.max() - x.min()) / 10., (y.max() - y.min()) / 10.))
		
		x = eigvec[ind, -2].flatten()
		y = eigvec[ind, -3].flatten()
		spots23 = [{'x': xi, 'y': yi, 'data': img_indi, 'brush': bi} \
		           for xi, yi, img_indi, bi in zip(x, y, img_ind, b)]
		
		roi23.setPos((x.min(), y.min()))
		roi23.setSize(((x.max() - x.min()) / 10., (y.max() - y.min()) / 10.))
		
		x = eigvec[ind, -1].flatten()
		y = eigvec[ind, -4].flatten()
		spots14 = [{'x': xi, 'y': yi, 'data': img_indi, 'brush': bi} \
		           for xi, yi, img_indi, bi in zip(x, y, img_ind, b)]
		
		roi14.setPos((x.min(), y.min()))
		roi14.setSize(((x.max() - x.min()) / 10., (y.max() - y.min()) / 10.))
		
		all_img_ind.append(img_ind)
		for ii in range(np.size(ind)):
			all_spots12.append(spots12[ii])
			all_spots13.append(spots13[ii])
			all_spots23.append(spots23[ii])
			all_spots14.append(spots14[ii])
	
	eigen12item.setData(all_spots12)
	eigen12item.setImageIndices(all_img_ind)
	
	# eigen13item.setData(all_spots13)
	# eigen13item.setImageIndices(all_img_ind)
	
	eigen23item.setData(all_spots23)
	eigen23item.setImageIndices(all_img_ind)
	
	eigen14item.setData(all_spots14)
	eigen14item.setImageIndices(all_img_ind)
	
	# eigen13item = ZoomScatterItem()
	
	ploteig = PlotEigVectors(eigvec, nclusters, cluster_labels)
	# ploteig.sigShowEig.connect(showFrame)
	# ploteig.addItem()
	# eigen13item.sigShowEig.connect(showframe)
	eigen13.addItem(ploteig)
	
	eigmenu.addAction('&Eig1', partial(ploteig.plot_eigvec, 1))
	eigmenu.addAction('&Eig2', partial(ploteig.plot_eigvec, 2))
	eigmenu.addAction('&Eig3', partial(ploteig.plot_eigvec, 3))
	eigmenu.addAction('&Eig4', partial(ploteig.plot_eigvec, 4))
	eigmenu.addAction('&Eig5', partial(ploteig.plot_eigvec, 5))
	eigmenu.addAction('&Eig6', partial(ploteig.plot_eigvec, 6))
	eigmenu.addAction('&Eig7', partial(ploteig.plot_eigvec, 7))
	eigmenu.addAction('&Eig8', partial(ploteig.plot_eigvec, 8))
	eigmenu.addAction('&Eig9', partial(ploteig.plot_eigvec, 9))
	eigmenu.addAction('&Eig10', partial(ploteig.plot_eigvec, 10))
	
	mainwindow.show()
	
	QApplication.instance().exec_()
