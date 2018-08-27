import numpy as np
import graph_utilities
import h5py
import time
import sys
import argparse
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
	sigShowImage = Signal(int)
	
	def __init__(self, eigvec, nclusters, labels, img_index, *args, **kwargs):
		
		super(PlotEigVectors, self).__init__(*args, **kwargs)
		self.eigvec = eigvec
		self.labels = labels
		self.nclusters = nclusters
		self.img_index = img_index
		self.sigClicked.connect(self.clicked)
	
	def plot_eigvec(self, eignum_1, eignum_2=None):
		
		all_spots = []
		all_indices = []
		s = 0
		for i in range(self.nclusters):
			
			ind = np.where(self.labels == i)
			img_ind = self.img_index[ind]
			b = np.array([pg.intColor(i, self.nclusters)] * np.size(ind))
			if eignum_2 is None:
				# plot of single eigen vector vs labels
				x = np.arange(s, s + np.size(ind))
				y = self.eigvec[ind, -eignum_1].flatten()
			else:
				x = self.eigvec[ind, -eignum_1].flatten()
				y = self.eigvec[ind, -eignum_2].flatten()
			
			spots = [{'x': xi, 'y': yi, 'data': img_indi, 'brush': bi}
			         for xi, yi, img_indi, bi in zip(x, y, img_ind, b)]
			
			all_indices.append(img_ind)
			for ii in range(np.size(ind)):
				all_spots.append(spots[ii])
			
			s += np.size(ind)
		
		self.setData(all_spots)
	
	def scatter_plot_data(self, eignum_1, eignum_2=None):
		
		all_spots = []
		all_indices = []
		s = 0
		for i in range(self.nclusters):
			
			ind = np.where(self.labels == i)
			img_ind = self.img_index[ind]
			b = np.array([pg.intColor(i, self.nclusters)] * np.size(ind))
			if eignum_2 is None:
				# plot of single eigen vector vs labels
				x = np.arange(s, s + np.size(ind))
				y = self.eigvec[ind, -eignum_1].flatten()
			else:
				x = self.eigvec[ind, -eignum_1].flatten()
				y = self.eigvec[ind, -eignum_2].flatten()
			
			spots = [{'x': xi, 'y': yi, 'data': img_indi, 'brush': bi}
			         for xi, yi, img_indi, bi in zip(x, y, img_ind, b)]
			
			all_indices.append(img_ind)
			for ii in range(np.size(ind)):
				all_spots.append(spots[ii])
			
			s += np.size(ind)
		
		return all_spots, all_indices
	
	def clicked(self, _, points):
		global lastclicked
		for p in lastclicked:
			p.resetPen()
		for point in points:
			point.setPen('b', width=2)
			index = point.data()
			self.sigShowImage.emit(index)
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
			ny, nx = fh[dfield][indices[0]].shape
			#nq = fh['data/saxs'][indices[0]].shape[0]
			dset_cart = fout.create_dataset(dfield, (npatt, ny, nx), dtype=float)
			#dset_saxs = fout.create_dataset('data/saxs', (npatt, nq), dtype=float)
			
			for i in range(npatt):
				dset_cart[i, :, :] = fh[dfield][indices[i], :, :]
				#dset_saxs[i, :] = fh['data/saxs'][indices[i], :]
			fout.create_dataset('data/timestamp', data=fh[tfield][indices])
			fout.create_dataset('data/mask', data=fh[mfield][()])
			fout.close()
	
	def affinityROI(self, fh, labels, image_ind):
		
		ny, nx = 300, 200
		indices = np.array(sorted(self.export()))
		#print indices
		
		metric = 'euclidean'
		knn = np.size(indices) - 1
		mutual = False
		
		label_roi = np.empty_like(indices)
		for i, ii in enumerate(indices):
			label_roi[i] = labels[np.where(image_ind == ii)][0]
		
		label_sorting_index = np.argsort(label_roi)
		label_sorted_data_ind = indices[label_sorting_index]
		
		print(label_sorted_data_ind)
		print(label_roi[label_sorting_index])
		
		d = np.zeros((np.size(indices), ny, nx))
		for i, ii in enumerate(label_sorted_data_ind):
			d[i, :, :] = fh[dfield][ii, 400:700, 400:600]
		spec_graph = graph_utilities.SpectralGraph(d, metric=metric)
		spec_graph.gaussian_similarity(sigma_ref=7)
		affinity, _ = spec_graph.knn_graph(knn=knn, mutual=mutual)
		self.sigSetImage.emit(affinity)
		
	def avgROI(self, fh):
		
		indices = sorted(self.export())
		
		avg = np.average(fh[dfield][indices, :, :], axis=0)
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
	cart_data = f[dfield]
        mask = f[mfield][()]
	ntotal = cart_data.shape[0]
	feature = 'cart'
	
	keys = kwargs.keys()
	if 'knn' in keys:
		knn = kwargs['knn']
	else:
		knn = ntotal - 1
	
	if 'metric' in keys:
		metric = kwargs['metric']
	else:
		metric = 'euclidean'
		
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
	
	select_ind = np.arange(nsamples)
	nsamples = select_ind.size

	x0,y0 = roi_0[0], roi_0[1]
        x1,y1 = roi_1[0], roi_1[1]
        print x0, y0, x1, y1
	mask_roi = mask[y0:y1, x0:x1]
        print(mask_roi.shape)
        mask_ind = np.nonzero(mask_roi.flatten())
        print('number of unmasked pixels in ROI = %4d' %(np.size(mask_ind)))
        t0 = time.time()
	if feature == 'saxs':
		d = saxs[select_ind, 20:80]
	else:
            d = cart_data[select_ind, y0:y1, x0:x1]
            print d.shape
	    #d = d.ravel()[mask_ind]
            #d = d / (np.sum(np.sum(d, axis=-1), axis=-1) + np.finfo(np.float32).eps)[:, None, None]
	
        t_load = time.time() - t0
	print('data loaded in %4.5f seconds' % t_load)
	print d.shape
	
	# Do clustering
	spec_graph = graph_utilities.SpectralGraph(d, metric=metric)
	spec_graph.gaussian_similarity(sigma_ref=7)
	affinity, _ = spec_graph.knn_graph(knn=knn, mutual=mutual)
	eigval, eigvec = graph_utilities.get_lap_eig(affinity)
	
	t0 = time.time()
	cluster_labels = clustering.spectral_clustering(eigvec, nclusters)
	t_cluster = time.time() - t0
	print('time taken for clustering = %4.5f' % t_cluster)
	
	return eigvec, cluster_labels, select_ind


def affinity_clusters(f, cluster_labels, nclusters, select_ind, **kwargs):
	
	cart_data = f[dfield]
	ntotal = cart_data.shape[0]
	
	keys = kwargs.keys()
	if 'knn' in keys:
		knn = kwargs['knn']
	else:
		knn = ntotal - 1
	
	if 'metric' in keys:
		metric = kwargs['metric']
	else:
		metric = 'euclidean'
	
	if 'mutual' in keys:
		mutual = kwargs['mutual']
	else:
		mutual = False
	
	ny = 300
	nx = 200
	d = np.zeros((np.size(cluster_labels), ny, nx))
	all_labels_sorted = []
	for i in range(nclusters):
		ind = select_ind[np.where(cluster_labels == i)]
		if np.size(ind) > 0:
			all_labels_sorted = np.append(all_labels_sorted, ind)
	
	for i, ii in enumerate(all_labels_sorted):
		d[i, :, :] = f[dfield][ii,400:700,400:600]
	
	spec_graph = graph_utilities.SpectralGraph(d, metric=metric)
	spec_graph.gaussian_similarity(sigma_ref=7)
	affinity, _ = spec_graph.knn_graph(knn=knn, mutual=mutual)
	
	return affinity


def cluster_data(f, **kwargs):
	
	# saxs = f['data/saxs']
	cart_data = f[dfield]
	mask = f[mfield][()]
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
	
	select_ind = np.arange(nsamples)
	nsamples = select_ind.size
	
        x0,y0 = roi_0[0], roi_0[1]
        x1,y1 = roi_1[0], roi_1[1]
	t0 = time.time()
        mask_roi = mask[y0:y1, x0:x1]
        mask_ind = np.nonzero(mask_roi) #.flatten())
        print(np.size(ind))
	if feature == 'saxs':
		d = saxs[select_ind, 20:80]
	else:
            d = cart_data[select_ind, y0:y1, x0:x1]
            d = d[:,mask_ind]
            #d = d.ravel()[mask_ind]
	    #d = d / (np.sum(np.sum(d, axis=-1), axis=-1) + np.finfo(np.float32).eps)[:, None,
            #		        None]
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
	sigInd1 = Signal(int)
	sigInd2 = Signal(int)
	
	def __init__(self, *args, **kwargs):
		super(ClusterPlotWidget, self).__init__(*args, **kwargs)
		layout = QVBoxLayout()
		self.setLayout(layout)
		#self.sigClicked.connect(self.clicked)
		
		# Comboboxes
		hbox = QHBoxLayout()
		layout.addLayout(hbox)
		self.index1 = QComboBox()
		self.index2 = QComboBox()
		self.index1.addItem('None')
		self.index2.addItem('None')
		self.index1.addItems([str(i) for i in range(10)])
		self.index2.addItems([str(i) for i in range(10)])
		hbox.addWidget(self.index1)
		hbox.addWidget(self.index2)
		
		# Plot widget
		self.plotwidget = pg.PlotWidget()
		layout.addWidget(self.plotwidget)
	
		#self.index1.currentIndexChanged(self.on_selector)
		
	def __getattr__(self, attr):  ## implicitly wrap methods from plotItem
		if hasattr(self.plotwidget.plotItem, attr):
			m = getattr(self.plotwidget.plotItem, attr)
			if hasattr(m, '__call__'):
				return m
		raise NameError(attr)
	
	# def clicked(self):
	#
	# 	self.sigInd1.emit(self.index1.currentText())
	#
	# def return_ind_1(self, i):
	#
	# 	#index = self.index1.itemText(i)
	# 	return self.index1.currentText()


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='hit-finding based on spectral clustering')
	#parser.add_argument('-h', '--help', help='')
	parser.add_argument('-f', '--fname', help='data filename', required=True)
	parser.add_argument('-df', '--dfield', help='data field', required=True)
	parser.add_argument('-mf', '--mfield', help='mask field', required=True)
	parser.add_argument('-tf', '--tfield', help='timestamp field', required=True)
	parser.add_argument('-nc', '--nclusters', help='number of clusters', default=5, type=int)
	parser.add_argument('-ns', '--nsamples', help='number of data points', default=100, type=int)
	parser.add_argument('-knn', '--neighbors', help='number of nearest neighbors', type=int)
	parser.add_argument('-m', '--metric', help='metric for distance calculation', default='euclidean', type=str)
	parser.add_argument('-g', '--graph', help='graph connection', default='fc', type=str)
        parser.add_argument('-roi','--roi',help='x1,y1,x2,y2 for region of interest',default='300,300,724,724') #,type=int)	
	args = parser.parse_args()
	
	# Get arguments
	if 'fname' in args:
		fname = args.fname
	else:
		print('no filename given: exiting')
		sys.exit()
	
	if 'dfield' in args:
		dfield = args.dfield
	else:
		print('no data field given: exiting')
	
	nclusters = args.nclusters
	nsamples = args.nsamples
	knn = args.neighbors
	graph = args.graph
	metric = args.metric
	mfield = args.mfield
	tfield = args.tfield
	roi = args.roi
        roi_0 = int(roi.split(',')[0]), int(roi.split(',')[1])
        roi_1 = int(roi.split(',')[2]), int(roi.split(',')[3])

        x0, y0 = roi_0[0], roi_0[1]
        x1, y1 = roi_1[0], roi_1[1]

	print(nclusters, nsamples, knn, graph, metric,roi_0,roi_1)

	# open file containing data
	f = h5py.File(fname, 'r')
	d = f[dfield]
	if nsamples > d.shape[0]:
		nsamples = d.shape[0]
	
	if knn >= nsamples:
		knn = nsamples - 1
	
	if graph == 'fc' or graph == 'fully-connected':
		mutual = False
		knn = nsamples - 1
	elif graph == 'mutual':
		mutual = True
	elif graph == 'knn' or graph == 'non-mutual':
		mutual = False
	
	# cluster the data
	arguments = {'nclusters': nclusters, 'nsamples': nsamples, 'knn': knn, 'mutual': mutual, 'metric': metric}
	eigvec, cluster_labels, select_ind = cluster_data_knn(f, **arguments)
	
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
	eigmenu = menubar.addMenu('&Eigenvectors')
	
	plot00 = ClusterPlotWidget()
	plot01 = ClusterPlotWidget()
	plot10 = ClusterPlotWidget()
	plot11 = ClusterPlotWidget()
	grid.addWidget(plot00, 0, 0, 1, 1)
	grid.addWidget(plot01, 0, 1, 1, 1)
	grid.addWidget(plot10, 1, 0, 1, 1)
	grid.addWidget(plot11, 1, 1, 1, 1)
	
	# Right splitter
	rightsplitter = QSplitter()
	rightsplitter.setOrientation(Qt.Vertical)
	splitter.addWidget(rightsplitter)
	# first image view
	img_view_1 = pg.ImageView()
	rightsplitter.addWidget(img_view_1)
	# Second image view
	img_view_2 = pg.ImageView()
	rightsplitter.addWidget(img_view_2)
	
	def showframe(index):
            img_view_1.setImage(f[dfield][index,y0:y1,x0:x1], autoLevels=False)
	
	# plot00item = ZoomScatterItem()
	# plot00item.sigShowCart.connect(showframe)
	# plot00.addItem(plot00item)
	#
	# plot10item = ZoomScatterItem()
	# plot10item.sigShowCart.connect(showframe)
	# plot10.addItem(plot10item)
	
	plot00item = PlotEigVectors(eigvec, nclusters, cluster_labels, select_ind)
	plot00item.sigShowImage.connect(showframe)
	plot00.addItem(plot00item)
	spots, indices = plot00item.scatter_plot_data(1, 2)
	plot00item.setData(spots)
	plot00item.setImageIndices(indices)
	
	roi00 = ExportROI(plot00item, (0, 0), (1, 1))
	plot00.addItem(roi00)
	roi00.sigSetImage.connect(img_view_2.setImage)
	roi00.setPos((0., 0.))
	roi00.setSize((0.01, 0.01))
	
	plot01item = PlotEigVectors(eigvec, nclusters, cluster_labels, select_ind)
	plot01item.sigShowImage.connect(showframe)
	plot01.addItem(plot01item)
	spots, indices = plot01item.scatter_plot_data(2, 3)
	plot01item.setData(spots)
	plot01item.setImageIndices(indices)
	
	plot10item = PlotEigVectors(eigvec, nclusters, cluster_labels, select_ind)
	plot10item.sigShowImage.connect(showframe)
	plot10.addItem(plot10item)
	spots, indices = plot01item.scatter_plot_data(2, 4)
	plot10item.setData(spots)
	plot10item.setImageIndices(indices)
	
	plot11item = PlotEigVectors(eigvec, nclusters, cluster_labels, select_ind)
	plot11item.sigShowImage.connect(showframe)
	plot11.addItem(plot11item)
	spots, indices = plot01item.scatter_plot_data(2, 4)
	plot11item.setData(spots)
	plot11item.setImageIndices(indices)
	
	roi01 = ExportROI(plot01item, (0, 1), (1, 1))
	plot01.addItem(roi01)
	roi01.sigSetImage.connect(img_view_2.setImage)
	roi01.setPos((0., 0.))
	roi01.setSize((0.1, 0.1))
	
	roi10 = ExportROI(plot10item, (1, 0), (1, 1))
	plot10.addItem(roi10)
	roi10.sigSetImage.connect(img_view_2.setImage)
	roi10.setPos((0., 0.))
	roi10.setSize((0.1, 0.1))
	
	roi11 = ExportROI(plot11item, (1, 0), (1, 1))
	plot11.addItem(roi11)
	roi11.sigSetImage.connect(img_view_2.setImage)
	roi11.setPos((0., 0.))
	roi11.setSize((0.1, 0.1))
	
	# ROI Menu bindings
	exportmenu00 = menubar.addMenu('&Export00')
	exportmenu00.addAction('&Average', partial(roi00.avgROI, f))
	exportmenu00.addAction('&Export Hits', partial(roi00.saveROI, f))
	exportmenu00.addAction('&Affinity', partial(roi00.affinityROI, f, cluster_labels, \
	                                            select_ind))

	exportmenu01 = menubar.addMenu('&Export01')
	exportmenu01.addAction('&Average', partial(roi01.avgROI, f))
	exportmenu01.addAction('&Export Hits', partial(roi01.saveROI, f))
	exportmenu01.addAction('&Affinity', partial(roi01.affinityROI, f, cluster_labels, \
	                                            select_ind))
	
	exportmenu10 = menubar.addMenu('&Export10')
	exportmenu10.addAction('&Average', partial(roi10.avgROI, f))
	exportmenu10.addAction('&Export Hits', partial(roi10.saveROI, f))
	exportmenu10.addAction('&Affinity', partial(roi10.affinityROI, f, cluster_labels, \
	                                            select_ind))
	
	exportmenu11 = menubar.addMenu('&Export11')
	exportmenu11.addAction('&Average', partial(roi11.avgROI, f))
	exportmenu11.addAction('&Export Hits', partial(roi11.saveROI, f))
	exportmenu11.addAction('&Affinity', partial(roi11.affinityROI, f, cluster_labels, \
	                                            select_ind))
	
	eigmenu.addAction('&Eig1', partial(plot11item.plot_eigvec, 1))
	eigmenu.addAction('&Eig2', partial(plot11item.plot_eigvec, 2))
	eigmenu.addAction('&Eig3', partial(plot11item.plot_eigvec, 3))
	eigmenu.addAction('&Eig4', partial(plot11item.plot_eigvec, 4))
	eigmenu.addAction('&Eig5', partial(plot11item.plot_eigvec, 5))
	eigmenu.addAction('&Eig6', partial(plot11item.plot_eigvec, 6))
	eigmenu.addAction('&Eig7', partial(plot11item.plot_eigvec, 7))
	eigmenu.addAction('&Eig8', partial(plot11item.plot_eigvec, 8))
	eigmenu.addAction('&Eig9', partial(plot11item.plot_eigvec, 9))
	eigmenu.addAction('&Eig10', partial(plot11item.plot_eigvec, 10))
	
	mainwindow.show()
	
	QApplication.instance().exec_()
