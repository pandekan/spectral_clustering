import numpy as np
from scipy.linalg import eigh, eig
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import time


class SpectralGraph:
	
	def __init__(self, data, metric='euclidean'):
		
		if len(data.shape) == 2:
			self.data = data
		else:
			self.data = data.reshape(data.shape[0], -1)
		
		self.nsamples = self.data.shape[0]
		self.metric = metric
		self.dist = squareform(pdist(self.data, metric=self.metric))
		self.similarity = np.zeros_like(self.dist)
		
	def gaussian_similarity(self, **kwargs):
		
		sigma = None
		sigma_ref = None
		
		keys = kwargs.keys()
		if 'sigma' in keys:
			sigma = kwargs['sigma']
			if sigma <= np.finfo(float).eps:
				sigma = None
		elif 'sigma_ref' in keys:
			sigma_ref = kwargs['sigma_ref']
		
		scaled_dist = np.zeros_like(self.dist)
		if sigma is None:
			if sigma_ref is None:
				sigma_ref = 7
			sigma_ind = np.argsort(self.dist, axis=1)[:, sigma_ref]
			sig = np.array(
				[self.dist[i, sigma_ind[i]] for i in range(self.nsamples)])
			for i in range(self.nsamples):
				sigma_ij = sig[i] * sig[:]
				scaled_dist[i, :] = -self.dist[i, :] ** 2 / sigma_ij[:]
		else:
			scaled_dist = -self.dist ** 2 / sigma ** 2
		
		# 0./0
		scaled_dist[np.where(np.isnan(scaled_dist))] = -10000.
		
		# similarity = exp(-d**2), and set diagonal elements to 0.
		self.similarity = np.exp(scaled_dist)
	
	def knn_graph(self, knn, mutual=False):
		
		assert(0 < knn <= self.nsamples - 1)
		
		adjacency = np.zeros_like(self.similarity)
		degree = np.zeros_like(self.similarity)
		
		if not mutual and knn > int(0.75*self.nsamples):
			adjacency = self.similarity
		else:
			for i in range(self.nsamples):
				
				ind = np.argsort(self.dist[i, :])
				if not mutual:
					adjacency[i, ind[1:knn+1]] = self.similarity[i, ind[1:knn+1]]
					adjacency[ind[1:knn+1], i] = self.similarity[ind[1:knn+1], i]
				else:
					for j in ind[1:knn+1]:
						indp = np.argsort(self.dist[j, :])
						if i in indp[1:knn+1]:
							adjacency[i, j] = self.similarity[i, j]
							adjacency[j, i] = self.similarity[j, i]
		
		for i in range(self.nsamples):
			degree[i, i] = np.sum(adjacency[i, :])
		
		return adjacency, degree
	
	def knn_graph_scikit(self, knn, mutual=False):
		
		assert(0 < knn <= self.nsamples - 1)
		
		adjacency = np.zeros_like(self.similarity)
		degree = np.zeros_like(self.similarity)
		if not mutual and knn > int(0.75*self.nsamples):
			adjacency = self.similarity
		else:
			neighbors = NearestNeighbors(knn, metric=self.metric, algorithm='brute',\
			                             njobs=-1).fit(self.data)
			neigh_graph = neighbors.kneighbors_graph(self.data, mode='connectivity').toarray()
			
			if mutual:
				for i in range(self.nsamples):
					for j in range(self.nsamples):
						if neigh_graph[i, j] * neigh_graph[j, i] > 0.:
							adjacency[i, j] = self.similarity[i, j]
							adjacency[j, i] = self.similarity[j, i]
			else:
				for i in range(self.nsamples):
					adjacency[i, :] = self.similarity[i, :] * neigh_graph[i, :]
		
		for i in range(self.nsamples):
			degree[i, i] = np.sum(adjacency[i, :])
			
		return adjacency, degree
	
		
def get_affinity(data, **kwargs):

	keys = kwargs.keys()

	allowed_metrics = ['euclidean', 'cosine', 'correlation', 'mahalanobis']
	if 'metric' in keys:
		metric = kwargs['metric']
	else:
		metric = 'euclidean'
	if metric not in allowed_metrics:
		metric = 'euclidean'

	if len(data.shape) == 2:
		n, m = data.shape
	elif len(data.shape) == 3:
		data = data.reshape(data.shape[0], -1)
		n, m = data.shape

	# get nXn distance matrix
	t0 = time.time()
	dist_matrix = squareform(pdist(data, metric=metric))
	t_dist = time.time() - t0
	print('time taken to compute distance matrix for %d samples = %4.5f' %(n, t_dist))
	
	# if no value for sigma is given, then compute sigma
	# based on input sigma_ref (default=7)
	sigma_in = None
	sigma_ref = None
	if 'sigma' in keys:
		sigma_in = kwargs['sigma']
		if sigma_in <= 0.:
			sigma_in = None

	if 'sigma_ref' in keys:
		sigma_ref = kwargs['sigma_ref']

	scaled_dist = np.zeros_like(dist_matrix)
	if sigma_in is None:
		if sigma_ref is None:
			sigma_ref = 7
		sigma_ind = np.argsort(dist_matrix,axis=1)[:,sigma_ref]
		sigma = np.array([dist_matrix[i, sigma_ind[i]] for i in range(dist_matrix.shape[0])])
		for i in range(n):
			sigma_ij = sigma[i]*sigma[:]
			scaled_dist[i,:] = -dist_matrix[i,:]**2/sigma_ij[:]
	else:
		scaled_dist = -dist_matrix**2/sigma_in**2

	# 0./0
	scaled_dist[np.where(np.isnan(scaled_dist))] = -10000.

	# affinity = exp(-d**2), and set diagonal elements to 0.
	affinity = np.exp(scaled_dist)
	for i in range(n):
		affinity[i, i] = 0.

	return affinity


def get_lap_eig(affinity):
	
	t0 = time.time()
	eps = np.finfo(np.float64).eps

	# make sure there are no NaNs
	affinity = np.nan_to_num(affinity)

	# symmetrize affinity
	a = np.tril(affinity, k=-1)
	aff = a + a.transpose()

	# compute diagonal matrix with sum of affinities
	d = np.diag(np.sum(aff, axis=0))
	inv_sqrtd = np.diag([1./np.sqrt(d[i, i]) if d[i, i] > eps else 0. for i in range(d.shape[0])])
	
	lap = np.dot(inv_sqrtd, np.dot(aff, inv_sqrtd))
	eigval, eigvec = eigh(lap)
	t_eig = time.time() - t0
	
	print('time taken for eigenvalue decomposition = %4.5f' %t_eig)
	
	return eigval, eigvec
