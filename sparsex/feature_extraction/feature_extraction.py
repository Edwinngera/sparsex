from sklearn.decomposition import DictionaryLearning
from ..tests.preprocessing_test import test_whitening
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class SparseCoding:
	def __init__(self):

		self.n_components = 100
		self.n_features = 64
		self.dict_init = np.random.rand(self.n_components, self.n_features)
		self.code_init = np.random.rand(self.n_components)
		self.max_iter = 5
		self.random_seed = 1111

		self.DL_obj = DictionaryLearning(n_components=self.n_components,
									   alpha=1,
									   max_iter=self.max_iter,
									   tol=1e-08,
									   fit_algorithm='lars',
									   transform_algorithm='omp',
									   transform_n_nonzero_coefs=None,
									   transform_alpha=None,
									   n_jobs=1,
									   code_init=None,
									   dict_init=self.dict_init,
									   verbose=False,
									   split_sign=False,
									   random_state=self.random_seed)


	def learn_dictionary(self, whitened_patches):
		# assert correct dimensionality of input data
		assert whitened_patches.ndim == 2, \
		"Whitened patches ndim is %d instead of 2" %whitened_patches.ndim

		# learn dictionary
		self.DL_obj.fit(whitened_patches)

		# print code
		print self.DL_obj.code_init

		return


	def get_sparse_code(self, whitened_patches):
		# assert correct dimensionality of input data
		assert whitened_patches.ndim == 2, \
		"Whitened patches ndim is %d instead of 2" %whitened_patches.ndim

		sparse_code = self.DL_obj.transform(whitened_patches)

		return sparse_code


if __name__ == "__main__":
	image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
	
	# get whitened patches
	whitened_patches = test_whitening(image_filename, False, False)

	# flatten individual patches
	whitened_patches = whitened_patches.reshape((whitened_patches.shape[0], -1))

	# create sparse coding object
	sparse_coding = SparseCoding()

	# learn dictionary on whitened patches
	sparse_coding.learn_dictionary(whitened_patches[:10])
	
	# get sparse code
	sparse_code = sparse_coding.get_sparse_code(whitened_patches[100:101])

	print sparse_code



