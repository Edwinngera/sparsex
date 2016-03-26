from sklearn.feature_extraction.image import extract_patches_2d
from scipy.misc import imresize
import numpy as np

class Preprocessing:
    def __init__(self):
        pass

    ## Pipeline - Step 0
    def get_resized_image(self, image_array, image_size=(64,64)):
        resized_image_array = imresize(image_array, image_size)
        return resized_image_array


    ## Pipeline - Step 1
    def extract_patches(self, image_array, patch_size=(8,8)):
        patches = extract_patches_2d(image_array, patch_size)
        return patches


    ## Pipeline - Step 2
    def get_contrast_normalized_patches(self, patches):
        original_patches_shape = patches.shape
        patches = patches.reshape((patches.shape[0], -1))
        patches = (patches - patches.mean(axis=1)[:, np.newaxis]) / np.sqrt(patches.var(axis=1))[:, np.newaxis]
        patches = patches.reshape(original_patches_shape)
        return patches


    ## Pipeline - Step 3
    def get_whitened_patches(self, normalized_patches):
        # original patches shape
        original_patches_shape = normalized_patches.shape

        # alias for convenience
        patches = normalized_patches

        # flatten individual patches such that rows = no. of patches, columns = pixel values
        patches = patches.reshape((patches.shape[0], -1))

        # Transpose to get matrix A
        A = patches.T

        # sigma = (1/(n-1)) * np.dot(A,A.T), where A = patches.T
        sigma = (1.0 / (A.shape[1] - 1)) * np.dot(A, A.T)

        # decompose and reconstruct
        u, s, v = np.linalg.svd(sigma, full_matrices=True)

        # whitening matrix
        epsilon = 0.01
        whitening_matrix = np.dot(u, np.dot((1.0 / (np.sqrt(np.diag(s)) + epsilon)), u.T))

        # Awhite = W.A
        whitened_A = np.dot(whitening_matrix, A)

        # Transpose to get back normalized patches
        whitened_patches = whitened_A.T

        # reshape whitened patches
        whitened_patches = whitened_patches.reshape(original_patches_shape)

        return whitened_patches


    ## Pipeline Combined
    def get_whitened_patches_from_image_array(self, image_array):
        resized_image_array = self.get_resized_image(image_array)
        patches = self.extract_patches(resized_image_array)
        normalized_patches = self.get_contrast_normalized_patches(patches)
        whitened_patches = self.get_whitened_patches(normalized_patches)
        return whitened_patches
