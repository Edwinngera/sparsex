from sklearn.feature_extraction.image import extract_patches_2d
from scipy.misc import imresize
import numpy as np

class Preprocessing:
    def __init__(self):
        pass

    ## Pipeline - Step 0
    def get_resized_image(self, image_array, image_size=(64,64)):
        """Returns (n',m') image from (n,m) image and (n',m') image size."""
        resized_image_array = imresize(image_array, image_size)
        return resized_image_array


    ## Pipeline - Step 1
    def extract_patches(self, image_array, patch_size=(8,8)):
        """Returns ((n-p+1)*(m-p+1),p,p) pathces from (n,m) image and (p,p) patch."""
        patches = extract_patches_2d(image_array, patch_size)
        return patches


    ## Pipeline - Step 2
    def get_contrast_normalized_patches(self, patches):
        """Returns (n,p,p) normalized_patches from (n,p,p) patches."""
        original_patches_shape = patches.shape
        patches = patches.reshape((patches.shape[0], -1))
        patches = (patches - patches.mean(axis=1)[:, np.newaxis]) / (np.sqrt(patches.var(axis=1))[:, np.newaxis] + 0.01)
        patches = patches.reshape(original_patches_shape)
        return patches


    ## Pipeline - Step 3
    def get_whitened_patches(self, normalized_patches):
        """Return (n,p,p) whitened_patches from (n,p,p) (normalized) patches."""
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
        """Returns ((n'-p+1)*(m'-p+1),p,p) whitened_patches from (n,m) image, (n',m') imsize, (p,p) patch."""
        resized_image_array = self.get_resized_image(image_array)
        patches = self.extract_patches(resized_image_array)
        normalized_patches = self.get_contrast_normalized_patches(patches)
        whitened_patches = self.get_whitened_patches(normalized_patches)
        return whitened_patches
