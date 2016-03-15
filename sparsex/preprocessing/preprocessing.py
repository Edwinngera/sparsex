from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np

class Preprocessing:
    def __init__(self):
        pass

    ## Pipeline - Step 1
    def extract_patches(self, image, patch_size=(8,8)):
        patches = extract_patches_2d(image, patch_size)
        print patches.shape
        return patches

    ## Pipeline - Step 2
    def get_contrast_normalized_patches(self, patches):
        original_patches_shape = patches.shape
        patches = patches.reshape((patches.shape[0], -1))
        patches = (patches - patches.mean(axis=1)[:, np.newaxis]) / np.sqrt(patches.var(axis=1))[:, np.newaxis]
        patches = patches.reshape(original_patches_shape)
        return patches


    ## Pipeline - Step 3
    def get_whitened_patches(self, patches):
        for i in range(patches.shape[0]):
            sigma = np.cov(patches[i].T)
            u, s, v = np.linalg.svd(sigma, full_matrices=True)
            eps = np.finfo(s.dtype).eps
            whitener = np.dot(u, np.dot(np.diag(1./np.sqrt(s) + eps), u.T))
            patches[i] = np.dot(whitener, patches[i])
        return patches
