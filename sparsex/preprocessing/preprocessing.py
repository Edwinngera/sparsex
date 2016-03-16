from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np

class Preprocessing:
    def __init__(self):
        pass

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
        patches = normalized_patches # alias for convenience
        for i in range(patches.shape[0]):
            sigma = np.cov(patches[i].copy().T)
            u, s, v = np.linalg.svd(sigma, full_matrices=True)
            #eps = np.finfo(s.dtype).eps
            eps = 0.01
            whitener = np.dot(u, np.dot(np.diag( 1./ np.sqrt(s) + eps), u.T))
            patches[i] = np.dot(whitener, patches[i])
        return patches

    ## Pipeline - Step 3 (Alternative)
    def zca_whitening(self, normalized_patches):
        patches = normalized_patches # alias for convenience
        for i in range(patches.shape[0]):
            inputs = patches[i].copy()
            sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
            U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
            epsilon = 0.1                #Whitening constant, it prevents division by zero
            ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
            patches[i] = np.dot(ZCAMatrix, inputs)   #Data whitening
        return patches


    ## Pipeline Combined
    def get_whitened_patches_from_image_array(self, image_array):
        patches = self.extract_patches(image_array)
        normalized_patches = self.get_contrast_normalized_patches(patches)
        whitened_patches = self.get_whitened_patches(normalized_patches)
        return whitened_patches
