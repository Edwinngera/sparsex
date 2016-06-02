from ..customutils.customutils import is_perfect_square, isqrt
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.misc import imresize
import numpy as np

class Preprocessing:
    def __init__(self):
        pass


    def get_resized_image(self, image_array, image_size=(64,64), multiple_images=False):
        """Returns (n',m') image from (n,m) image and (n',m') image size for single image."""

        # validate image_size
        assert image_size[0] == image_size[1], "image_size is {0} instead of being square".format(image_size)

        # multiple images
        # expecting shape (number_images, x, y)
        if multiple_images:
            assert image_array.ndim == 3, "image_array.ndim is {0} instead of 3".format(image_array.ndim)

            # storing original shape for use later
            number_images = image_array.shape[0]

            # creating empty array to store resized images into
            resized_image_array = np.empty((number_images, image_size[0], image_size[1]), dtype=float)

            # for every image in the array, resize it and populate the empty resized array
            for image_index in range(number_images):
                resized_image_array[image_index] = imresize(image_array[image_index], image_size)

        else:
            # resize image array
            resized_image_array = imresize(image_array, image_size)

        # returning shape (number_images, image_size[0], image_size[1]) for multiple_images
        # returning shape (image_size[0], image_size[1]) for single image
        return resized_image_array


    def extract_patches(self, image_array, patch_size=(8,8), multiple_images=False):
        """Returns ((n-p+1)*(m-p+1),p,p) pathces from (n,m) image and (p,p) patch for single image."""

        # validate patch_size
        assert patch_size[0] == patch_size[1], "patch_size is {0} instead of being square".format(patch_size)

        # validate single image array shape
        assert image_array.shape[-1] == image_array.shape[-2], "each image has shape {0} instead of being square.".format(image_array.shape[-2:])

        if multiple_images:
            # expecting shape (number_images, x, y)
            assert image_array.ndim == 3, "image_array.ndim is {0} instead of 3".format(image_array.ndim)

            number_images = image_array.shape[0]
            number_patches = (image_array.shape[1] - patch_size[0] + 1)**2

            # creating empty array to store patches for multiple images
            patches = np.empty((number_images, number_patches, patch_size[0], patch_size[1]), dtype=float)

            # for every image in the array, resize it and populate the empty resized array
            for image_index in range(number_images):
                patches[image_index] = extract_patches_2d(image_array[image_index], patch_size)

        else:
            # expecting shape (x, y)
            assert image_array.ndim == 2, "image_array.ndim is {0} instead of 2".format(image_array.ndim)

            # extract patches for single image
            patches = extract_patches_2d(image_array, patch_size)

        # returning shape (number_images, number_patches, patch_side, patch_side) for multiple_images
        # returning shape (number_patches, patch_side, patch_side) for single image
        return patches


    def get_contrast_normalized_patches(self, patches, multiple_images=False):
        """Returns (n,p,p) normalized_patches from (n,p,p) patches for single image."""

        # From [Zim15], While implementing contrast normalization, we subtract the pixel mean of each patch after extraction
        # and divide by the standard deviation of all it's pixels. Before the division, we add a small value alpha to the pixel variance
        # in order to prevent potential errors due to division by zero.

        if multiple_images:
            # expecting (number_images, number_patches, patch_side, patch_side)
            assert patches.ndim == 4, "patches.ndim is {0} instead of 4".format(patches.ndim)

            # store shapes
            original_patches_shape = patches.shape
            number_images = patches.shape[0]
            number_patches = patches.shape[1]

            # reshape based on [Zim15] i.e. (number_images * number_patches, p**2)
            patches = patches.reshape(number_images * number_patches, -1)

        else:
            # expecting (number_patches, patch_side, patch_side)
            assert patches.ndim == 3, "patches.ndim is {0} instead of 3".format(patches.ndim)

            # store shapes
            original_patches_shape = patches.shape
            number_patches = patches.shape[0]

            # reshape based on [Zim15] i.e. (number_images * number_patches, p**2)
            patches = patches.reshape((number_patches, -1))

        # overwrite normalized patches onto the original patches
        patches = (patches - patches.mean(axis=1)[:, np.newaxis]) / (np.sqrt(patches.var(axis=1))[:, np.newaxis] + 0.01)

        # reshape into original shape
        patches = patches.reshape(original_patches_shape)

        # returning shape (number_images, number_patches, patch_side, patch_side) for multiple_images
        # returning shape (number_patches, patch_side, patch_side) for single image
        return patches


    ## Pipeline - Step 3
    def get_whitened_patches(self, patches, multiple_images=False):
        """Return (n,p,p) whitened_patches from (n,p,p) (normalized) patches for single image."""

        def whiten_image_patches(image_patches):
            # expecting (number_patches, patch_side, patch_side)
            assert image_patches.ndim == 3, "image_patches.ndim is {0} instead of 3".format(image_patches.ndim)

            # validating number_patches > 1
            image_patches_shape = image_patches.shape
            number_patches = image_patches.shape[1]
            assert number_patches > 1, "number_patches is {0} instead of being greater than 1, otherwise whitening is not possible".format(number_patches)

            # flatten individual image_patches such that rows = no. of image_patches, columns = pixel values
            image_patches = image_patches.reshape((image_patches.shape[0], -1))

            # Transpose to get matrix A
            A = image_patches.T

            # sigma = (1/(n-1)) * np.dot(A,A.T), where A = image_patches.T
            sigma = (1.0 / (number_patches - 1)) * np.dot(A, A.T)

            # decompose and reconstruct
            u, s, v = np.linalg.svd(sigma, full_matrices=True)

            # whitening matrix
            epsilon = 0.01
            whitening_matrix = np.dot(u, np.dot((1.0 / (np.sqrt(np.diag(s)) + epsilon)), u.T))

            # Awhite = W.A
            whitened_A = np.dot(whitening_matrix, A)

            # Transpose to get back normalized image_patches
            whitened_image_patches = whitened_A.T

            # reshape whitened image_patches
            whitened_image_patches = whitened_image_patches.reshape(image_patches_shape)

            # from matplotlib import pyplot as plt
            # from matplotlib import cm
            # plt.imshow(sigma, cmap=cm.Greys)
            # plt.show()
            # subset = whitened_image_patches[np.random.rand(whitened_image_patches.shape[0]) > 0.97]
            # print "...........", subset.shape
            # plt.imshow(np.dot(whitened_image_patches.reshape(3249,-1).T, whitened_image_patches.reshape(3249,-1)), cmap=cm.Greys)
            # plt.show()

            return whitened_image_patches

        # Whitenening is done per image i.e. decorrelating all the patches for one image
        if multiple_images:
            assert patches.ndim == 4, "patches.ndim is {0} instead of 4".format(patches.ndim)

            # store shapes
            number_images = patches.shape[0]

            # whiten the pages for every image
            for image_index in range(number_images):
                patches[image_index] = whiten_image_patches(patches[image_index])

        else:
            # expecting (number_patches, patch_side, patch_side)
            assert patches.ndim == 3, "patches.ndim is {0} instead of 3".format(patches.ndim)

            # whiten the patches
            patches = whiten_image_patches(patches)

        # returning shape (number_images, number_patches, patch_side, patch_side) for multiple_images
        # returning shape (number_patches, patch_side, patch_side) for single image
        return patches


    ## Pipeline Combined
    def get_whitened_patches_from_image_array(self, image_array):
        """Returns ((n'-p+1)*(m'-p+1),p,p) whitened_patches from (n,m) image, (n',m') imsize, (p,p) patch."""
        resized_image_array = self.get_resized_image(image_array)
        patches = self.extract_patches(resized_image_array)
        normalized_patches = self.get_contrast_normalized_patches(patches)
        whitened_patches = self.get_whitened_patches(normalized_patches)
        return whitened_patches
