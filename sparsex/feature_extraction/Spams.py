from ..tests.preprocessing_test import test_whitening
from ..customutils.customutils import write_dictionary_to_pickle_file, read_dictionary_from_pickle_file
import spams
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class Spams(object):
    
    def __init__(self, model_filename=None, **kwargs):
        if model_filename == None:
            self.params = {'K':100, 'lambda1':0.15, 'numThreads':-1, 'batchsize':400, 'iter':10, 'verbose' : False}
            self.params.update(kwargs)
        else:
            self.load_model(model_filename)
            self.params.update(kwargs)


    def save_model(self, filename):
        write_dictionary_to_pickle_file(filename, self.params)


    def load_model(self, filename):
        self.params = read_dictionary_from_pickle_file(filename)


    def learn_dictionary(self, whitened_patches):
        # flattening whitened_patches to 2 dimensions
        if whitened_patches.ndim == 3:
            whitened_patches = whitened_patches.reshape((whitened_patches.shape[0], -1))
        assert whitened_patches.ndim == 2, "Whitened patches ndim is %d instead of 2" %whitened_patches.ndim
            
        # spams.trainDL excepts X to be (p**2,n) with n patches and p**2 features,
        # which is opposite to the convention used in sparsex. Therefore we transpose it.
        X = whitened_patches.T

        # spams.trainDL expects arrays to be in fortran order. Rememeber to reconvert it to 'C' order when
        # in the get_dictionary mehtod.
        X = np.asfortranarray(X)
        
        # updating the params so that the next time trainDL uses the already learnt dictionary from params.
        # D is of shape (p**2, k) which is opposite to the sparse shape convention. We will need to transpose D
        # in the get_dictionary method.
        self.params['D'] = spams.trainDL(X, **self.params)


    def get_dictionary(self):
        # transpose D from (p**2, k) to (k, p**2) to adhere to sparsex shape convention.
        # CAUTION!!! array.T seems to be changing order from C to F and vice versa.
        D = self.params['D'].T
        
        # convert D to contiguous array from fortran array.
        return np.ascontiguousarray(D)
        

    def get_sparse_features(self, whitened_patches):
        raise NotImplementedError

    def get_sign_split_features(self, sparse_features):
        raise NotImplementedError

    def get_pooled_features(self, input_feature_map, filter_size):
        raise NotImplementedError

    def get_pooled_features_from_whitened_patches(self, whitened_patches, filter_size):
        raise NotImplementedError



if __name__ == "__main__":
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    
    # get whitened patches
    whitened_patches = test_whitening(image_filename, False, False)

    # create sparse coding object
    sparse_coding = Spams()

    # learn dictionary on whitened patches
    print "learn dictionary"
    sparse_coding.learn_dictionary(whitened_patches) # making sure the model has a trained dictionary
    sparse_coding.learn_dictionary(whitened_patches) # making sure the model has a trained dictionary
    
    print "get dictionary"
    D = sparse_coding.get_dictionary() # making sure the model has a trained dictionary
    
    print "dictionary shape\n", D.shape
    print "dictionary F order\n", D.flags['F_CONTIGUOUS']
    print "dictionary C order\n", D.flags['C_CONTIGUOUS']
    
    print "saving model"
    model_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/feature_extraction_model_spams.pkl"))
    sparse_coding.save_model(model_filename)
    
    print "reloading model"
    model_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/feature_extraction_model_spams.pkl"))
    sparse_coding = Spams(model_filename=model_filename)

    print "dictionary Shape :"
    print sparse_coding.get_dictionary().shape
    