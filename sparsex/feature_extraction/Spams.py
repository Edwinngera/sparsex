from ..tests.preprocessing_test import test_whitening
from ..customutils.customutils import write_dictionary_to_pickle_file, read_dictionary_from_pickle_file
from skimage.util.shape import view_as_windows
import spams
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class Spams(object):
    
    TRAIN_DL_PARAMS = ['K','D','lambda1','numThread','batchsize','iter','verbose']
    ENCODING_PARAMS = ['L','lambda1','lambda2','mode','pos','ols','numThreads','length_path','verbose','cholesky','return_reg_path']
    
    def __init__(self, model_filename=None, **kwargs):
        if model_filename == None:
            self.params = {'K':100,'lambda1':0.15,'numThreads':-1,'batchsize':400,'iter':10,'verbose':False,
            'return_reg_path':False,'mode':spams.PENALTY}
            self.params.update(kwargs)
            self._extract_params()
        else:
            self.load_model(model_filename)
            self.params.update(kwargs)
            self._extract_params()


    def _extract_params(self):
        # extract train params from global params
        self.train_params = {}
        for train_param_name in Spams.TRAIN_DL_PARAMS:
            if train_param_name in self.params:
                self.train_params[train_param_name] = self.params[train_param_name]
                
        # extract decomposition params from global params
        self.encoding_params = {}
        for encoding_param_name in Spams.ENCODING_PARAMS:
            if encoding_param_name in self.params:
                self.encoding_params[encoding_param_name] = self.params[encoding_param_name]


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
        self.params['D'] = spams.trainDL(X, **self.train_params)
        
        # update params
        self._extract_params()


    def get_dictionary(self):
        # transpose D from (p**2, k) to (k, p**2) to adhere to sparsex shape convention.
        # CAUTION!!! array.T seems to be changing order from C to F and vice versa.
        D = self.params['D'].T
        
        # convert D to contiguous array from fortran array.
        return np.ascontiguousarray(D)
        

    def get_sparse_features(self, whitened_patches):
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
        
        try:
            # get encoding, which is a sparse matrix
            encoding = spams.lasso(X, self.params['D'], **self.encoding_params)
            
            # convert the sparse matrix to a full matrix
            encoding = encoding.toarray()
            
            # tranpose encoding (k,n) to (n,k) to adhere to sparsex shape convention.
            encoding = encoding.T
            
            # convert encoding to contiguous array from fortran array
            return np.ascontiguousarray(encoding)
            
        except KeyError:
            raise KeyError("It is possible feature extraction dictionary has not yet been learnt for this model. " \
                         + "Train the feature extraction model at least once to prevent this error.")
        except ValueError as e:
            raise ValueError(e.message + "\n" \
                + "Sparsex Note : It is possible the feature extraction dictionary has not yet been learnt for this model. " \
                + "Train the feature extraction model at least once to prevent this error.")


    def get_sign_split_features(self, sparse_features):
        n_samples, n_components = sparse_features.shape
        sign_split_features = np.empty((n_samples, 2 * n_components))
        sign_split_features[:, :n_components] = np.maximum(sparse_features, 0)
        sign_split_features[:, n_components:] = -np.minimum(sparse_features, 0)
        return sign_split_features


    def get_pooled_features(self, input_feature_map, filter_size):
        # assuming square filters and images
        filter_side = filter_size[0]

        # reshaping incoming features from 2d to 3d i.e. (3249,20) to (57,57,20)
        input_feature_map_shape = input_feature_map.shape
        if input_feature_map.ndim == 2:
            input_feature_map_side = int(np.sqrt(input_feature_map.shape[0]))
            input_feature_map = input_feature_map.reshape((input_feature_map_side, input_feature_map_side, input_feature_map_shape[-1]))
        assert input_feature_map.ndim == 3, "Input features dimension is %d instead of 3" %input_feature_map.ndim

        # get windows (57,57,20) to (3,3,1,19,19,20)
        input_feature_map_windows = view_as_windows(input_feature_map,
                                                    window_shape=(filter_size[0], filter_size[1], input_feature_map.shape[-1]),
                                                    step=filter_size[0])

        # reshape windows (3,3,1,19,19,20) to (3**2, 19**2, 20) == (9, 361, 20)
        input_feature_map_windows = input_feature_map_windows.reshape((input_feature_map_windows.shape[0]**2,
                                                                       filter_size[0]**2,
                                                                       input_feature_map.shape[-1]))

        # calculate norms (9, 361, 20) to (9,361)
        input_feature_map_window_norms = np.linalg.norm(input_feature_map_windows, ord=2, axis=-1)

        # calculate indexes of max norms per window (9,361) to (9,1). One max index per window.
        max_norm_indexes = np.argmax(input_feature_map_window_norms, axis=-1)

        # max pooled features are the features that have max norm indexes (9, 361, 20) to (9,20). One max index per window.
        pooled_features = input_feature_map_windows[np.arange(input_feature_map_windows.shape[0]), max_norm_indexes]

        # return pooled feature map
        return pooled_features
        
        
    def get_pooled_features_from_whitened_patches(self, whitened_patches, filter_size):
        sparse_features = self.get_sparse_features(whitened_patches)
        sign_split_features = self.get_sign_split_features(sparse_features)
        pooled_features = self.get_pooled_features(sign_split_features, filter_size)
        return pooled_features



if __name__ == "__main__":
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    
    # get whitened patches
    whitened_patches = test_whitening(image_filename, False, False)

    # create sparse coding object
    sparse_coding = Spams()

    # # get sparse code without learning dictionary (It will throw an error. Comment out when not needed.)
    # print "sparse features (before dictionary learning)"
    # sparse_features = sparse_coding.get_sparse_features(whitened_patches)
    # print "sparse_features shape\n", sparse_features.shape
    # print "sparse_features F order\n", sparse_features.flags['F_CONTIGUOUS']
    # print "sparse_features C order\n", sparse_features.flags['C_CONTIGUOUS']
    
    # learn dictionary on whitened patches
    print "learn dictionary"
    sparse_coding.learn_dictionary(whitened_patches) # making sure the model has a trained dictionary
    sparse_coding.learn_dictionary(whitened_patches) # making sure the model has a trained dictionary
    
    # get sparse code
    sparse_features = sparse_coding.get_sparse_features(whitened_patches)

    # get feature sign split
    sign_split_features = sparse_coding.get_sign_split_features(sparse_features)

    # get pooled features
    pooled_features = sparse_coding.get_pooled_features(input_feature_map=sign_split_features, filter_size=(19,19))

    print "dictionary Shape :"
    print sparse_coding.get_dictionary().shape

    print "sparse features shape :"
    print sparse_features.shape

    print "sign split features shape :"
    print sign_split_features.shape

    print "pooled features shape :"
    print pooled_features.shape
    
    print "saving model"
    model_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/feature_extraction_model_spams.pkl"))
    sparse_coding.save_model(model_filename)
    
    print "reloading model"
    model_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/feature_extraction_model_spams.pkl"))
    sparse_coding = Spams(model_filename=model_filename)

    print "dictionary Shape :"
    print sparse_coding.get_dictionary().shape

    print "sparse features shape"
    print sparse_coding.get_sparse_features(whitened_patches).shape
    
    print "pooled features"
    print sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches, filter_size=(19,19)).shape
    