from ..tests.preprocessing_test import test_whitening
from ..customutils.customutils import write_dictionary_to_pickle_file, read_dictionary_from_pickle_file, is_perfect_square, isqrt
from skimage.util.shape import view_as_windows
import spams, os, sys, logging, time
import numpy as np

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class Spams(object):
    
    DEFAULT_MODEL_FILENAME = os.path.realpath(os.path.join(THIS_FILE_PATH,
                             "../tests/data/feature_extraction_model_spams.pkl"))
    DEFAULT_TRAINED_MODEL_FILENAME = os.path.realpath(os.path.join(THIS_FILE_PATH,
                                     "../tests/data/trained_feature_extraction_test_model_spams.pkl"))
    STANDARD_TRAINED_MODEL_FILENAME = os.path.realpath(os.path.join(THIS_FILE_PATH,
                                     "../training/trained_feature_extraction_model_spams.pkl"))
    
    # train dl params
    TRAIN_DL_PARAMS = ['K', 'D', 'lambda1', 'numThread', 'batchsize', 'iter', 'verbose']
    
    # encoding params
    LASSO_ENCODING_PARAMS = ['L', 'lambda1', 'lambda2', 'mode', 'pos', 'ols',' numThreads',
                             'length_path', 'verbose', 'cholesky', 'return_reg_path']
    OMP_ENCODING_PARAMS = ['L', 'eps', 'lambda1', 'return_reg_path', 'numThreads']
    
    # default params
    DEFAULT_MODEL_PARAMS = {'K':10, 'lambda1':0.15, 'numThreads':-1, 'batchsize':400, 'iter':10,
                            'verbose':False, 'return_reg_path':False, 'mode':spams.PENALTY,
                            'encoding_algorithm':'omp'}
    
    def __init__(self, model_filename=None, **kwargs):
        if model_filename == None:
            self.params = Spams.DEFAULT_MODEL_PARAMS
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
                
        # choose the encoding params
        # NOTE : choose encoding_function based on the algorithm chosen
        if self.params['encoding_algorithm'] == 'lasso':
            _encoding_params = Spams.LASSO_ENCODING_PARAMS
            self.encoding_function = spams.lasso
        else:
            _encoding_params = Spams.OMP_ENCODING_PARAMS
            self.encoding_function = spams.omp
        
        # extract the encoding params from the global params
        self.encoding_params = {}
        for encoding_param_name in _encoding_params:
            if encoding_param_name in self.params:
                self.encoding_params[encoding_param_name] = self.params[encoding_param_name]


    def save_model(self, filename):
        write_dictionary_to_pickle_file(filename, self.params)


    def load_model(self, filename):
        self.params = read_dictionary_from_pickle_file(filename)


    def learn_dictionary(self, patches, multiple_images=False):
        """Returns None from (n,p**2) patches for a single image."""
        
        # store original shape
        original_patches_shape = patches.shape
        
        if multiple_images:
            # expecting (number_images, number_patches, patch_side**2)
            assert patches.ndim == 3, "patches.ndim is {0} instead of 3".format(patches.ndim)
            
            # store shapes
            number_images = original_patches_shape[0]
            number_patches = original_patches_shape[1]
            
            # reshape patches, dictionary is learnt on each patch so it does not matter which image it comes from
            # therefore, flatten the array to (number_images * number_patches, patch_side**2)
            patches = patches.reshape(number_images * number_patches, -1)
            
        else:
            # expecting (number_patches, patch_side**2)
            assert patches.ndim == 2, "patches.ndim is {0} instead of 2".format(patches.ndim)
            
        # spams.trainDL expects X to be (p**2,n) with n patches and p**2 features,
        # which is opposite to the convention used in sparsex. Therefore we transpose it.
        X = patches.T

        # spams.trainDL expects arrays to be in fortran order. Rememeber to reconvert it to 'C' order when
        # in the get_dictionary mehtod.
        X = np.asfortranarray(X)
        
        # updating the params so that the next time trainDL uses the already learnt dictionary from params.
        # D is of shape (p**2, k) which is opposite to the sparse shape convention. We will need to transpose D
        # in the get_dictionary method.
        self.params['D'] = spams.trainDL(X, **self.train_params)
        
        # update params
        self._extract_params()
        
        if multiple_images:
            # precautionary reshape back to original shape, since its possible the patches referenced here may be used
            # elsewhere, therefore they will be reshaped due to the manipulations made above in multple images.
            # Although X has its flags changed, patches retains its original flags.
            patches = patches.reshape(original_patches_shape)
        else:
            # no reshaping required
            pass
        
        # return None
        return


    def get_dictionary(self):
        """Returns (k, p**2) dictionary with k dictionary elements and p**2 features each."""
        # transpose D from (p**2, k) to (k, p**2) to adhere to sparsex shape convention.
        # CAUTION!!! array.T seems to be changing order from C to F and vice versa.
        D = self.params['D'].T
        
        # convert D to contiguous array from fortran array.
        return np.ascontiguousarray(D)
        

    def get_sparse_features(self, patches, multiple_images=False):
        """Returns (n,k) encoding from (n,p**2) patches and (k,p**2) internal dictionary for single image."""
        
        # store original shape
        original_patches_shape = patches.shape
        
        if multiple_images:
            # expecting (number_images, number_patches, patch_side**2)
            assert patches.ndim == 3, "patches.ndim is {0} instead of 3".format(patches.ndim)
        
            # store shapes
            number_images = original_patches_shape[0]
            number_patches = original_patches_shape[1]
            
            # reshape patches, sparse encoding is done on each patch and does not depend on image.
            patches = patches.reshape(number_images * number_patches, -1)
        
        else:
            # expecting (number_patches, patch_side**2)
            assert patches.ndim == 2, "patches.ndim is {0} instead of 2".format(patches.ndim)

        # get dictionary_size
        try:
            # spams dictionary is shaped (number_features, dictionary_size)
            dictionary_size = self.params['D'].shape[1]
        except KeyError:
            raise KeyError("It is possible feature extraction dictionary has not yet been learnt for this model. " \
                         + "Train the feature extraction model at least once to prevent this error.")
        except ValueError as e:
            raise ValueError(e.message + "\n" \
                + "Sparsex Note : It is possible the feature extraction dictionary has not yet been learnt for this model. " \
                + "Train the feature extraction model at least once to prevent this error.")
        
        # create empty encoding array
        # shape is transposed (number_images * number_patches, dictionary_size)    
        encoding = np.empty((patches.shape[0], dictionary_size), dtype=float)
        
        logging.debug("get sparse encoding")
        logging.debug("number_images * number_patches (total_patches): {0}".format(patches.shape[0]))
        sys.stdout.flush()
        
        # time keeping
        start_time = time.time()
        percent_time = time.time()
        
        for patch_index in range(patches.shape[0]):
            # # debug code
            # # check if the patches is entirely zero
            # patches_non_zero_count = np.sum(~(patches[patch_index].ravel() == 0))
            # if patches_non_zero_count == 0:
            #     single_encoding = np.zeros(dictionary_size, dtype=float)
            # 
            # else:
            #     # encoding_function returns (dictionary_size, 1) for one sample of (number_features, 1) but is scipy.sparse
            #     # convert sparse matrix to full matrix using toarray()
            #     single_encoding = self.encoding_function(np.asfortranarray(patches[patch_index][:,np.newaxis]), self.params['D'], **self.encoding_params).toarray()
            
            # encoding_function returns (dictionary_size, 1) for one sample of (number_features, 1) but is scipy.sparse
            # convert sparse matrix to full matrix using toarray()
            single_encoding = self.encoding_function(np.asfortranarray(patches[patch_index][:,np.newaxis]), self.params['D'], **self.encoding_params).toarray()
            
            # column matrix is populated into encoding as row matrix by flattnening it (ravel).
            encoding[patch_index, :] = single_encoding.ravel()
            
            if (patch_index * 100) % patches.shape[0] == 0:
                # basic debug to check if encoding is working
                now_time = time.time()
                percent_diff = now_time - percent_time
                time_elapsed = now_time - start_time
                percent_time = now_time
                logging.debug("encoding progress : {0}%, {1} second/percent, {2} seconds elapsed".format((patch_index * 100) // patches.shape[0], percent_diff, time_elapsed))
                encoding_non_zero_count = np.sum(~(encoding[patch_index].ravel() == 0))
                patches_non_zero_count = np.sum(~(patches[patch_index].ravel() == 0))
                logging.debug("encoding non-zero count : {0} / {1}, patches non-zero count  : {2} / {3}".format(encoding_non_zero_count, encoding[patch_index].shape[0], patches_non_zero_count, patches[patch_index].shape[0]))
                sys.stdout.flush()
        
        # final time keeping
        end_time = time.time()
        percent_diff = end_time - percent_time
        time_elapsed = end_time - start_time
        
        logging.debug("encoding progress : 100%, {0} second/percent, {1} seconds elapsed".format(percent_diff, time_elapsed))
        sys.stdout.flush()
        
        if multiple_images:
            # precautionary reshape back to original shape, since its possible the patches referenced here may be used
            # elsewhere, therefore they will be reshaped due to the manipulations made above in multple images.
            # Although X has its flags changed, patches retains its original flags.
            patches = patches.reshape(original_patches_shape)
            
            # reshape encoding to (number_images, number_patches, k)
            encoding = encoding.reshape(number_images, number_patches, -1)
            
        else:
            # no reshaping required
            pass
            
        # convert encoding to contiguous array from fortran array
        # returning shape (number_images, number_patches, k) for multiple images
        # returning shape (number_patches, k) for single image
        return np.ascontiguousarray(encoding)
        
        
        # # spams excepts X to be (p**2,n) with n patches and p**2 features,
        # # which is opposite to the convention used in sparsex. Therefore we transpose it.
        # X = patches.T
        # 
        # # spams.trainDL expects arrays to be in fortran order. Rememeber to reconvert it to 'C' order when
        # # in the get_dictionary mehtod.
        # X = np.asfortranarray(X)
        # 
        # try:
        #     # get encoding, which is a sparse matrix
        #     encoding = self.encoding_function(X, self.params['D'], **self.encoding_params)
        #     
        #     # convert the sparse matrix to a full matrix
        #     encoding = encoding.toarray()
        #     
        #     # tranpose encoding (k,n) to (n,k) to adhere to sparsex shape convention.
        #     encoding = encoding.T
        #     
        # except KeyError:
        #     raise KeyError("It is possible feature extraction dictionary has not yet been learnt for this model. " \
        #                  + "Train the feature extraction model at least once to prevent this error.")
        # except ValueError as e:
        #     raise ValueError(e.message + "\n" \
        #         + "Sparsex Note : It is possible the feature extraction dictionary has not yet been learnt for this model. " \
        #         + "Train the feature extraction model at least once to prevent this error.")
        #     
        # finally:
        #     if multiple_images:
        #         # precautionary reshape back to original shape, since its possible the patches referenced here may be used
        #         # elsewhere, therefore they will be reshaped due to the manipulations made above in multple images.
        #         # Although X has its flags changed, patches retains its original flags.
        #         patches = patches.reshape(original_patches_shape)
        #         
        #         # reshape encoding to (number_images, number_patches, k)
        #         encoding = encoding.reshape(number_images, number_patches, -1)
        #         
        #     else:
        #         # no reshaping required
        #         pass
        #     
        #     # convert encoding to contiguous array from fortran array
        #     # returning shape (number_images, number_patches, k) for multiple images
        #     # returning shape (number_patches, k) for single image
        #     return np.ascontiguousarray(encoding)


    def get_sign_split_features(self, encoding, multiple_images=False):
        """Returns (n,2k) features from (n,k) encoding for single image."""
        
        original_encoding_shape = encoding.shape
        
        if multiple_images:
            # expecting (number_images, number_patches, k)
            assert encoding.ndim == 3, "encoding.ndim is {0} instead of 3".format(encoding.ndim)
        
            # store shapes
            number_images = original_encoding_shape[0]
            number_patches = original_encoding_shape[1]
            number_samples = number_images * number_patches
            number_features = original_encoding_shape[2]
            
            # reshape encoding, sign split is done on each patch and does not depend on image.
            encoding = encoding.reshape(number_images * number_patches, -1)
        
        else:
            # expecting (number_patches, k)
            assert encoding.ndim == 2, "encoding.ndim is {0} instead of 2".format(encoding.ndim)
            
            # store shapes
            number_samples = original_encoding_shape[0]
            number_features = original_encoding_shape[1]
            
        sign_split_features = np.empty((number_samples, 2 * number_features))
        sign_split_features[:, :number_features] = np.maximum(encoding, 0)
        sign_split_features[:, number_features:] = -np.minimum(encoding, 0)
        
        if multiple_images:
            # precautionary reshape of encoding to original shape in case it may be used elsewhere
            encoding = encoding.reshape(original_encoding_shape)
            
            # reshape sign_split_features to (number_images, number_patches, 2k)
            sign_split_features = sign_split_features.reshape(number_images, number_patches, -1)
        
        else:
            # no reshaping required
            pass
            
        # returning shape (number_images, number_patches, 2*k) for multiple images.
        # returning shape (number_patches, 2*k) for single image.
        return sign_split_features


    def get_pooled_features(self, encoding, filter_size, multiple_images=False):
        """Returns (n/s**2,f) pooled features from (n,f) features and (s,s) filter size for single image."""
        
        def pool_features(input_feature_map):
            # expecting shape (sqrt_number_patches, sqrt_number_patches, number_features)
            assert input_feature_map.ndim == 3, "input_feature_map.ndim is {0} instead of 3".format(input_feature_map.ndim)
        
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
            pooled_feature_map = input_feature_map_windows[np.arange(input_feature_map_windows.shape[0]), max_norm_indexes]

            # return pooled feature map
            return pooled_feature_map
        
        
        # validate filter_size
        assert filter_size[0] == filter_size[1], "filter_size is {0} instead of being square".format(filter_size)
        
        # store shape
        original_encoding_shape = encoding.shape
        
        if multiple_images:
            # expecting (number_images, number_patches, number_features)
            assert encoding.ndim == 3, "encoding.ndim is {0} instead of 3".format(encoding.ndim)
        
            # store shapes
            number_images, number_patches, number_features = original_encoding_shape
            
            # validate number_patches to be perfect square so that it can be reshaped into a square map/window
            assert is_perfect_square(number_patches), "number_patches is {0} and is not a perfect_square".format(number_patches)

            # calculate sqrt of number_patches
            sqrt_number_patches = isqrt(number_patches)
            
            # reshape encoding to (number_images, sqrt_number_patches, sqrt_number_patches, number_features)
            encoding_map = encoding.reshape(number_images, sqrt_number_patches, sqrt_number_patches, number_features)
            
            # create empty pooled features to populate all the feature maps for each image
            # shape (number_images, number_patches/filter_side**2, number_features)
            pooled_features = np.empty((number_images, number_patches / filter_size[0]**2, number_features), dtype=float)

            # for each image, get pooled features from the encoding/feature_map
            for image_index in range(number_images):
                pooled_features[image_index] = pool_features(encoding_map[image_index])
            
        else:
            # expecting (number_patches, number_features)
            assert encoding.ndim == 2, "encoding.ndim is {0} instead of 2".format(encoding.ndim)
            
            number_patches, number_features = original_encoding_shape
            
            # validate number_patches to be perfect square so that it can be reshaped into a square map/window
            assert is_perfect_square(number_patches), "number_patches is {0} and is not a perfect_square".format(number_patches)

            # calculate sqrt of number_patches
            sqrt_number_patches = isqrt(number_patches)
            
            # reshape encoding to (sqrt_number_patches, sqrt_number_patches, number_features)
            encoding_map = encoding.reshape(sqrt_number_patches, sqrt_number_patches, number_features)
            
            # get pooled features
            pooled_features = pool_features(encoding_map)
        
        # returning shape (number_images, number_patches/filter_side**2, number_features) for multiple images
        # returning shape (number_patches/filter_side**2, number_features) for single image
        return pooled_features        
        
        
    def pipeline(self, patches, sign_split=True, pooling=True, pooling_size=(3,3), reshape_2d=False, multiple_images=False):
        """Returns (n/s**2,2k) feature map from (n,p**2), sign_split, pooling, (s,s) pooling_size, (k,p**2) internal dictionary for single image."""
        features = self.get_sparse_features(patches, multiple_images)
        
        if sign_split:
            features = self.get_sign_split_features(features, multiple_images)
        
        if pooling:
            features = self.get_pooled_features(features, pooling_size, multiple_images)
        
        if reshape_2d and multiple_images:
            # reshape (number_images, number_features)
            features = features.reshape(features.shape[0], -1)
        
        elif reshape2d and not multiple_images:
            # reshape (1, number_features)
            features = features.reshape(1, -1)
        
        return features



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
    pooled_features = sparse_coding.get_pooled_features(sign_split_features, filter_size=(19,19))

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
    