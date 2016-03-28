from sklearn.decomposition import DictionaryLearning
from skimage.util.shape import view_as_windows
from ..tests.preprocessing_test import test_whitening
from ..customutils import customutils
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class SparseCoding:

    DEFAULT_MODEL_PARAMS = {
        'n_components' : 10,
        'n_features' : 64,
        'max_iter' : 5,
        'random_state' : 1111
    }

    def __init__(self, model_params_file=None, model_weights_file=None):
        # load parameters
        self.model_params_file = model_params_file
        if self.model_params_file is not None:
            self.load_model_params(self.model_params_file)
        else:
            # default model params
            self.n_components = SparseCoding.DEFAULT_MODEL_PARAMS['n_components']
            self.n_features = SparseCoding.DEFAULT_MODEL_PARAMS['n_features']
            self.max_iter = SparseCoding.DEFAULT_MODEL_PARAMS['max_iter']
            self.random_state = SparseCoding.DEFAULT_MODEL_PARAMS['random_state']

        # load weights
        self.model_weights_file = model_weights_file
        if self.model_weights_file is not None:
            self.load_model_weights(self.model_weights_file)
            # setting the components_ (a.k.a dictionary atoms) of past model as dict_init of this model
            if self.components_ is not None:
                self.dict_init = self.components_
        else:
            # default weights init
            self.dict_init = None
            self.code_init = None
            self.components_ = None

        self.DL_obj = DictionaryLearning(n_components=self.n_components,
                                       alpha=1,
                                       max_iter=self.max_iter,
                                       tol=1e-08,
                                       fit_algorithm='lars',
                                       transform_algorithm='omp',
                                       transform_n_nonzero_coefs=None,
                                       transform_alpha=None,
                                       n_jobs=1,
                                       code_init=self.code_init,
                                       dict_init=self.dict_init,
                                       verbose=False,
                                       split_sign=False,
                                       random_state=self.random_state)

        # one time fit call to prevent NotFittedError - DIRTY HACK
        self.DL_obj.fit(np.random.rand(1,self.n_features))


    def save_model_params(self, model_params_file=None):
        # generate model params filename if not provided
        if model_params_file is None:
            string_timestamp = customutils.get_current_string_timestamp()
            model_params_file = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/model_params_{0}.json".format(string_timestamp)))

        # generate model_params dictionary
        model_params = self.DL_obj.get_params(deep=True)

        # delete non Json serializable keys (or) possibly very big keys (or) keys that come under model weights
        for unwanted_key in ['code_init','dict_init']:
            if unwanted_key in model_params:
                del model_params[unwanted_key]

        # write dictionary to json file
        print "Saving model_params : ", model_params
        customutils.write_dictionary_to_json_file(model_params_file, model_params)
        print "Model_params saved to file :", model_params_file


    def save_model_weights(self, model_weights_file=None):
        # generate model weights filename if not provided
        if model_weights_file is None:
            string_timestamp = customutils.get_current_string_timestamp()
            model_weights_file = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/model_weights_{0}.h5".format(string_timestamp)))
        
        # generate model weights
        model_weights = {}
        if self.code_init is not None: model_weights['code_init'] = self.code_init
        if self.dict_init is not None: model_weights['dict_init'] = self.dict_init
        if hasattr(self.DL_obj, 'components_'): model_weights['components_'] = self.DL_obj.components_

        # write weights to hdf5 file
        print "Saving model_weights :", model_weights.keys()
        customutils.write_dictionary_to_h5_file(model_weights_file, model_weights)
        print "Model weights saved to file :", model_weights_file


    def load_model_params(self, model_params_file):
        model_params = customutils.read_dictionary_from_json_file(model_params_file)
        for wanted_key in SparseCoding.DEFAULT_MODEL_PARAMS.keys():
            if wanted_key in model_params:
                setattr(self, wanted_key, model_params[wanted_key])
                print "Loaded model param :", wanted_key, model_params[wanted_key]
            else:
                setattr(self, wanted_key, SparseCoding.DEFAULT_MODEL_PARAMS[wanted_key])
                print "Loaded model param :", wanted_key, "DEFAULT:", SparseCoding.DEFAULT_MODEL_PARAMS[wanted_key]


    def load_model_weights(self, model_weights_file):
        model_weights = customutils.read_dictionary_from_h5_file(model_weights_file)
        for wanted_key in ['code_init', 'dict_init', 'components_']:
            if wanted_key in model_weights:
                setattr(self, wanted_key, model_weights[wanted_key])
                print "Loaded model weight :", wanted_key, model_weights[wanted_key].shape
            else:
                setattr(self, wanted_key, None)
                print "Loaded model weight :", wanted_key, "None"   


    def learn_dictionary(self, whitened_patches):
        # assert correct dimensionality of input data
        if whitened_patches.ndim == 3:
            whitened_patches = whitened_patches.reshape((whitened_patches.shape[0], -1))
        assert whitened_patches.ndim == 2, \
        "Whitened patches ndim is %d instead of 2" %whitened_patches.ndim

        # learn dictionary
        self.DL_obj.fit(whitened_patches)


    def get_sparse_features(self, whitened_patches):
        # assert correct dimensionality of input data
        if whitened_patches.ndim == 3:
            whitened_patches = whitened_patches.reshape((whitened_patches.shape[0], -1))
        assert whitened_patches.ndim == 2, "Whitened patches ndim is %d instead of 2" %whitened_patches.ndim
        sparse_code = self.DL_obj.transform(whitened_patches)
        return sparse_code


    def get_sign_split_features(self, sparse_features):
        n_samples, n_components = sparse_features.shape
        sign_split_features = np.empty((n_samples, 2 * n_components))
        sign_split_features[:, :n_components] = np.maximum(sparse_features, 0)
        sign_split_features[:, n_components:] = -np.minimum(sparse_features, 0)
        return sign_split_features


    def get_pooled_features(self, input_feature_map, filter_size=(19,19)):
        # assuming square filters and images
        filter_side = filter_size[0]

        # reshaping incoming features from 2d to 3d i.e. (3249,20) to (57,57,20)
        input_feature_map_shape = input_feature_map.shape
        if input_feature_map.ndim == 2:
            input_feature_map_side = int(np.sqrt(input_feature_map.shape[0]))
            input_feature_map = input_feature_map.reshape((input_feature_map_side, input_feature_map_side, input_feature_map_shape[-1]))
        assert input_feature_map.ndim == 3, "Input features dimension is %d instead of 3" %input_feature_map.ndim

        # calculate norms = (57,57,20) = (57,57)
        input_feature_map_norms = np.linalg.norm(input_feature_map, ord=2, axis=-1)
        assert input_feature_map_norms.ndim == 2, "Input feature norms dimension is %d instead of 2" %input_feature_map_norms.ndim

        # extract pooling windows with stride = side, n_windows = (3,3) ndim_window = (19,19) ndim_windows = (3,3,19,19)
        input_feature_map_norms_pooling_windows = view_as_windows(input_feature_map_norms,
                                                  window_shape=filter_size,
                                                  step=filter_size[0])
        assert input_feature_map_norms_pooling_windows.ndim == 4, "Pooling windows dimension is %d instead of 4" %input_feature_map_norms_pooling_windows.ndim

        # choose maximums from the from windows such that (3,3,19,19) = (3,3)
        pooled_feature_map = np.amax(input_feature_map_norms_pooling_windows, axis=(-2,-1))
        assert pooled_feature_map.ndim == 2, "Pooled features dimension is %d instead of 2" %pooled_feature_map.ndim

        # return pooled feature map
        return pooled_feature_map


    # Combined Pipeline
    def get_pooled_features_from_whitened_patches(self, whitened_patches):
        sparse_features = self.get_sparse_features(whitened_patches)
        sign_split_features = self.get_sign_split_features(sparse_features)
        pooled_features = self.get_pooled_features(sign_split_features)
        return pooled_features


if __name__ == "__main__":
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    
    # get whitened patches
    whitened_patches = test_whitening(image_filename, False, False)

    # create sparse coding object
    sparse_coding = SparseCoding()

    # learn dictionary on whitened patches
    sparse_coding.learn_dictionary(whitened_patches[:100])
    
    # get sparse code
    sparse_features = sparse_coding.get_sparse_features(whitened_patches)

    # get feature sign split
    sign_split_features = sparse_coding.get_sign_split_features(sparse_features)

    # get pooled features
    pooled_features = sparse_coding.get_pooled_features(input_feature_map=sign_split_features)

    print "Dictionary Shape :"
    print sparse_coding.DL_obj.components_.shape

    print "sparse features shape :"
    print sparse_features.shape

    print "sign split features shape :"
    print sign_split_features.shape

    print "pooled features shape :"
    print pooled_features.shape

    # get pooled features directly from whitened patches using feature extraction pipeline
    pooled_features_from_whitened_patches = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches)

    print "pooled features from whitened patches shape :"
    print pooled_features_from_whitened_patches.shape
