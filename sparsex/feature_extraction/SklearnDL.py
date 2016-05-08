from sklearn.decomposition import DictionaryLearning
from skimage.util.shape import view_as_windows
from sklearn.externals import joblib
from sklearn.utils.validation import NotFittedError
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class SklearnDL(object):
    
    DEFAULT_MODEL_PARAMS = {
        'n_components' : 10,
        'n_features' : 64,
        'max_iter' : 5,
        'random_state' : 1,
        'dict_init' : None,
        'code_init' : None
    }

    def __init__(self, model_filename=None, **kwargs):
        if model_filename is not None:
            self.load_model(model_filename)
        else:
            # default model params
            self.n_components = SklearnDL.DEFAULT_MODEL_PARAMS['n_components']
            self.n_features = SklearnDL.DEFAULT_MODEL_PARAMS['n_features']
            self.max_iter = SklearnDL.DEFAULT_MODEL_PARAMS['max_iter']
            self.random_state = SklearnDL.DEFAULT_MODEL_PARAMS['random_state']
            self.dict_init = SklearnDL.DEFAULT_MODEL_PARAMS['dict_init']
            self.code_init = SklearnDL.DEFAULT_MODEL_PARAMS['code_init']

            # initialize Dictionary Learning object with default params and weights
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


    def save_model(self, filename):
        # save DL object to file, compress is also to prevent multiple model files.
        joblib.dump(self.DL_obj, filename, compress=3)


    def load_model(self, filename):
        # load DL Object from file
        self.DL_obj = joblib.load(filename)

        # set certain model params as class attributes. Get values from DL Obj.get_params() or use default values.
        DL_params = self.DL_obj.get_params()
        for param in SklearnDL.DEFAULT_MODEL_PARAMS:
            if param in DL_params:
                setattr(self, param, DL_params[param])
            else:
                setattr(self, param, SklearnDL.DEFAULT_MODEL_PARAMS[param])


    def learn_dictionary(self, whitened_patches):
        # assert correct dimensionality of input data
        if whitened_patches.ndim == 3:
            whitened_patches = whitened_patches.reshape((whitened_patches.shape[0], -1))
        assert whitened_patches.ndim == 2, "Whitened patches ndim is %d instead of 2" %whitened_patches.ndim

        # learn dictionary
        self.DL_obj.fit(whitened_patches)


    def get_dictionary(self):
        try:
            return self.DL_obj.components_
        except AttributeError:
            raise AttributeError("Feature extraction dictionary has not yet been learnt for this model. " \
                                 + "Train the feature extraction model at least once to prevent this error.")


    def get_sparse_features(self, whitened_patches):
        # assert correct dimensionality of input data
        if whitened_patches.ndim == 3:
            whitened_patches = whitened_patches.reshape((whitened_patches.shape[0], -1))
        assert whitened_patches.ndim == 2, "Whitened patches ndim is %d instead of 2" %whitened_patches.ndim
        try:
            sparse_code = self.DL_obj.transform(whitened_patches)
        except NotFittedError:
            raise NotFittedError("Feature extraction dictionary has not yet been learnt for this model, " \
                                 + "therefore Sparse Codes cannot be extracted. Train the feature extraction model " \
                                 + "at least once to prevent this error.")
        return sparse_code


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


    # Combined Pipeline
    def get_pooled_features_from_whitened_patches(self, whitened_patches, filter_size):
        sparse_features = self.get_sparse_features(whitened_patches)
        sign_split_features = self.get_sign_split_features(sparse_features)
        pooled_features = self.get_pooled_features(sign_split_features, filter_size)
        return pooled_features
