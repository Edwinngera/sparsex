import spams
from ..feature_extraction.feature_extraction import SparseCoding
from ..classification.classification import Classifier
from ..customutils.customutils import read_dictionary_from_pickle_file


def dataset_extraction_function():
    pass


config_params = {
    "dataset_path" : "/home/nitish/mas_course_ss2015/assignments/sparsex/datasets/yale_face_b_ext_cropped/CroppedYale",
    "dataset_extraction_function":None,
    "preprocess_resize" : (64,64),
    "preprocess_patch_size" : (8,8),
    "preprocess_normalization" : True,
    "preprocess_whitening" : True,
    "feature_extraction_library" : SparseCoding.SPAMS,
    "feature_extraction_params": {
        'encoding_algorithm':'omp', 'L':5, # omp (default None), lasso (default -1)
        'K':10,
        'iter':100,
        'batchsize':1,
        'lambda1':0.11,
        'numThreads':-1,
        'verbose':False, 
        'return_reg_path':False, 
        'mode':spams.PENALTY,
        'eps':1.0
    },
    "feature_extraction_output_model_filename" : "/home/nitish/mas_course_ss2015/assignments/sparsex/sparsex/training/trained_feature_extraction_model_spams.pkl",
    "feature_extraction_sign_split" :True,
    "feature_extraction_pooling" : True,
    "feature_extraction_pooling_filter_size" : (19,19),
    "classification_library" : Classifier.JOACHIMS_SVM,
    "classification_params" : {
        "c":0.1,
        "t":0,
        "d":0,
        "g":0.1
    },
    "classification_training_repeats":1,
    "classification_output_model_filename" : "/home/nitish/mas_course_ss2015/assignments/sparsex/sparsex/training/trained_classification_model_joachimssvm.pkl",
    "random_seed":2902,
    "validation":True,
    "validation_split":0.3,
    "cross_validation":False,
    "cross_validation_folds":3
}

