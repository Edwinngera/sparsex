import spams
from ..feature_extraction.feature_extraction import SparseCoding
from ..classification.classification import Classifier

config_params = {
    "dataset_path" : "/home/nitish/mas_course_ss2015/assignments/sparsex/datasets/yale_face_b_ext_cropped/CroppedYale",
    "preprocess_resize" : (67,67),
    "preprocess_patch_size" : (8,8),
    "preprocess_normalization" : True,
    "preprocess_whitening" : True,
    "feature_extraction_library" : SparseCoding.SPAMS,
    "feature_extraction_params": {
        'encoding_algorithm':'omp',
        'K':10,
        'L':5,
        'iter':10,
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
    "feature_extraction_pooling_filter_size" : (12,12),
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

