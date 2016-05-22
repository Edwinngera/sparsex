import spams
from ..feature_extraction.feature_extraction import SparseCoding
from ..classification.classification import Classifier

config_params = {
    "dataset_path" : "/home/nitish/mas_course_ss2015/assignments/sparsex/datasets/yale_face_b_ext_cropped/CroppedYale_small",
    "preprocess_resize" : (64,64),
    "preprocess_patch_size" : (8,8),
    "preprocess_normalization" : True,
    "preprocess_whitening" : True,
    "feature_extraction_library" : SparseCoding.SPAMS,
    "feature_extraction_params": {
        'K':10,
        'lambda1':0.15,
        'numThreads':-1,
        'batchsize':400,
        'iter':10,
        'verbose':False, 
        'return_reg_path':False, 
        'mode':spams.PENALTY
    },
    "feature_extraction_output_model_filename" : "/home/nitish/mas_course_ss2015/assignments/sparsex/sparsex/training/trained_feature_extraction_model_spams.pkl",
    "feature_extraction_sign_split" :True,
    "feature_extraction_pooling" : True,
    "feature_extraction_pooling_filter_size" : (3,3),
    "classification_library" : Classifier.JOACHIMS_SVM,
    "classification_params" : {
        "c":0.1,
        "t":1,
        "d":2
    },
    "classification_output_model_filename" : "/home/nitish/mas_course_ss2015/assignments/sparsex/sparsex/training/trained_classification_model_joachimssvm.pkl",
}

