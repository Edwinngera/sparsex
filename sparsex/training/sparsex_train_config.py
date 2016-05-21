config_params = {
    "dataset_path" : "/home/nitish/mas_course_ss2015/assignments/sparsex/datasets/yale_face_b_ext_cropped/CroppedYale_64x64",
    "preprocess_resize" : (64,64),
    "preprocess_patch_size" : (8,8),
    "preprocess_normalization" : True,
    "preprocess_whitening" : True,
    "feature_extraction_library" : "Spams",
    "feature_extraction_dictionary_size" : 10,
    "feature_extraction_output_model_filename" : "/home/nitish/mas_course_ss2015/assignments/sparsex/sparsex/learning/learnt_feature_extraction_model_spams.pkl",
    "feature_extraction_pooling_filter_size" : (19,19),
    "classification_library" : "JoachimsSVM",
    "classification_output_model_filename" : "/home/nitish/mas_course_ss2015/assignments/sparsex/sparsex/learning/learnt_classification_model_joachimssvm.pkl",
    "classification_param_c" : 0.1,
    "classification_param_t" : 1,
    "classification_param_d" : 2
}

