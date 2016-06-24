from ..preprocessing.preprocessing import Preprocessing
from ..feature_extraction.feature_extraction import SparseCoding
from ..classification.classification import Classifier
from ..customutils.customutils import get_image_from_file, save_image, write_dictionary_to_pickle_file
from scipy.misc import imresize
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
import sys, argparse, os, logging, imghdr
import numpy as np
import sparsex_train_config
# from matplotlib import pyplot as plt
# from matplotlib import cm
from scipy.misc import imsave

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def get_config_params():
    config_params = sparsex_train_config.config_params
    config_params_print = "\n".join(["config-param: {0} : {1}".format(param,value) for param, value in config_params.iteritems()])
    logging.debug("config_params: \n{0}".format(config_params_print))
    return sparsex_train_config.config_params


# dataset_dict = {class_id: {class_name:str, class_path:str, image_paths:list(str)}}
def get_dataset_dictionary(dataset_path):
    logging.info("generating dataset dictionary")
    # get full path of the dataset
    dataset_path = os.path.realpath(dataset_path)
    
    # get list of directories in dataset path which will become the class names
    # sorting is also to make sure that the order is always the same during different iterations
    dataset_dirs = sorted([d for d in os.listdir(dataset_path) 
                          if not os.path.isfile(os.path.join(dataset_path, d))])
    
    # create dict
    # dataset_dict = {
    #     class_id: { # alphabetical order of class_names
    #         class_name: string,
    #         class_path: string,
    #         image_paths: list[string] # alphabetical order of image_path names
    #     },
    #     number_classes = int,
    #     number_images = int,
    #     classes_list = list[int] # basically Y, [1,1,1,2,2,2,3,3,3]
    # }
    dataset_dict = {}
    number_classes = len(dataset_dirs)
    number_images = 0
    classes_list = []
    
    # populate the dict
    for class_id in range(number_classes):
        # class_name is the directory name
        class_name = dataset_dirs[class_id]
        
        # class_path is the full path of the directory
        class_path = os.path.join(dataset_path, class_name)
        
        # get full paths of all images within the class_path. Check if its a file.
        image_paths = sorted([os.path.join(class_path, image) for image in os.listdir(class_path)
                              if os.path.isfile(os.path.join(class_path, image)) and
                              imghdr.what(os.path.join(class_path, image)) != None])
        number_images_in_class = len(image_paths)
        number_images += number_images_in_class
        
        # add as many class_ids to the ordered_classes as there are number_images
        # list addition + list multiplication [1,1,1,2,2] + ([3] * 2) = [1,1,1,2,2,3,3]
        classes_list += ([class_id] * number_images_in_class)
        
        dataset_dict[class_id] = {"class_name":class_name,
                                  "class_path":class_path,
                                  "image_paths":image_paths}
        
        logging.debug("class_id: {0}, class_name: {1}, number_images: {2}".format(class_id,
                                                                                  class_name,
                                                                                  len(image_paths)))
    
    dataset_dict["number_classes"] = number_classes
    dataset_dict["number_images"] = number_images
    dataset_dict["classes_list"] = classes_list
    
    logging.info("total classes : {0}, total images: {1}".format(number_classes, number_images))
    logging.info("done, generating dataset dictionary")
    
    # for key,value in dataset_dict.iteritems():
    #     if isinstance(key, int):
    #         for image_path in dataset_dict[key]["image_paths"]:
    #             plt.imshow(get_image_from_file(image_path).astype(np.int), interpolation='nearest', cmap=cm.Greys)
    #             plt.show()
    #             
    #             print get_image_from_file(image_path).shape
    #             print get_image_from_file(image_path).dtype
    #             print get_image_from_file(image_path).astype(np.uint8).dtype
    #             print get_image_from_file(image_path).astype(np.uint8)[36:48,78:96]
    #             save_image(get_image_from_file(image_path), "/home/nitish/mas_course_ss2015/assignments/sparsex/datasets/yale_face_b_ext_cropped/test.pgm")
    #             break
    #     break
    return dataset_dict


def get_dataset_from_dataset_dict(dataset_dict, standard_size=(64,64)):
    logging.info("generating dataset from dataset dict")
    number_images = dataset_dict["number_images"]
    number_classes = dataset_dict["number_classes"]
    
    # shape (number_images, image_size[0], image_size[1])
    X = np.empty((number_images, standard_size[0], standard_size[1]), dtype=float)
    Y = np.array(dataset_dict["classes_list"], dtype=int)
    
    X_index = 0
    # for each class_id in dictionary [in ascending order] 
    for class_id in range(number_classes):
        class_image_paths = dataset_dict[class_id]["image_paths"]
        
        # for each image path in list for the class_id [in ascending order] extract image and place into X
        for image_path in class_image_paths:
            X[X_index] = imresize(get_image_from_file(image_path),standard_size)
            X_index += 1

    logging.info("X.shape: {0}".format(X.shape))
    logging.info("Y.shape: {0}".format(Y.shape))
    logging.info("done, generating dataset from dataset dict")
    # image_array, Y
    return X, Y


def get_dataset(config_params):
    if config_params["dataset_extraction_function"] == None:
        logging.info("extracting data from dataset_path")
        dataset_dict = get_dataset_dictionary(config_params["dataset_path"])
        X_raw, Y_raw = get_dataset_from_dataset_dict(dataset_dict, config_params["preprocess_resize"])
        logging.info("done extracting data")
    else:
        logging.info("extracting data from dataset_extraction_function")
        X, Y = config_params["dataset_extraction_function"]()
        X_raw, Y_raw = X, Y
        logging.debug("number_classes: {0}".format(len(set(Y_raw))))
        logging.debug("number_images: {0}".format(Y_raw.shape[0]))
        logging.info("done extracting data")
        
    return X_raw, Y_raw


def preprocess(image_array, config_params):
    logging.info("preprocessing")
    preprocessing = Preprocessing()
    patches = preprocessing.pipeline(image_array=image_array, 
                                     image_size=config_params["preprocess_resize"],
                                     patch_size=config_params["preprocess_patch_size"],
                                     normalize=config_params["preprocess_normalization"],
                                     whiten=config_params["preprocess_whitening"],
                                     multiple_images=True)
    
    logging.debug("patches.shape: {0}".format(patches.shape))
    logging.info("done, preprocessing")
    return patches
    

def train_sparse_feature_extractor(patches, config_params):
    logging.info("training feature extractor")
    
    if config_params["feature_extraction_input_mode_filename"] != None:
        logging.info("loading dictionary from model file: {0}".format(config_params["feature_extraction_input_mode_filename"]))
        # initialize sparse_coding object
        sparse_coding = SparseCoding(library_name=config_params["feature_extraction_library"],
                                     model_filename=config_params["feature_extraction_input_mode_filename"],
                                     **config_params["feature_extraction_params"])
        
        # dictionary shape
        logging.debug("dictionary_shape: {0}".format(sparse_coding.get_dictionary().shape))

        logging.info("done, loading dictionary from model file")
        
    else:
        # initialize sparse_coding object
        sparse_coding = SparseCoding(library_name=config_params["feature_extraction_library"],
                                     model_filename=None,
                                     **config_params["feature_extraction_params"])
        
        # learn dictionary
        sparse_coding.learn_dictionary(patches, multiple_images=True)
        logging.debug("dictionary_shape: {0}".format(sparse_coding.get_dictionary().shape))

        # save model
        logging.info("saving sparse coding model : {0}".format(config_params["feature_extraction_output_model_filename"]))
        sparse_coding.save_model(config_params["feature_extraction_output_model_filename"])

        logging.info("done, training feature extractor")
        
    return sparse_coding
    
    
def train_classifier(X, Y, config_params):
    logging.info("training classifier")
    
    if config_params["classification_input_model_filename"] != None:
        logging.info("loading classification model file: {0}".format(config_params["classification_input_model_filename"]))
    
        # initialize classifier object
        classifier = Classifier(library_name=config_params["classification_library"],
                                model_filename=config_params["classification_input_model_filename"],
                                **config_params["classification_params"])
        
        logging.info("done, loading classifier from model file")
        
    else:
        # initialize classifier object
        classifier = Classifier(library_name=config_params["classification_library"],
                                model_filename=None,
                                **config_params["classification_params"])
    
        # train
        classifier.train(X,Y)
        
        # save model
        logging.info("saving classification model : {0}".format(config_params["classification_output_model_filename"]))
        classifier.save_model(config_params["classification_output_model_filename"])
    
        logging.info("done, training classifier")
    
    return classifier


def validate(X_train, Y_train, X_test, Y_test, config_params): 
    ## training
    # preprocess 
    patches = preprocess(X_train, config_params)

    # train sparse feature extractor
    sparse_coding = train_sparse_feature_extractor(patches, config_params)

    # extract sparse features
    sparse_X_train = sparse_coding.pipeline(patches,
                                            sign_split=config_params["feature_extraction_sign_split"],
                                            pooling=config_params["feature_extraction_pooling"],
                                            pooling_size=config_params["feature_extraction_pooling_filter_size"],
                                            post_pooling_standardization=config_params["feature_extraction_post_pooling_standardization"],
                                            multiple_images=True)

    # repeat the training data
    sparse_X_train = np.tile(sparse_X_train, (config_params["classification_training_repeats"],1))
    Y_train = np.tile(Y_train, config_params["classification_training_repeats"])

    # train classifier
    classifier = train_classifier(sparse_X_train, Y_train, config_params)

    ## testing
    # preprocess 
    patches = preprocess(X_test, config_params)

    # extract sparse features
    sparse_X_test = sparse_coding.pipeline(patches,
                                           sign_split=config_params["feature_extraction_sign_split"],
                                           pooling=config_params["feature_extraction_pooling"],
                                           pooling_size=config_params["feature_extraction_pooling_filter_size"],
                                           post_pooling_standardization=config_params["feature_extraction_post_pooling_standardization"],
                                           multiple_images=True)

    # get predictions
    Y_predict = classifier.get_predictions(sparse_X_test)

    # scores
    correct_predictions = np.sum(Y_test == Y_predict)
    total_predicitions = Y_test.shape[0]
    percentage_correct = (correct_predictions * 100.0) / total_predicitions

    logging.debug("Y_test    : {0}".format(Y_test))
    logging.debug("Y_predict : {0}".format(Y_predict))
    logging.info("PREDICTION_RATE: {0}% ... {1} / {2}".format(percentage_correct, correct_predictions, total_predicitions))
    
    # cm = confusion_matrix(Y_test, Y_predict)
    # # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # imsave(os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/confusion_matrix.png")), cm)
    # imsave(os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/confusion_matrix.jpg")), cm)
    
    confusion_matrix_dict = {
        "Y_test":Y_test,
        "Y_predict":Y_predict
    }
    write_dictionary_to_pickle_file(os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/confusion_matrix_dict.pkl")),
                                    confusion_matrix_dict)
    


def main():
    config_params = get_config_params()
    
    # validate config_params (pending)

    # get dataset
    X_raw, Y_raw = get_dataset(config_params)
    
    if config_params["validation"] and not config_params["cross_validation"] :
        # single validation
        random_state = np.random.RandomState(config_params["random_seed"])
        train_index = random_state.rand(Y_raw.shape[0]) > config_params["validation_split"]
        test_index = ~train_index
        X_train, Y_train = X_raw[train_index], Y_raw[train_index]
        X_test, Y_test = X_raw[test_index], Y_raw[test_index]
        validate(X_train, Y_train, X_test, Y_test, config_params)
        
    elif config_params["validation"] and config_params["cross_validation"]:
        # cross validation
        random_state = np.random.RandomState(config_params["random_seed"])
        skf = StratifiedKFold(Y_raw, n_folds=config_params["cross_validation_folds"], shuffle=True, random_state=random_state)
        for train_index, test_index in skf:
            ## training
            X_train, Y_train = X_raw[train_index], Y_raw[train_index]
            X_test, Y_test = X_raw[test_index], Y_raw[test_index]
            validate(X_train, Y_train, X_test, Y_test, config_params)
            
            # relearn dictionary
            if config_param["cross_validation_relearn_dictionary"]:
                pass
            # don't relearn dictionary
            else:
                config_params["feature_extraction_input_mode_filename"] = config_params["feature_extraction_output_model_filename"]
    
    else:
        ## no validation, only training
        X_train, Y_train = X_raw, Y_raw
        
        # preprocess 
        patches = preprocess(X_train, config_params)

        # train sparse feature extractor
        sparse_coding = train_sparse_feature_extractor(patches, config_params)

        # extract sparse features
        sparse_X_train = sparse_coding.pipeline(patches,
                                                sign_split=config_params["feature_extraction_sign_split"],
                                                pooling=config_params["feature_extraction_pooling"],
                                                pooling_size=config_params["feature_extraction_pooling_filter_size"],
                                                post_pooling_standardization=config_params["feature_extraction_post_pooling_standardization"],
                                                multiple_images=True)

        # repeat the training data
        sparse_X_train = np.tile(sparse_X_train, (config_params["classification_training_repeats"],1))
        Y_train = np.tile(Y_train, config_params["classification_training_repeats"])

        # train classifier
        classifier = train_classifier(sparse_X_train, Y_train, config_params)




if __name__=="__main__":
    logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)30s() ] %(message)s",
                        level=logging.DEBUG,
                        stream=sys.stdout)
    
    # parser = argparse.ArgumentParser(description='Script for training the Sparsex feature extraction and classification models.',
    #                                  prog='python sparsex_learn.py',
    #                                  usage='%(prog)s config_filename')
    # 
    # # positional arguments
    # parser.add_argument('config_filename', metavar='config_filename', type=str,
    #                     help='config filename containing all the learning config params')
    # 
    # args = parser.parse_args()
    # 
    # config_filename = args.config_filename
    # 
    # config_params = get_config_params_from_file(config_filename)
    
    main()
    
    
    