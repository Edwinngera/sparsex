from ..preprocessing.preprocessing import Preprocessing
from ..feature_extraction.feature_extraction import SparseCoding
from ..classification.classification import Classifier
from ..customutils.customutils import get_image_from_file
from scipy.misc import imresize
from sklearn.cross_validation import StratifiedKFold
import sys, argparse, os, logging, imghdr
import numpy as np
import sparsex_train_config

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def get_config_params():
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
            
            break
        break

    logging.info("X.shape: {0}".format(X.shape))
    logging.info("Y.shape: {0}".format(Y.shape))
    logging.info("done, generating dataset from dataset dict")
    # image_array, Y
    return X, Y


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
    

def extract_features(patches, config_params):
    logging.info("feature_extraction")
    sparse_coding = SparseCoding(library_name=config_params["feature_extraction_library"],
                                      model_filename=None,
                                      **config_params["feature_extraction_params"])

    # learning dictionary
    learn_dictionary = sparse_coding.learn_dictionary(patches, multiple_images=True)
    logging.debug("dictionary_shape: {0}".format(sparse_coding.get_dictionary().shape))

    # extract sparse_features
    sparse_features = sparse_coding.pipeline(patches, sign_split=True, pooling=True, pooling_size=(3,3), multiple_images=True)
    logging.debug("sparse_features.shape: {0}".format(sparse_features.shape))

    # save model
    logging.info("saving sparse coding model : {0}".format(config_params["feature_extraction_output_model_filename"]))
    sparse_coding.save_model(config_params["feature_extraction_output_model_filename"])

    # reshape (number_image, number_features)
    sparse_features = sparse_features.reshape(sparse_features.shape[0], -1)

    logging.info("done, feature_extraction")
    return sparse_features
    
    
def train_classifier(X, Y, config_params):
    logging.info("training classifier")
    classifier = Classifier(library_name=config_params["classification_library"],
                            **config_params["classification_params"])
    
    # train
    classifier.train(X,Y)
    
    # save model
    logging.info("saving classification model : {0}".format(config_params["classification_output_model_filename"]))
    classifier.save_model(config_params["classification_output_model_filename"])
    
    logging.info("done, training classifier")
    
    return classifier


def main():
    config_params = get_config_params()
    
    # validate config_params

    # convert dataset to a dictionary containing classes, images paths, other meta-data
    dataset_dict = get_dataset_dictionary(config_params["dataset_path"])
    
    # get data
    X_raw, Y_raw = get_dataset_from_dataset_dict(dataset_dict, config_params["preprocess_resize"])
    
    # cross validation
    random_seed = 2902 # 29th Feb
    random_state = np.random.RandomState(2902)
    skf = StratifiedKFold(Y_raw, n_folds=3, shuffle=True, random_state=random_state)
    for train_index, test_index in skf:
        ## training
        X_train, Y_train = X_raw[train_index], Y_raw[train_index]
        
        # preprocess 
        patches = preprocess(X_train, config_params)
        
        # extract features
        sparse_X_train = extract_features(patches, config_params)
        
        # repeat the training data
        sparse_X_train = np.tile(sparse_X_train, (config_params["classification_training_repeats"],1))
        Y_train = np.tile(Y_train, config_params["classification_training_repeats"])
        
        # train classifier
        classifier = train_classifier(sparse_X_train, Y_train, config_params)
        
        ## testing
        X_test, Y_test = X_raw[test_index], Y_raw[test_index]
        
        # preprocess 
        patches = preprocess(X_test, config_params)
        
        # extract features
        sparse_X_test = extract_features(patches, config_params)
        
        # get predictions
        Y_predict = classifier.get_predictions(sparse_X_test)
        
        # scores
        correct_predictions = np.sum(Y_test == Y_predict)
        total_predicitions = Y_test.shape[0]
        percentage_correct = (correct_predictions * 100.0) / total_predicitions
        
        logging.info("PREDICTION_RATE: {0}% ... {1} / {2}".format(percentage_correct, correct_predictions, total_predicitions))



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
    
    
    