from ..preprocessing.preprocessing import Preprocessing
from ..feature_extraction.feature_extraction import SparseCoding
from ..classification.classification import Classifier
from ..customutils.customutils import get_image_from_file
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
        number_images += len(image_paths)
        
        # add as many class_ids to the ordered_classes as there are number_images
        # list addition + list multiplication
        classes_list += [class_id] * number_images
        
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


def construct_empty_data_array(config_params, dataset_dict):
    logging.info("constructing empty data array for preprocessed patches")
    # calculate the size of the combined data patches
    number_images = dataset_dict["number_images"]
    image_size = config_params['preprocess_resize'][0]
    patch_size = config_params['preprocess_patch_size'][0]
    # e.g. 64-8+1 = 57,  57**2 = 3249
    number_patches = (image_size - patch_size + 1) ** 2
    dataset_dict["number_patches"] = number_patches
    # e.g. 8**2 = 64
    number_features = patch_size ** 2
    dataset_dict["number_features"] = number_features
    # create array for patches
    empty_data_array = np.empty((number_images, number_patches, patch_size, patch_size), dtype=np.float) ########
    
    logging.debug("empty_data_array.shape : {0}".format(empty_data_array.shape))
    logging.info("done, constructing empty data array")
    return empty_data_array
    

def get_preprocessed_patches(config_params, dataset_dict, empty_preprocessed_patches):
    logging.info("getting preprocessed patches")
    preprocessing = Preprocessing()
    number_classes = dataset_dict["number_classes"]
    
    # loop through each class and each image and get whitenened patches
    for class_id in range(number_classes): #######
        for image_index in range(len(dataset_dict[class_id]["image_paths"])):
            image_path = dataset_dict[class_id]["image_paths"][image_index]
            image_array = get_image_from_file(image_path)
            image_array = preprocessing.get_resized_image(image_array, image_size=config_params["preprocess_resize"])
            patches = preprocessing.extract_patches(image_array, patch_size=config_params["preprocess_patch_size"])
            if config_params["preprocess_whitening"]:
                patches = preprocessing.get_contrast_normalized_patches(patches)
                patches = preprocessing.get_whitened_patches(patches)
            elif config_params["preprocess_normalization"]:
                patches = preprocessing.get_contrast_normalized_patches(patches)
            
            logging.debug("{0} {1} patches.shape : {2}".format(class_id, image_index, patches.shape))
            empty_preprocessed_patches[class_id] = patches # persistence check, done.

    logging.info("done, getting preprocessed patches")
    return empty_preprocessed_patches


def learn_dictionary(config_params, preprocessed_patches):
    logging.info("learning dictionary")
    original_shape = preprocessed_patches.shape
    # n_image x n_patches x patch_size x patch_size = (n_image * n_patches) x (patch_size * patch_size)
    if preprocessed_patches.ndim == 4:
        logging.debug("preprocessed_patches.ndim is 4, reshaping it to 2")
        logging.debug("preprocessed_patches.shape : {0}".format(preprocessed_patches.shape))
        preprocessed_patches = preprocessed_patches.reshape(original_shape[0]*original_shape[1],
                                                            original_shape[2]*original_shape[3])
    
    logging.debug("preprocessed_patches.shape : {0}".format(preprocessed_patches.shape))
    assert preprocessed_patches.ndim == 2, "preprocessed_patches.ndim = {0} instead of 2".format(preprocessed_patches.ndim)

    sparse_coding = SparseCoding(config_params["feature_extraction_library"],
                                 **config_params["feature_extraction_params"])
    
    sparse_coding.learn_dictionary(preprocessed_patches)
    
    logging.info("saving sparse coding model : {0}".format(config_params["feature_extraction_output_model_filename"]))
    sparse_coding.save_model(config_params["feature_extraction_output_model_filename"])
    
    logging.debug("reshaping preprocessed_patches to original shape : {0}".format(original_shape))
    preprocessed_patches = preprocessed_patches.reshape(original_shape)
    logging.info("done, learning dictionary")
    return sparse_coding, preprocessed_patches
    
    
def get_sparse_encoding(config_params, sparse_coding, preprocessed_patches):
    logging.info("getting sparse encoding")
    original_shape = preprocessed_patches.shape
    # n_image x n_patches x patch_size x patch_size = (n_image * n_patches) x (patch_size * patch_size)
    if preprocessed_patches.ndim == 4:
        logging.debug("preprocessed_patches.ndim is 4, reshaping it to 2")
        logging.debug("preprocessed_patches.shape : {0}".format(preprocessed_patches.shape))
        preprocessed_patches = preprocessed_patches.reshape(original_shape[0]*original_shape[1],
                                                            original_shape[2]*original_shape[3])
    
    logging.debug("preprocessed_patches.shape : {0}".format(preprocessed_patches.shape))
    assert preprocessed_patches.ndim == 2, "preprocessed_patches.ndim = {0} instead of 2".format(preprocessed_patches.ndim)
    
    sparse_encoding = sparse_coding.get_sparse_features(preprocessed_patches)
    logging.debug("sparse_encoding.shape: {0}".format(sparse_encoding.shape))
    
    logging.debug("reshaping preprocessed_patches to original shape : {0}".format(original_shape))
    preprocessed_patches = preprocessed_patches.reshape(original_shape)
    logging.info("done, getting sparse encoding")
    return sparse_encoding, preprocessed_patches


def get_sign_split_features(sparse_coding, sparse_features):
    logging.info("getting sign split features")
    
    sign_split_features = sparse_coding.get_sign_split_features(sparse_features)
    logging.debug("sign_split_features.shape : {0}".format(sign_split_features.shape))
    
    logging.info("done, getting sign split features")
    return sign_split_features


def get_pooled_features(config_params, dataset_dict, sparse_coding, sparse_features):
    logging.info("getting pooled features")
    
    # shape is supposed to be (number_images * number_patches) x number_feautres
    logging.debug("sparse_features.shape : {0}".format(sparse_features.shape))
    assert sparse_features.ndim == 2, "sparse_features.shape is {0} instead of 2".format(sparse_features.ndim)
    
    number_images = dataset_dict["number_images"]
    number_patches = dataset_dict["number_patches"]
    number_features = sparse_features.shape[-1]
    
    # reshape to number_images x number_patches x number_features
    sparse_features = sparse_features.reshape(number_images, number_patches, -1)
    
    logging.debug("number_images : {0}".format(number_images))
    logging.debug("number_patches : {0}".format(number_patches))
    logging.debug("number_features : {0}".format(number_features))
    logging.debug("sparse_features.shape : {0}".format(sparse_features.shape))
    
    # (19,19)
    filter_size = config_params["feature_extraction_pooling_filter_size"]
    
    # (19,19)[0] = 19
    filter_side = filter_size[0]
    
    # np.sqrt(3249) = 57
    pooling_input_feature_map_side = int(np.sqrt(number_patches)) # validate it is an int
    
    # 57 / 19 = 3
    pooled_feature_map_side = int(pooling_input_feature_map_side / filter_side)
    
    # (3**2) * 20 = 180
    number_pooled_features = (pooled_feature_map_side**2) * number_features
    
    logging.debug("pooling filter_size : {0}".format(filter_size))
    logging.debug("pooling filter_side : {0}".format(filter_side))
    logging.debug("pooling_input_feature_map_side : {0}".format(pooling_input_feature_map_side))
    logging.debug("pooled_feature_map_side : {0}".format(pooled_feature_map_side))
    logging.debug("number_pooled_features : {0}".format(number_pooled_features))
    
    # construct new pooled_features
    pooled_features = np.empty((number_images, number_pooled_features))
    
    # loop through each image (and its patches) and get pooled features
    for image_index in range(number_images):
        # extract pooled features, flatten, and populate into combined pooled features
        # sparse_feature[images_index].shape = number_patches x number_features
        pooled_features[image_index] = sparse_coding.get_pooled_features(sparse_features[image_index],
                                                                         filter_size=filter_size).reshape(-1)

    logging.debug("pooled_features.shape : {0}".format(pooled_features.shape))
    logging.info("done, getting pooled features")
    return pooled_features


def get_target_vector(dataset_dict):
    logging.info("getting target vector")
    target_vector = np.array(dataset_dict["classes_list"], dtype=int)
    
    logging.info("done, getting target vector")
    return target_vector
    
    
def train_classifier(config_params, X, Y):
    logging.info("training classifier")
    
    classifier = Classifier(library_name=config_params["classification_library"],
                            **config_params["classification_params"])
    
    # need X and Y
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
    
    # construct empty data array for the preprocessed patches
    empty_preprocessed_patches = construct_empty_data_array(config_params, dataset_dict)
    
    # populate empty_preprocessed_patches
    preprocessed_patches = get_preprocessed_patches(config_params, dataset_dict, empty_preprocessed_patches)
    
    # split into train and test set
    
    # learn dictionary
    sparse_coding, preprocessed_patches = learn_dictionary(config_params, preprocessed_patches)
    
    # get sparse encoding
    sparse_features, preprocessed_patches = get_sparse_encoding(config_params,
                                                                sparse_coding,
                                                                preprocessed_patches)
    
    if config_params["feature_extraction_sign_split"]:
        sparse_features = get_sign_split_features(sparse_coding, sparse_features)
    
    if config_params["feature_extraction_pooling"]:
        sparse_features = get_pooled_features(config_params, dataset_dict,
                                              sparse_coding, sparse_features)
        
    # final sparse_features reshape, is number of samples x number of features
    sparse_features = sparse_features.reshape(dataset_dict["number_images"], -1)
    
    # get target vector
    target_vector = get_target_vector(dataset_dict)
    
    classifier = train_classifier(config_params, sparse_features, target_vector)



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
    
    
    