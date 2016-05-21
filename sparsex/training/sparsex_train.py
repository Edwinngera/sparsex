from ..preprocessing.preprocessing import Preprocessing
from ..classification.classification import Classifier
from ..customutils.customutils import read_string_from_file
import sys, argparse, os, logging
import sparsex_train_config

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def get_config_params():
    return sparsex_train_config.config_params


# dataset_dict = {class_id: {class_name:str, class_path:str, image_paths:list(str)}}
def get_dataset_dictionary(dataset_path):
    logging.info("Generating dataset dictionary")
    # get full path of the dataset
    dataset_path = os.path.realpath(dataset_path)
    
    # get list of directories in dataset path which will become the class names
    # sorting is also to make sure that the order is always the same during different iterations
    dataset_dirs = sorted([d for d in os.listdir(dataset_path) 
                    if not os.path.isfile(os.path.join(dataset_path, d))])
    
    # create dict
    dataset_dict = {}
    number_images = 0
    
    # populate the dict
    for class_id in range(len(dataset_dirs)):
        # class_name is the directory name
        class_name = dataset_dirs[class_id]
        
        # class_path is the full path of the directory
        class_path = os.path.join(dataset_path, class_name)
        
        # get full paths of all images within the class_path. Check if its a file.
        image_paths = sorted([os.path.join(class_path, image) for image in os.listdir(class_path)
                              if os.path.isfile(os.path.join(class_path, image))])
        number_images += len(image_paths)
        
        dataset_dict[class_id] = {"class_name":class_name,
                                  "class_path":class_path,
                                  "image_paths":image_paths}
        
        logging.debug("class_id: {0}, class_name: {1}, number_image: {2}".format(class_id,
                                                                                 class_name,
                                                                                 len(image_paths)))
    
    logging.info("Total classes : {0}, total images: {1}".format(len(dataset_dirs),
                                                                 number_images))
    logging.info("Done, generating dataset dictionary")
    return dataset_dict


def learn():
    config_params = get_config_params()

    # convert dataset to classes
    dataset_dict = get_dataset_dictionary(config_params["dataset_path"])


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
    
    
    learn()
    
    
    