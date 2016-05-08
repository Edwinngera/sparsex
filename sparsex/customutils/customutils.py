from sklearn.feature_extraction.image import extract_patches_2d
from PIL import Image
import numpy as np
import os
import json, h5py
import datetime, time
import cPickle

# source : http://stackoverflow.com/questions/13990465/3d-numpy-array-to-2d
def get_giant_patch_image(patches, dtype='uint8', scale=False):
    patches = patches.copy()
    if scale:
        original_patches_shape = patches.shape
        patches = patches.reshape((patches.shape[0], -1))
        patches = patches - patches.min(axis=1)[:, np.newaxis]
        patches = patches * (255.0 / np.abs((patches.max(axis=1) - patches.min(axis=1))[:, np.newaxis]))
        patches = patches.reshape(original_patches_shape)
    sqrt_of_rows = int(np.sqrt(patches.shape[0]))
    broken_patches = patches.reshape(sqrt_of_rows, sqrt_of_rows, patches.shape[1], patches.shape[2])
    giant_patch_image = broken_patches.swapaxes(1,2).reshape(sqrt_of_rows * patches.shape[1],-1)
    return giant_patch_image.astype(dtype)


def save_image(image_array, destination_filename, dtype='uint8'):
    image_pil = Image.fromarray(image_array.astype(dtype))
    image_pil.save(destination_filename)


def get_image_from_file(image_filename, dtype='float'):
    image_pil = Image.open(image_filename)
    image = np.array(image_pil).astype(dtype)
    return image


def resize_image_to_64x64(image_filename):
    image_pil_64x64 = Image.open(image_filename).resize((64,64))
    new_image_filename = image_filename.split(".")[0] + "_64x64." + image_filename.split(".")[1] 
    image_pil_64x64.save(new_image_filename)

## json
def get_json_from_dictionary(dictionary):
    json_string = json.dumps(dictionary)
    return json_string

def write_dictionary_to_json_file(filename, dictionary):
    json_string = get_json_from_dictionary(dictionary)
    with open(filename, "w") as f:
        f.write(json_string)

def read_dictionary_from_json_file(filename):
    with open(filename, 'r') as f:
        json_string = f.read()
        dictionary = json.loads(json_string)
    return dictionary

## h5py
def write_dictionary_to_h5_file(filename, dictionary):
    with h5py.File(filename, 'w') as f:
        for key in dictionary.keys():
            f.create_dataset(key, data=dictionary[key])

def read_dictionary_from_h5_file(filename):
    with h5py.File(filename, 'r') as f:
        dictionary = {}
        for key in f.keys():
            dictionary[key] = f[key][:]
    return dictionary

## cPickle
def write_dictionary_to_pickle_file(filename, dictionary):
    with open(filename, "wb") as f:
        cPickle.dump(dictionary, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
def read_dictionary_from_pickle_file(filename):
    return np.load(filename)


# source : http://stackoverflow.com/questions/13890935/does-pythons-time-time-return-the-local-or-utc-timestamp
def get_current_string_timestamp(datetime_format="%Y-%m-%d_%H:%M:%S.%f"):
    string_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(datetime_format)
    return string_timestamp
