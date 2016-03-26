# README

This is an RnD project part of the curriculum for the Master of Autonomous Systems program in University of Applied Sciences Bonn-Rhein-Sieg. The aim is to be build an object recognition tool using deep learning and sparse coding for feature extraction.

Project is currently maintained by Nitish Reddy Koripalli (21st February 2016).


# Installation
**(Tested on Ubuntu 14.04, Python 2.7.11)**

### Using pip (from Bitbucket)
Execute the following commands (be sure to enter your USERNAME & TAG):

1. ```pip uninstall sparsex```
2. ```pip install git+https://<USERNAME>@bitbucket.org/nitred/sparsex.git@<TAG>```

### Using pip (from Github)
Execute the following commands (be sure to enter your USERNAME & TAG):

1. ```pip uninstall sparsex```
2. ```pip install git+https://<USERNAME>@github.com/nitred/sparsex.git@<TAG>```


### Using setuptools
1. Clone this repository using by executing the following command on the command line:
    * ```git clone git@bitbucket.org:nitred/sparsex.git```
2. On the command line, cd into the sparsex directory
    * ```cd sparsex```
3. On the command line, execute the following:
    * ```python setup.py install```


# Development
### To-Do

#### Priority
* Preprocessing, Add "reshape to some fixed image size" functionality to preprocessing. The incoming images must be reshaped to a standard/pre-defined image-size in the pipeline. Make sure that there is one constant image size for the entirety of the pipeline from training to classification.
* Feature extraction, save and load dictionary to file functionality. Save to a fixed folder. Use timestamps for naming.
* Feature extraction, get dictionary-encoded weight-vector of an image
* Feature extraction, feature sign-split
* Feature extraction, pooling
* Feature extraction, combined pipeline


#### Backlog
* Feature extraction, dictionary learning
* Pipeline, Add a constant "image_size" as an attribute for the pipeline as a whole so that there is one constant image size for the entirety of the pipeline from training to classification.
* Pipeline, create a "command-dictionary" for client-server communication. It can include:
    * "action:classify/train/get_encoding/etc"
    * "image_type:jpg_image/png_image/np_array"
    * "image_shape:(600,100)/None/etc"
    * "image_data:...."
* Classification, get class after extracting features.
* Training, train classifier after extracting features for one image.
* Pipeline, be able to choose the dictionary learning library.
* Pipeline, be able to choose which classifier/classifier-library.
* Pipeline, classification pipeline, finish simple-classifier, server-classifier, server-thread-classifier
* Pipeline, server-client image acquisition, attributes and data transfer dictionary.
* Pipeline, make sure its an image when received before proceeding to preprocessing after receiving an image on the server.
* Pipeline, make sure image array is always float once received by server.
* Preprocessing, Inplace = True/False option for some of the preprocessing steps. Perhaps for memory conservation.
* Project, fully setuptools/pip installable
* Project, documentation
    * Latex Folder with .tex file and .pdf
* Project, update license


#### Wishlist
* Logging functionality
* Continuity functionality
    * If there's an error in the code and the process breaks. You can pickup where you left off after you fix the bug.
* On-line Training
    * Train on the go, one image at a time.
    * This might not be the best idea since having bulk training samples instead lets you use the GPU more efficiently.


### Possible Issues
* Image acquisition, client does not handle well when the server goes down. Client sends a request to an unavailable server and waits for a response but does not recover when the server comes back online.
* Make sure most dtypes are float when converting images from grayscale
* sys.stdout.flush() seems to flush messages from both server and client stuff for some reason. Probably because in this case I am running server and client on my own system and they share stdout. This should be resolved in a more elegant manner. I am unsure why the messages from server would end on client side. Possible bug?
* Which base datatype should be used before extracting patches. Size constraints? Computation constraints and bottlenecks?