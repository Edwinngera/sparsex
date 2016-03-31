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

#### Priority
* Pipeline, add Protobuf Response message.
* Pipeline, server actions, get predicitions pipeline.
* Pipeline, server actions, get features pipeline.
* Pipeline, server actions, shutdown server.
* Pipeline, test on different Requests and Responses.


#### Backlog
* Pipeline, Add a constant "image_size" as an attribute for the pipeline as a whole so that there is one constant image size for the entirety of the pipeline from training to classification.
* Training, dictionary learning.
* Training, train classifier after extracting features for one image.
* Feature extraction, use Spams.
* Classification, use Joachim's SVM-Light.
* Classification, the classifier needs to know how many features it is requires so that we can put a check if number of incoming/input features is the same as number of features required by the classifier.
* Pipeline, be able to choose the dictionary learning library.
* Pipeline, be able to choose which classifier/classifier-library.
* Pipeline, make sure its an image when received before proceeding to preprocessing after receiving an image on the server.
* Pipeline, make sure image array is always float once received by server.
* Pipeline, come up with standardized shapes for preprocessing, feature_extraction and classification. i.e. whether the incoming image arrays, patches or features are flattened or 2-d or 3-d etc.
* Pipeline, API / function calls for single images or multi-images. This is mostly to avoid confusion in expecting shapes of incoming/input arrays when extracting features or classifying. Should be part of the standardization of shapes.
* Pipeline, catch TypeError in server-client communication for when wrong data format is being set.
* Pipeline, catch all known Server related errors so that sockets and client connections can be cleanly terminated.
* Pipeline, create pre-defined Response messages for "empty-results", "server-error", "server-interrupt" etc.
* Pipeline, client and server timeout.
* Preprocessing, Inplace = True/False option for some of the preprocessing steps. Perhaps for memory conservation.
* Project, fully setuptools/pip installable
* Project, documentation
    * Latex Folder with .tex file and .pdf
* Project, update license
* Project, saving models with pickle.HIGHEST_PROTOCOL
* Tests, making feature extraction tests more dynamic. Tests are working great but its just that if we need to be able to test different things then there is no API to do it.


#### Wishlist
* Pipeline, handle different color channels.
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


# Notes

#### Protocol Buffer (Protobuf) Installation and Message Compilation
1. Install Protobuf on your system such that the ```protoc``` command is available on the command line.
    * Install Protobuf Compiler on Ubuntu 14.04
        * ```sudo apt-get install protobuf-compiler```
    * For more information, follow this [link](https://github.com/google/protobuf).
2. Create a ```<FILENAME>.proto``` message file.
    * Follow the instructions mentioned on this [link](https://developers.google.com/protocol-buffers/docs/pythontutorial#defining-your-protocol-format).
3. Compile the message file to generate any available language compatible file.
    * For Python, the command is:
        * ```protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/<FILENAME>.proto```
    * For more language options and support, follow this [link](https://developers.google.com/protocol-buffers/docs/pythontutorial#compiling-your-protocol-buffers).
