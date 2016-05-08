# README

This is an RnD project part of the curriculum for the Master of Autonomous Systems program in University of Applied Sciences Bonn-Rhein-Sieg. The aim is to be build an object recognition tool using deep learning and sparse coding for feature extraction.

Project is currently maintained by Nitish Reddy Koripalli (21st February 2016).


# Installation
**(Tested on Ubuntu 14.04, Python 2.7.11)**

### Using setuptools
1. Clone this repository by running the following command on the command line (make sure you have the access rights to the repository):
    * ```$ git clone git@bitbucket.org:nitred/sparsex.git```
2. On the command line, cd into the sparsex directory
    * ```$ cd sparsex```
3. On the command line, execute the following:
    * ```$ python setup.py install```
4. Test installation by running the following commands on the command line:
    * Test Communications : ```$ sparsex_communication_test```
    * Test Get Features : ```$ sparsex_get_features_test```
    * Test Get Predictions : ```$ sparsex_get_predictions_test```


# Development

#### Priority
* Feature extraction, use Spams.
* Feature extraction, spams encoding.
* Feature extraction, spams feature sign split.
* Feature extraction, spams max pooling.

#### Backlog
* Tests, use PyUnit.
* Pipeline, expose API for handling many intricasies for fine tuning the server config. Also think of a way of letting the user choose a custom pipeline by choosing any of the following. Sort of like a pipeline buffet where it transfers the output of one to the other. Also the user can provide the data to any given function rather than relying on the pipeline.
    * Preprocessing:
        * Set image resize size, get resized image.
        * Set patch size, get patches. Option flatten.
        * Get normalized image, get normalized patches, get normalization coefficients. Option flatten.
        * Get whitened image, get whitened patches, get whitening coefficients. Option flatten.
        * Or always flatten.
    * Feature extraction:
        * Use feature extraction model file.
        * Train dictionary.
        * Get dictionary.
        * Get sparse code.
        * Get sign split features.
        * Set pooling window size, get pooled features.
    * Classification:
        * Use classification model file.
        * Train classifier.
        * Get predictions.
* Pipeline, split requests into meta-data and data so that the server can make sure meta-data is in perfect order before transfer of data/dataset.
* Pipeline, handle different color channels.
* Pipeline, handle h5 files in requests for training datasets.
* Pipeline, add "checksum" functionality in request and response messages for data (image / image_array) verfication.
* Pipeline, make sure all/most data types for the data bytes are handled.
* Pipeline, Add a constant "image_size" as an attribute for the pipeline as a whole so that there is one constant image size for the entirety of the pipeline from training to classification.
* Pipeline, come up with standardized shapes for preprocessing, feature_extraction and classification. i.e. whether the incoming image arrays, patches or features are flattened or 2-d or 3-d etc. Take into consideration that some classes/functions are more time consuming than others. 
* Training, toggle between TrainDL and TrainDL_memory.
* Training, script file. Add as entry_points-console script with default config.
* Training, script file, maybe use command line arguments and --help.
* Training, config file, maybe use command line like arguments for configrations. For example -D for user provided dictionary.
* Training, install Joachim's SVC outside of python.
* Training, test basic linSVC functionality.
* Training, either use subprocess to call linSVC or itegrate it into the python environment.
* Training, train classifier after extracting features for one image.
* Feature extraction, standardize/normalize features option after encoding and pooling.
* Feature extraction, think about using inheritance or composition.
* Feature extraction, change learn_dictionary method argument name from whitened_patches to something more generic.
* Feature extraction, spams, figure out which parameters to use.
* Feature extraction, spams, loading and saving models. Catch and handle exceptions properly.
* Feature extraction, spams, when updating self.params with kwargs. Try to handle exceptions and wrong keyword arguments.
* Feature extraction, spams, choose between different decomposition/encoding approaches.
* Feature extraction, spams, identify and extract all parameters from kwargs for train_params and encoding_params.
* Classification, use Joachim's SVM-Light.
* Classification, the classifier needs to know how many features it is requires so that we can put a check if number of incoming/input features is the same as number of features required by the classifier.
* Pipeline, server configuration state.
* Pipeline, be able to choose the dictionary learning library.
* Pipeline, be able to choose which classifier/classifier-library.
* Pipeline, make sure its an image when received before proceeding to preprocessing after receiving an image on the server.
* Pipeline, make sure image array is always float once received by server.
* Pipeline, API / function calls for single images or multi-images. This is mostly to avoid confusion in expecting shapes of incoming/input arrays when extracting features or classifying. Should be part of the standardization of shapes.
* Pipeline, catch TypeError in server-client communication for when wrong data format is being set.
* Pipeline, catch all known Server related errors so that sockets and client connections can be cleanly terminated.
* Preprocessing, Inplace = True/False option for some of the preprocessing steps. Perhaps for memory conservation.
* Project, installing numpy and other hard to install packages.
* Project, pip installable.
* Project, use subprocess instead of threads.
* Project, logging functionality.
* Project, documentation.
    * Latex Folder with .tex file and .pdf
* Project, update license.
* Project, saving models with pickle.HIGHEST_PROTOCOL
* Project, look through code for method or variable name inconsistencies.
* Project, use of subprocess instead of threading.
* Project, shift into new-style classes.
* Project, arguments sent into methods should not be altered in case those argument values are used eleswhere. Create new variables to store manipulations of the arguments.
* Tests, making feature extraction tests more dynamic. Tests are working great but its just that if we need to be able to test different things then there is no API to do it.
* Tests, installation test using virtual environments.


#### Wishlist
* Continuity functionality
    * If there's an error in the code and the process breaks. You can pickup where you left off after you fix the bug.
* On-line Training
    * Train on the go, one image at a time.
    * This might not be the best idea since having bulk training samples instead lets you use the GPU more efficiently.


### Possible Issues
* Make sure most dtypes are float when converting images from grayscale.
* sys.stdout.flush() seems to flush messages from both server and client stuff for some reason. Probably because in this case I am running server and client on my own system and they share stdout. This should be resolved in a more elegant manner. I am unsure why the messages from server would end on client side. Possible bug?
* Which base datatype should be used before extracting patches. Size constraints? Computation constraints and bottlenecks?
* Using .T to transpose a matrix in numpy, seems to be interchanging between 'C' and 'F' orders.


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

#### Pip Installation Quick Codes
* ```pip install git+file:///path/to/your/git/repo@mybranch```
* ```pip install --extra-index-url https://testpypi.python.org/pypi sparsex```

#### Permission issues with /tmp
* careful!!! : ```mount -o remount exec /tmp```

#### Python Run Package Files (-m Flag)
* This will prevent errors due to importing modules using ".."
* python -m sparsex.[package_name].[package_filename_without .py extension]

# Useful Links
1. Installation Notes
    * Tool Recommendations
        * [Installation and Packaging Tool Recommendations](https://python-packaging-user-guide.readthedocs.io/en/latest/current/)
    * Dependency Installation
        * [Building and Distributing Packages with Setuptools](https://pythonhosted.org/setuptools/setuptools.html#dependencies-that-aren-t-in-pypi)
        * [Declaring Dependencies](http://pythonhosted.org/setuptools/setuptools.html#declaring-dependencies)
        * [C & C++ Extensions](https://docs.python.org/2/extending/building.html)
        * [SO : How can I make setuptools install a package that's not on PyPI?](http://stackoverflow.com/questions/3472430/how-can-i-make-setuptools-install-a-package-thats-not-on-pypi)
    * Installing Package : Numpy
        * [SO : Installing numpy as a dependency with setuptools](http://stackoverflow.com/questions/8710918/installing-numpy-as-a-dependency-with-setuptools)
        * [SO : How to Bootstrap numpy installation in setup.py](http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py)
        * [SO : Why doesn't setup_requires work properly for numpy?](http://stackoverflow.com/questions/21605927/why-doesnt-setup-requires-work-properly-for-numpy)
        * [Blas vs Atlas vs OpenBlas vs MKL, Benchmarks](http://blog.nguyenvq.com/blog/2014/11/10/optimized-r-and-python-standard-blas-vs-atlas-vs-openblas-vs-mkl/)
    * Installing Package : PyZMQ
        * [Building and Installing PyZMQ](https://github.com/zeromq/pyzmq/wiki/Building-and-Installing-PyZMQ)
    * Installing Package : h5py
        * [SO : Permission issues with /tmp/](http://stackoverflow.com/questions/26097398/installing-python-cryptography-error/27983562#27983562)
    * Installing Package : scipy
        * [Issue with importing scipy.misc.imresize](https://github.com/Newmu/stylize/issues/1)
        * [OS : Installing SciPy with pip](http://stackoverflow.com/questions/2213551/installing-scipy-with-pip)
        * [OS : PIP Install Numpy throws an error “ascii codec can't decode byte 0xe2](http://stackoverflow.com/questions/26473681/pip-install-numpy-throws-an-error-ascii-codec-cant-decode-byte-0xe2)
    * Installing Package : Atlas
        * [Important notes and installation guide](https://bazaar.launchpad.net/~ubuntu-branches/ubuntu/trusty/atlas/trusty/view/head:/debian/README.Debian)
        * [SO : Building ATLAS (and later Octave w/ ATLAS)](http://askubuntu.com/questions/472146/building-atlas-and-later-octave-w-atlas)
        * [Part of Caffe installation](http://caffe.berkeleyvision.org/install_apt.html)
        * [libgfortran linking issue](https://github.com/ContinuumIO/anaconda-issues/issues/686)

1. Packaging and Distributing
    * [Packaging and Distributing Projects](https://python-packaging-user-guide.readthedocs.io/en/latest/distributing/#packaging-and-distributing-projects)
    * [Choosing a versioning scheme](https://python-packaging-user-guide.readthedocs.io/en/latest/distributing/#choosing-a-versioning-scheme)

1. Packages
    * [SPAMS](http://spams-devel.gforge.inria.fr/)
    * [Intel MKL Buy/Free](https://software.intel.com/en-us/intel-mkl/try-buy)

1. Virtual Environments
    * [Virtual Environment Guide](http://docs.python-guide.org/en/latest/dev/virtualenvs/)
    * [Conda Virtual Environments](http://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

1. Ubuntu Dependencies
    * Atlas
        * ```sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler```
        * ```sudo apt-get install --no-install-recommends libboost-all-dev```
        * ```sudo apt-get install libatlas-base-dev```
        * Source : [Part of Caffe installation](http://caffe.berkeleyvision.org/install_apt.html)

1. Jypyter Notebooks
    * [Run Jupyterhub on a Supercomputer](http://zonca.github.io/2015/04/jupyterhub-hpc.html)

1. Python Tips & Tricks
    * [Inheritance Versus Composition](http://learnpythonthehardway.org/book/ex44.html)
