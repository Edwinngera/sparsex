# README

This is an RnD project part of the curriculum for the Master of Autonomous Systems program in University of Applied Sciences Bonn-Rhein-Sieg. The aim is to be build an object recognition tool using deep learning and sparse coding for feature extraction.

Project is currently maintained by Nitish Reddy Koripalli (21st February 2016).


# Installation

**(Tested on Ubuntu 14.04, Python 2.7.11)**

1. Clone this repository using by executing the following command on the command line:
  * ```git clone git@bitbucket.org:nitred/sparsex.git```
2. On the command line, cd into the sparsex directory
  * ```cd sparsex```
3. On the command line, execute the following:
  * ```python setup.py install```


#### To-Do

* Documentation
  * Latex Folder with .tex file and .pdf1
* Update license
* Make sure its an image when received before proceeding to preprocess.
* Make sure its a 64x64 or a square image?
* Make sure image array is always float once received by server.
* Inplace = True/False option for some of the preprocessing steps. Perhaps for memory conservation.


#### Possible Features

* Logging functionality
* Continuity functionality
  * If there's an error in the code and the process breaks. You can pickup where you left off after you fix the bug.
* On-line Training
  * Train on the go, one image at a time.
  * This might not be the best idea since having bulk training samples instead lets you use the GPU more efficiently.


#### Possible Issues
* Image acquisition, client does not handle well when the server goes down. Client sends a request to an unavailable server and waits for a response but does not recover when the server comes back online.
* Make sure most dtypes are float when converting images from grayscale
* sys.stdout.flush() seems to flush messages from both server and client stuff for some reason. Probably because in this case I am running server and client on my own system and they share stdout. This should be resolved in a more elegant manner. I am unsure why the messages from server would end on client side. Possible bug?
* Which base datatype should be used before extracting patches. Size constraints? Computation constraints and bottlenecks?