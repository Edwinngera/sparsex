from ..image_acquisition.image_acquisition_client import ImageAcquisitionClient
from ..image_acquisition.image_acquisition_server import ImageAcquisitionServer
from ..preprocessing.preprocessing import Preprocessing

import threading
import time
import sys, os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def main():
    try:
        print "Creating thread"
        image_filename = os.path.join(THIS_FILE_PATH, "./data/yaleB01_P00A-005E-10_64x64.pgm")
        image_acq_client = ImageAcquisitionClient(ip="127.0.0.1",port="5556")
        image_acq_client_thread = threading.Thread(name='image_acq_client_thread',
                                                   target=image_acq_client.loop_image_request,
                                                   args=(image_filename,))

        print "Starting Client thread"
        image_acq_client_thread.start()

        # get instance of Preprocessing for callback
        preprocessing = Preprocessing()

        print "Start Server"
        image_acq_server = ImageAcquisitionServer()
        image_acq_server.run_server(callback=preprocessing.get_whitened_patches_from_image_array)
        
        
    except KeyboardInterrupt:
        print "\nreceived keyboard interrupt"
        #image_acq_client_thread.join()
        print "Exiting main thread..."
        return


if __name__ == "__main__":
    main()
    