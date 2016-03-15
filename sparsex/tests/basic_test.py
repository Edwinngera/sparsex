from ..image_acquisition.image_acquisition_client import ImageAcquisitionClient
from ..image_acquisition.image_acquisition_server import ImageAcquisitionServer

import threading
import time
import sys

def main():
    try:
        print "Creating thread"
        image_filename = "/home/nitish/mas_course_ss2015/assignments/sparsex/datasets/yale_face_b_ext_cropped/CroppedYale_64x64/yaleB01/yaleB01_P00A+000E+00.pgm"
        image_acq_client = ImageAcquisitionClient(ip="127.0.0.1",port="5556")
        image_acq_client_thread = threading.Thread(name='image_acq_client_thread',
                                                   target=image_acq_client.loop_image_request,
                                                   args=(image_filename,))

        print "Starting Client thread"
        image_acq_client_thread.start()

        image_acq_server = ImageAcquisitionServer()
        image_acq_server.run_server()
        
        
    except KeyboardInterrupt:
        print "\nreceived keyboard interrupt"
        #image_acq_client_thread.join()
        print "Exiting main thread..."
        return


if __name__ == "__main__":
    main()
    