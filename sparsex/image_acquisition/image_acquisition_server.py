### Image Acq server --- receive bytearray/buffer --- convert it --- save it --- check for consistency

import zmq
import sys
import time
import base64
import numpy as np
from StringIO import StringIO
from PIL import Image

class ImageAcquisitionServer:
    def __init__(self, ip="127.0.0.1", port="5556", is_debug=True):
        self.ip = ip
        self.port = port

    def null_callback(self, arg):
        return "null-callback"

    def run_server(self, callback=None):
        ## if no callback is passed
        if callback is None:
            callback = self.null_callback

        ## server socket
        context = zmq.Context()
        socket = context.socket(zmq.REP)

        try:
            socket.bind("tcp://%s:%s" %(self.ip, self.port))
            print "Bound to port..."
            print
        except zmq.ZMQError as err:
            print "ERROR..."
            print err.args
            raise

            
        ## process requests
        while True:
            print "Waiting to receive..."
            sys.stdout.flush()
            msg = socket.recv()
            
            print "Received..."
            print type(msg) # Image as a string message
            
            tempImage_strio = StringIO(msg) # convert string to file object
            print type(tempImage_strio)
            
            tempImage_pil = Image.open(tempImage_strio).convert('L') # convert file object to Grayscale PIL Image
            print type(tempImage_pil)
            
            tempImage = np.array(tempImage_pil.convert('L')).astype(float) # convert PIL Image to numpy array
            print type(tempImage)
            
            result = callback(tempImage)
            
            socket.send(result)
            print
            sys.stdout.flush()
            time.sleep(0.1)


def main():
    run_server()
        
if __name__ == "__main__":
    main()