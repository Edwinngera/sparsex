### Image acquisition client -- read file --- send to server --- wait for classify

import threading
import zmq
import sys, os
import time
import base64

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class ImageAcquisitionClient:
    def __init__(self, ip="127.0.0.1", port="5556", is_debug=True):
        self.ip = ip
        self.port = port
        self.is_debug = is_debug

    def get_image_bytearray(self, image_filename):
        with open(image_filename,'rb') as image_file:
            image_bytearray = bytearray(image_file.read())
        return image_bytearray

    def loop_request(self, request_data="", loop_delay_time=10.0):
        ## Creating client
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        print "Clinet connecting..."
        socket.connect("tcp://%s:%s" %(self.ip,self.port))
        print "Client connected...."
        print

        # send file every few seconds
        while True:
            print "Client sending..."
            socket.send(request_data)
            print "Client sent..."

            print "Client waiting on server..."
            sys.stdout.flush()
            msg = socket.recv()
            print "Client received message : \n%s\n" %(msg)

            print "Client going to sleep..."
            print
            sys.stdout.flush()
            time.sleep(loop_delay_time)

        return

    def loop_image_request(self, image_filename=None):
        if image_filename is not None:
            image_bytearray = self.get_image_bytearray(image_filename)
            self.loop_request(request_data=image_bytearray)
        return
            
    def send_request(self, request_data=""):
        ## Creating client
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        print "Client connecting..."
        socket.connect("tcp://%s:%s" %(self.ip,self.port))
        print "Client connected...."
        print

        print "Client sending Request..."
        socket.send(request_data)
        print "Client sent..."

        print "Client waiting on server..."
        sys.stdout.flush()
        msg = socket.recv()
        print "Clinet received message : \n%s\n" %(msg)

        return
    

def main():
    image_filename = os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm")
    image_acq_client = ImageAcquisitionClient()   
    image_bytearray = image_acq_client.get_image_bytearray(image_filename)
    image_acq_client.loop_request(request_data=image_bytearray)
    return
    

if __name__ == "__main__":
    main()
    