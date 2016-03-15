### Image acquisition client -- read file --- send to server --- wait for classify

import threading
import zmq
import sys
import time
import base64

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
        print "Connecting..."
        socket.connect("tcp://%s:%s" %(self.ip,self.port))
        print "Connected...."
        print

        # send file every few seconds
        while True:
            print "Sending..."
            socket.send(request_data)
            print "Sent..."

            print "Waiting on server..."
            sys.stdout.flush()
            msg = socket.recv()
            print "Received message : \n%s\n" %(msg)

            print "Going to sleep..."
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
        print "Connecting..."
        socket.connect("tcp://%s:%s" %(self.ip,self.port))
        print "Connected...."
        print

        print "Sending Request..."
        socket.send(request_data)
        print "Sent..."

        print "Waiting on server..."
        sys.stdout.flush()
        msg = socket.recv()
        print "Received message : \n%s\n" %(msg)

        return
    

def main():
    image_filename = "../../datasets/yale_face_b_ext_cropped/CroppedYale_64x64/yaleB01/yaleB01_P00A+000E+00.pgm"
    image_acq_client = ImageAcquisitionClient()   
    image_bytearray = image_acq_client.get_image_bytearray(image_filename)
    image_acq_client.loop_request(request_data=image_bytearray)
    return
    

if __name__ == "__main__":
    main()