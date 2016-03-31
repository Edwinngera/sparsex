import zmq
import sys, os, time
from messages_pb2 import Request, Response
from PIL import Image
import numpy as np

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class Client:
    def __init__(self):
        pass

    def create_request(self, request_type, input_type=None, data_type=None, data_shape=None, data=None):
        # create empty request
        request = Request()

        # request_type is required
        request.request_type = request_type

        # only if there is an input_type, will there be any data in the request
        if input_type is not None:
            request.input_type = input_type
            request.data_type = data_type
            request.data_shape = data_shape
            request.data = data

        return request


    def get_serialized_request(self, request):
        # make sure incoming request is a protobuf message
        assert isinstance(request, Request), "client request is of type {0} instead of Protobuf Request Message".format(type(request))

        # serialize the request and return
        return request.SerializeToString()


    def get_deserialized_response(self, serialized_response):
        # de-serialize response object
        response = Response()
        response.ParseFromString(serialized_response)
        return response


    def send_request(self, ip="127.0.0.1", port="5556", request=None):
        # creating zmq client
        context = zmq.Context()
        socket = context.socket(zmq.REQ)

        # if no request is provided, create an empty one
        if request is None:
            request = self.create_request(request_type=Request.EMPTY_REQUEST)

        # serialize the request
        serialized_request = self.get_serialized_request(request)

        print "client connecting to server at tcp://%s:%s" %(ip,port)
        socket.connect("tcp://%s:%s" %(ip, port))

        print "client connected to server"
        print "client sending request :\n", request
        socket.send(serialized_request)

        print "client waiting on serialized response from server"
        sys.stdout.flush()
        serialized_response = socket.recv()

        # deserialize response
        response = self.get_deserialized_response(serialized_response)

        print "client received response :\n", response

        return response
    
    print "client shutting down"  
    

if __name__ == "__main__":
    # creating client object
    client = Client()

    # create empty request
    request = Request()
    request.request_type = Request.EMPTY_REQUEST

    # send None/Default Request
    response = client.send_request(request=request)

    # sleep for a few seconds before next request
    time.sleep(4.0)



    # create server FEATURES request with Image
    request = Request()
    request.request_type = Request.GET_FEATURES
    request.input_type = Request.IMAGE

    ## adding image byte string as request data
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    with open(image_filename, 'r') as image_file:
        request.data = image_file.read()

    # sending a request to the server
    response = client.send_request(request=request)

    # check if response is correct
    print "response data type :\n", response.data_type
    print "response data shape :\n", response.data_shape
    
    features = np.frombuffer(response.data, dtype='float').reshape(response.data_shape)

    print "response features type :\n", type(features)
    print "response features data type :\n", features.dtype
    print "response features shape :\n", features.shape

    # sleep for a few seconds before next request
    time.sleep(4.0)



    # create server SHUTDOWN request
    request = Request()
    request.request_type = Request.SHUTDOWN

    # sending a request to the server
    response = client.send_request(request=request)

    