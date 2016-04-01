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


    def create_request_from_image_file(self, request_type, image_filename):
        # initialize basic request fields
        request = Request()
        request.request_type = request_type
        request.input_type = Request.IMAGE

        ## adding image byte string as request data
        with open(image_filename, 'r') as image_file:
            request.data = image_file.read()

        return request


    def create_request_from_image_array(self, request_type, image_array):
        # initialize basic request fields
        request = Request()
        request.request_type = request_type
        request.input_type = Request.IMAGE_ARRAY

        # update data type
        if image_array.dtype in [np.uint8]:
            request.data_type = Request.UINT8
        elif image_array.dtype in [int, np.int32, np.int64]:
            request.data_type = Request.INT64
        else:
            request.data_type = Request.FLOAT64

        # update data shape
        request.data_shape.extend([i for i in image_array.shape])

        # update data
        request.data = str(np.getbuffer(image_array))

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

    

if __name__ == "__main__":
    # creating client object
    client = Client()

    # create empty request
    request = Request()
    request.request_type = Request.EMPTY_REQUEST

    # send None/Default Request
    response = client.send_request(request=request)

    # sleep for a few seconds before next request
    time.sleep(3.0)



    # creating client object
    client = Client()

    # request FEATURES from IMAGE FILE
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    request = client.create_request_from_image_file(Request.GET_FEATURES, image_filename)

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
    time.sleep(3.0)



    # creating client object
    client = Client()

    # request FEATURES from IMAGE ARRAY
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    image_pil = Image.open(image_filename)
    image_array = np.array(image_pil)

    # create request from image array
    request = client.create_request_from_image_array(request_type=Request.GET_FEATURES, image_array=image_array)

    # sending request to server
    response = client.send_request(request=request)

    # check if response is correct
    print "response data type :\n", response.data_type
    print "response data shape :\n", response.data_shape
    
    features = np.frombuffer(response.data, dtype='float').reshape(response.data_shape)

    print "response features type :\n", type(features)
    print "response features data type :\n", features.dtype
    print "response features shape :\n", features.shape

    # sleep for a few seconds before next request
    time.sleep(3.0)



    # creating client object
    client = Client()

    # request PREDICTIONS from IMAGE FILE
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    request = client.create_request_from_image_file(Request.GET_PREDICTIONS, image_filename)

    # sending a request to the server
    response = client.send_request(request=request)

    # check if response is correct
    print "response data type :\n", response.data_type
    print "response data shape :\n", response.data_shape
    
    # predcitions = np.frombuffer(response.data, dtype='float').reshape(response.data_shape)
    predcitions = np.frombuffer(response.data, dtype='float')

    print "response predcitions type :\n", type(predcitions)
    print "response predcitions data type :\n", predcitions.dtype
    print "response predcitions shape :\n", predcitions.shape

    # sleep for a few seconds before next request
    time.sleep(3.0)



    # creating client object
    client = Client()

    # request PREDICTIONS from IMAGE ARRAY
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    image_pil = Image.open(image_filename)
    image_array = np.array(image_pil)

    # create request from image array
    request = client.create_request_from_image_array(request_type=Request.GET_PREDICTIONS, image_array=image_array)

    # sending request to server
    response = client.send_request(request=request)

    # check if response is correct
    print "response data type :\n", response.data_type
    print "response data shape :\n", response.data_shape
    
    # predcitions = np.frombuffer(response.data, dtype='float').reshape(response.data_shape)
    predcitions = np.frombuffer(response.data, dtype='float')

    print "response predcitions type :\n", type(predcitions)
    print "response predcitions data type :\n", predcitions.dtype
    print "response predcitions shape :\n", predcitions.shape

    # sleep for a few seconds before next request
    time.sleep(3.0)



    # creating client object
    client = Client()

    # create server SHUTDOWN request
    request = Request()
    request.request_type = Request.SHUTDOWN

    # sending a request to the server
    response = client.send_request(request=request)

    