import zmq
import sys, os, time
from messages_pb2 import Request

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class Client:
    def __init__(self):
        pass


    def create_request(self, request_type, input_type=None, data_type=None, data_shape=None, data=None):
        request = Request()

        # action is required
        request.request_type = request_type

        # only if there is an input_type, will there be any data in the request
        if input_type is not None:
            request.input_type = input_type
            request.data_type = data_type
            request.data_shape = data_shape
            request.data = data

        return request


    def serialize_request(self, request):
        # make sure incoming request is a protobuf message
        assert isinstance(request, Request), "client request is of type {0} instead of Protobuf Request Message".format(type(request))

        # serialize the request and return
        return request.SerializeToString()


    def send_request(self, ip="127.0.0.1", port="5556", request=None):
        # creating zmq client
        context = zmq.Context()
        socket = context.socket(zmq.REQ)

        # if no request is provided, create an empty one
        if request is None:
            request = self.create_request(request_type=Request.NONE)

        # serialize the request
        request = self.serialize_request(request)

        print "client connecting to server at tcp://%s:%s" %(ip,port)
        socket.connect("tcp://%s:%s" %(ip, port))

        print "client connected to server, sending request"
        socket.send(request)

        print "client waiting on response from server"
        sys.stdout.flush()
        response = socket.recv()

        print "client received message :\n", response
    
    print "client shutting down"  
    

if __name__ == "__main__":
    # creating client object
    client = Client()

    # sending a request to the server
    client.send_request()
    