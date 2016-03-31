import zmq
import os, sys, time
from messages_pb2 import Request, Response
from ..feature_extraction.feature_extraction import SparseCoding
from ..classification.classification import Classifier

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class ServerActions():
    def __init__(self, feature_extraction_model_file=None, classification_model_file=None):
        # use default feature extraction model if none given
        if feature_extraction_model_file is None:
            feature_extraction_model_file = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/trained_feature_extraction_test_model.pkl"))

        # create instance of feature extraction module
        self.sparse_coding = SparseCoding(model_filename=feature_extraction_model_file)

        # use default classification model if none given
        if classification_model_file is None:
            classification_model_file = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/trained_classification_test_model.pkl"))

        # create instance of classification module
        self.classifier = Classifier(model_filename=classification_model_file)


    def get_serialized_response(self, response):
        # serialize the response and return
        return response.SerializeToString()


    def get_deserialized_request(self, serialized_request):
        # de-serialize request object
        request = Request()
        request.ParseFromString(serialized_request)
        return request


    def act_on_request(self, serialized_request):
        # make sure incoming request is string before we can de-serialize it
        assert isinstance(serialized_request, str), "client request is of type {0} instead of String".format(type(serialized_request))

        request = self.get_deserialized_request(serialized_request)
        print "server received request :\n", request

        # handle requests / get response
        if request.request_type == Request.EMPTY_REQUEST:
            response = self.pipeline_empty(request)
        elif request.request_type == Request.SHUTDOWN:
            response = self.pipeline_shutdown(request)
        elif request.request_type == Request.GET_FEATURES:
            response = self.pipeline_get_features(request)
        else:
            response = self.pipeline_none(request)

        # serialize response
        serialized_response = self.get_serialized_response(response)

        # return response_type for any special server actions and return serialized response for the client
        return response, serialized_response


    def pipeline_empty(self, request):
        # generate empty response type and return
        response = Response()
        response.response_type = Response.EMPTY_RESPONSE
        return response


    def pipeline_error(self, request, additional_information=None):
        # generate error reponse and possibly add addtional information
        response = Response()
        response.response_type = Response.ERROR

        # add additional information if available
        if additional_information is not None:
            response.additional_information = additional_information

        return response


    def pipeline_shutdown(self, request):
        # generate SHUTDOWN response and return
        response = Response()
        response.response_type = Response.SHUTDOWN
        return response


    def pipeline_get_features(self, request):
        # if input type is unknown then return ERROR response
        if request.input_type == Request.UNKNOWN_INPUT_TYPE:
            return self.pipeline_error(request, additional_information="unknown input type")
        else:
            return self.pipeline_empty(request)



class Server:
    def __init__(self, ip="127.0.0.1", port="5556", max_requests=0, feature_extraction_model_file=None, classification_model_file=None):
        self.ip = ip
        self.port = port
        self.max_requests = max_requests
        self.server_actions = ServerActions()

    def start(self):
        ## server socket
        context = zmq.Context()
        socket = context.socket(zmq.REP)

        try:
            socket.bind("tcp://%s:%s" %(self.ip, self.port))
            print "server bound to tcp://%s:%s" %(self.ip, self.port)

        except zmq.ZMQError as err:
            raise zmq.ZMQError(msg="server side ZMQError when binding to port. " \
                              + "Error possibley due to port already being used.\n" \
                              + "err.args :\n" + str(err.args))
  
        # keep request count and stop when greater than max_requests. if max_requests = 0/-ve, loop forever
        request_count = 0

        # flag to check if a client request is being handled, so that socket can be cleanly terminated if interrupted.
        is_handling_request = False

        # handling incoming requests
        print "server start"
        try:
            while True:
                print "server waiting for request : {0}".format(request_count + 1)
                sys.stdout.flush()
                request = socket.recv()

                # set flag
                is_handling_request = True

                # perform actions on the request
                response, serialized_response = self.server_actions.act_on_request(request)
                print "server sending response :\n", response

                # stop server if requested to shutdown
                if response.response_type == Response.SHUTDOWN:
                    print "server received shutdown, server shutting down!"
                    socket.send(serialized_response)
                    socket.close()
                    sys.exit()

                # send serialized response to client
                socket.send(serialized_response)

                # un-set flag
                is_handling_request = False

                # increment result count
                request_count += 1

                # compare request count with max_requests
                if self.max_requests <= 0: # if 0/-ve, loop forever
                    pass
                elif request_count >= self.max_requests:
                    print "max_requests reached, server shutting down!"
                    socket.close()
                    sys.exit()

                # required breather, currently handling only 10 requests a second.
                time.sleep(0.1)

        except KeyboardInterrupt:
            # inform client that server is shutting down
            if is_handling_request:
                socket.send('server shutting down!')

            print "\nkeyboard interrupt, server shutting down!"
            socket.close()
            sys.exit()


if __name__ == "__main__":
    # create server object
    server = Server(max_requests=3)

    # start server
    server.start()
    