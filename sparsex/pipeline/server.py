import zmq
import os, sys, time
from messages_pb2 import Request, Response
from ..preprocessing.preprocessing import Preprocessing
from ..feature_extraction.feature_extraction import SparseCoding
from ..classification.classification import Classifier
import StringIO
from PIL import Image
import numpy as np

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class ServerActions():
    def __init__(self, feature_extraction_model_file=None, classification_model_file=None):
        # initialize preprocessing object
        self.preprocessing = Preprocessing()

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


    def get_image_array_from_byte_data(self, request):
        if request.input_type == Request.IMAGE:
            image_byte_string = request.data
            image_pil = Image.open(StringIO.StringIO(image_byte_string))
            image_array = np.array(image_pil)
            return image_array

        # request.input_type == Request.IMAGE_ARRAY:
        else:
            # get data type
            if request.data_type == Request.UINT8:
                data_type = np.uint8
            elif request.data_type == Request.INT64:
                data_type = np.int64
            else:
                data_type = np.float64

            data_shape = request.data_shape
            image_array = np.frombuffer(request.data, dtype=data_type).reshape(data_shape)
            return image_array


    def get_response_from_array(self, response_type, array):
        # initialize basic response fields
        response = Response()
        response.response_type = response_type

        # update data type
        if array.dtype in [np.uint8]:
            response.data_type = Response.UINT8
        elif array.dtype in [int, np.int32, np.int64]:
            response.data_type = Response.INT64
        else:
            response.data_type = Response.FLOAT64

        # update data shape
        response.data_shape.extend([i for i in array.shape])

        # update data
        response.data = str(np.getbuffer(array))

        return response


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
        elif request.request_type == Request.GET_PREDICTIONS:
            response = self.pipeline_get_predictions(request)
        else:
            response = self.pipeline_none(request)

        # serialize response
        serialized_response = self.get_serialized_response(response)

        # return response_type for any special server actions and return serialized response for the client
        return response, serialized_response


    def pipeline_empty(self, request, additional_information=None):
        # generate empty response type and return
        response = Response()
        response.response_type = Response.EMPTY_RESPONSE

        # add additional information if available
        if additional_information is not None:
            response.additional_information = additional_information

        return response


    def pipeline_error(self, request, additional_information=None):
        # generate error reponse and possibly add addtional information
        response = Response()
        response.response_type = Response.ERROR

        # add additional information if available
        if additional_information is not None:
            response.additional_information = additional_information

        return response


    def pipeline_shutdown(self, request, additional_information=None):
        # generate SHUTDOWN response and return
        response = Response()
        response.response_type = Response.SHUTDOWN

        # add additional information if available
        if additional_information is not None:
            response.additional_information = additional_information

        return response


    def pipeline_get_features(self, request):
        # if input type is unknown then return ERROR response
        if request.input_type == Request.UNKNOWN_INPUT_TYPE:
            return self.pipeline_error(request, additional_information="unknown input type")

        # parse image array from the data bytes of the request
        image_array = self.get_image_array_from_byte_data(request)

        # get features
        whitened_patches = self.preprocessing.get_whitened_patches_from_image_array(image_array)
        pooled_features = self.sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches)

        # construct and return response
        response = self.get_response_from_array(response_type=Response.FEATURES, array=pooled_features)
        
        return response


    def pipeline_get_predictions(self, request):
        # if input type is unknown then return ERROR response
        if request.input_type == Request.UNKNOWN_INPUT_TYPE:
            return self.pipeline_error(request, additional_information="unknown input type")

        # parse image array from the data bytes of the request
        image_array = self.get_image_array_from_byte_data(request)

        # get features
        whitened_patches = self.preprocessing.get_whitened_patches_from_image_array(image_array)
        pooled_features = self.sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches)
        pooled_features = pooled_features.ravel().reshape((1,-1)) # will be removed when pipeline has standardized shapes.
        predictions = self.classifier.get_predictions(pooled_features)

        # construct and return response
        response = self.get_response_from_array(response_type=Response.PREDICTIONS, array=predictions)
        
        return response



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
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.close()
                    context.term()
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
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.close()
                    context.term()
                    sys.exit()

                # required breather, currently handling only 10 requests a second.
                time.sleep(0.1)

        except KeyboardInterrupt:
            # inform client that server is shutting down
            if is_handling_request:
                socket.send('server shutting down!')

            print "\nkeyboard interrupt, server shutting down!"
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
            context.term()
            sys.exit()

        # should normally not come here
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        context.term()



if __name__ == "__main__":
    # create server object
    server = Server(max_requests=0)

    # start server
    server.start()
    