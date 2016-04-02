from ..pipeline.server import Server
from ..pipeline.client import Client
from ..pipeline.messages_pb2 import Request, Response
from threading import Thread
import sys, os
import numpy as np
from PIL import Image

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def test_pipeline_empty():
    print "##### Testing Pipeline Empty Request"
    server = Server(max_requests=1)
    server_thread = Thread(target=server.start,args=())
    server_thread.start()
    client = Client()
    request = Request()
    request.request_type = Request.EMPTY_REQUEST
    response = client.send_request(request=request)

    if response:
        if response.response_type == Response.EMPTY_RESPONSE:
            print "correct response type"
        else:
            print "wrong response type"
    else:
        print "empty response, possible server error / server unavailable"

    server_thread.join()



def test_pipeline_shutdown():
    print "##### Testing Pipeline Empty Request"
    server = Server(max_requests=1)
    server_thread = Thread(target=server.start,args=())
    server_thread.start()
    client = Client()
    request = Request()
    request.request_type = Request.SHUTDOWN
    response = client.send_request(request=request)

    if response:
        if response.response_type == Response.SHUTDOWN:
            print "correct response type"
        else:
            print "wrong response type"
    else:
        print "empty response, possible server error / server unavailable"

    server_thread.join()



def test_pipeline_get_features_from_image_file(image_filename=None):
    if image_filename is None:
        image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    print "##### Testing Pipeline Get Features From Image File"
    server = Server(max_requests=1)
    server_thread = Thread(target=server.start,args=())
    server_thread.start()
    client = Client()
    request = client.create_request_from_image_file(Request.GET_FEATURES, image_filename)
    response = client.send_request(request=request)

    if response:
        if response.response_type == Response.FEATURES:
            features = np.frombuffer(response.data, dtype='float').reshape(response.data_shape)
            print "response features type :\n", type(features)
            print "response features data type :\n", features.dtype
            print "response features shape :\n", features.shape
            print "correct response type"
        else:
            print "wrong response type"
    else:
        print "empty response, possible server error / server unavailable"

    server_thread.join()



def test_pipeline_get_features_from_image_array(image_array=None):
    if image_array is None:
        image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
        image_pil = Image.open(image_filename)
        image_array = np.array(image_pil)
    print "##### Testing Pipeline Get Features From Image Array"
    server = Server(max_requests=1)
    server_thread = Thread(target=server.start,args=())
    server_thread.start()
    client = Client()
    request = client.create_request_from_image_array(Request.GET_FEATURES, image_array)
    response = client.send_request(request=request)

    if response:
        if response.response_type == Response.FEATURES:
            features = np.frombuffer(response.data, dtype='float').reshape(response.data_shape)
            print "response features type :\n", type(features)
            print "response features data type :\n", features.dtype
            print "response features shape :\n", features.shape
            print "correct response type"
        else:
            print "wrong response type"
    else:
        print "empty response, possible server error / server unavailable"

    server_thread.join()



def test_pipeline_get_predictions_from_image_file(image_filename=None):
    if image_filename is None:
        image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    print "##### Testing Pipeline Get Predictions From Image File"
    server = Server(max_requests=1)
    server_thread = Thread(target=server.start,args=())
    server_thread.start()
    client = Client()
    request = client.create_request_from_image_file(Request.GET_PREDICTIONS, image_filename)
    response = client.send_request(request=request)

    if response:
        if response.response_type == Response.PREDICTIONS:
            predictions = np.frombuffer(response.data, dtype='float').reshape(response.data_shape)
            print "response predictions type :\n", type(predictions)
            print "response predictions data type :\n", predictions.dtype
            print "response predictions shape :\n", predictions.shape
            print "correct response type"
        else:
            print "wrong response type"
    else:
        print "empty response, possible server error / server unavailable"

    server_thread.join()



def test_pipeline_get_predictions_from_image_array(image_array=None):
    if image_array is None:
        image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
        image_pil = Image.open(image_filename)
        image_array = np.array(image_pil)
    print "##### Testing Pipeline Get Features From Image Array"
    server = Server(max_requests=1)
    server_thread = Thread(target=server.start,args=())
    server_thread.start()
    client = Client()
    request = client.create_request_from_image_array(Request.GET_PREDICTIONS, image_array)
    response = client.send_request(request=request)

    if response:
        if response.response_type == Response.PREDICTIONS:
            predictions = np.frombuffer(response.data, dtype='float').reshape(response.data_shape)
            print "response predictions type :\n", type(predictions)
            print "response predictions data type :\n", predictions.dtype
            print "response predictions shape :\n", predictions.shape
            print "correct response type"
        else:
            print "wrong response type"
    else:
        print "empty response, possible server error / server unavailable"

    server_thread.join()



if __name__ == "__main__":
    # test empty pipeline
    test_pipeline_empty()

    # test server shutdown request pipeline
    test_pipeline_shutdown()

    # test get features from image file
    test_pipeline_get_features_from_image_file()
    
    # test get features from image array
    test_pipeline_get_features_from_image_array()

    # test get predictions from image file
    test_pipeline_get_predictions_from_image_file()

    # test get predictions from image array
    test_pipeline_get_predictions_from_image_array()
