import zmq
import sys, os, time

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class Client:
    def __init__(self):
        pass

    def send_request(self, ip="127.0.0.1", port="5556", request=""):
        ## Creating client
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
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
    