import zmq
import os, sys
import time

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class ServerActions():
    def __init__(self):
        pass

    def act_on_message(self, message):
        pass


class Server:
    def __init__(self, ip="127.0.0.1", port="5556", max_requests=0):
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

        # flag to check if a request is being handling for clean closing
        is_handling_request = False

        # handling incoming requests
        print "server start"
        try:
            while True:
                print "server waiting for request"
                sys.stdout.flush()
                message = socket.recv()
                is_handling_request = True

                # incoming message type
                print "server received message type :\n", type(message)

                # perform actions on the message
                results = self.server_actions.act_on_message(message)
                print "server results :\n", results

                # stop server if results are empty or they ask for server to shutdown
                if (results == 'shutdown') or (results is None):
                    print "results are shutdown/None, server shutting down!"
                    socket.send('server shutting down!')
                    socket.close()
                    break

                # return results to client
                socket.send(results)
                is_handling_request = False

                # increment result count
                request_count += 1

                # compare request count with max_requests
                if self.max_requests <= 0: # if 0/-ve, loop forever
                    pass
                elif request_count >= self.max_requests:
                    print "max_requests reached, server shutting down!"
                    socket.close()

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
    server = Server(max_requests=1)

    # start server
    server.start()
    