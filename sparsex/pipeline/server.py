import zmq
import os, sys, time

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class ServerActions():
    def __init__(self):
        pass

    def act_on_request(self, request):
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

                # incoming request type
                print "server received request type :\n", type(request)

                # perform actions on the request
                results = self.server_actions.act_on_request(request)
                print "server results :\n", results

                # stop server if requested to shutdown
                if results == 'shutdown':
                    print "server received shutdown, server shutting down!"
                    socket.send("server received shutdown, server shutting down!")
                    socket.close()
                    sys.exit()

                # return results to client
                if results == None:
                    print "server has no results"
                    socket.send("no results")
                else:
                    socket.send(results)

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
    server = Server(max_requests=1)

    # start server
    server.start()
    