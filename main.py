"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import datetime
import socket
import json
import cv2
import numpy as np
from sys import platform


import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

Browser_ON = False
# Get correct CPU extension
if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    CODEC = 0x00000021
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    print("Unsupported OS.")
    exit(1)


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-mod", "--mod", type=str, default='vid',
                        help="Set the mode to img or video")

    return parser



def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client =  mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocessing(frame,net_input_shape):
    """ 
    Per-processing on image as required a by network model for input image
    """
    image_resize = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]),
                            interpolation = cv2.INTER_AREA)
    image = np.moveaxis(image_resize, -1, 0)
    return image



def infer_on_stream(args,client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialize the Inference Engine
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    total = 0
    current = 0
    last = 0

    duration = 0
    in_ROI = [380,450]
    out_ROI = [550, 400]
    state = 'empty'
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, CPU_EXTENSION, num_requests=0)

    # Get a Input blob shape
    net_input_shape = infer_network.get_input_shape()
    print("Input shape: ", net_input_shape)

    ### TODO: Handle the input stream ###
    try:
        cap = cv2.VideoCapture(args.input)
    except FileNotFoundError:
        print("Cannot locate video file: "+ args.input)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    cap.open(args.input)
    _, frame = cap.read()
    fps = 0
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        prev_time = time.time()
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        image = preprocessing(frame,net_input_shape)
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(image, request_id=0)
        inference_time = datetime.timedelta(seconds=time.time() - prev_time)
        #print("Infrence Time: ",inference_time)
        #fps = 1/inference_time
        ### TODO: Wait for the result ###
        if infer_network.wait(request_id=0) == 0:

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request_id=0)
            ### TODO: Extract any desired stats from the results ###
            for box in result[0][0]: # Output shape is 1x1x100x7
                conf = box[2]
                
                if conf >= prob_threshold:
                    x1 = int(box[3] * frame.shape[1])
                    y1 = int(box[4] * frame.shape[0])
                    x2 = int(box[5] * frame.shape[1])
                    y2 = int(box[6] * frame.shape[0])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    
                    
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
                    if x1 < in_ROI[0] and y2 < in_ROI[1]:  
                        if state == "empty":
                            # Count a people
                            current += 1
                            total += 1
                            # Start the timer
                            start_time = time.perf_counter()
                            # Person entered a room 
                            state = 'standing'
                            # Publish people_count messages to the MQTT server
                           
                            client.publish("person", json.dumps({"total": total}))
                                                                 

                    if x1 > out_ROI[0] and y2 < out_ROI[1]:
                        if state == "standing":
                            # Change the state to exit - state  change
                            
                            stop_time = time.perf_counter()
                            duration = int(stop_time - start_time)
                            state = 'empty'
                            # Update average time
    
                            current = 0

                            client.publish("person/duration", json.dumps({"duration": duration}))
                          
                    client.publish("person", json.dumps({"count": current}))
                

            info = [("People in Frame", current) ,             
                    ("People counter", total),]
            position = [(10, 50),(10, 85)]  # position at which writing has to start 

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)                
                cv2.putText(frame, text,position[i],  
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            #frmae = cv2.cvtColor(frmae, cv2.COLOR_RGB2BGR)   
        #     client.publish("person duration", json.dumps({"duration": time_elapsed}))
        #     client.publish("person", json.dumps({"count": person))
        # ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if args.mod == "img":
            cv2.imwrite("Image.jpg", img)
        elif args.mod == "vid":
            pass
            #cv2.imshow('frame', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
            
    cap.release()
    client.disconnect()
    cv2.destroyAllWindows()

    print("FPS: ",fps)
    print("Infrence Time: ",inference_time)
    print("Total people count: ",total)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args,client)


if __name__ == '__main__':
    main()
