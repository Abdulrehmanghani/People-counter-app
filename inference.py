#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.infer_network = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None, num_requests=0):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the pluginn
        self.infer_network = IECore()
        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.infer_network.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
    
        ### TODO: Check for supported layers ###
        supported_layers = self.infer_network.query_network(network=self.network, device_name="CPU")
 
        # know if anything is missing. Exit the program, if so.
        # unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        # if len(unsupported_layers) != 0:
        #     print("Unsupported layers found: {}".format(unsupported_layers))
        #     print("Check whether extensions are available to add to IECore.")
        #     exit(1)

        ### TODO: Add any necessary extensions ###
         # Load the IENetwork into the plugin
        self.exec_network = self.infer_network.load_network(self.network, device, num_requests=num_requests)
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return  self.network.inputs[self.input_blob].shape

    def exec_net(self, image, request_id):
        ### TODO: Start an asynchronous request ###
        self.exec_network.start_async(request_id=request_id, 
            inputs={self.input_blob: image})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        status = self.exec_network.requests[request_id].wait(-1)
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self, request_id):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[request_id].outputs[self.output_blob]
