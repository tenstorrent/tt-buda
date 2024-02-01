First PyBuda Model
------------------

Below is a "Hello World" example of writing a new PyBuda module and running on a Tenstorrent device.

.. code-block:: python

  import pybuda
  import pybuda.op
  import torch
  from pybuda import PyBudaModule, TTDevice

  # Create a simple matmul transformation module
  class BudaMatmul(PyBudaModule):

    def __init__(self, name):
        super().__init__(name)
        self.weights = pybuda.Parameter(1, 1, 32, 32, requires_grad=False)

    def forward(self, act):
        return pybuda.op.Matmul(self, "matmul", act, self.weights)

  if __name__ == '__main__':
      tt0 = TTDevice("grayskull0")   # Create a Tensorrent device
      lin0 = BudaMatmul("matmul0")   # Instantiate the above model
      tt0.place_module(lin0)         # Place the model on TT device

      act = torch.rand(1, 1, 32, 32) # Feed some random data to the device
      tt0.push_to_inputs(act)

      results = pybuda.run_inference(num_inputs=1, sequential=True)  # Run inference and collect the results


In this code example, we define a BudaMatmul module in a similar way to a PyTorch module. The parameters are defined
in the constructor, and the ops (only one, in this case) in the `forward()` method.

To run this module, we need to create a device to run it on, instantiate the module, and then place it on the device.
Multiple modules can be placed onto one device, and they will run sequentially, however, we only have one module
in this simple example.

Each device contains input queues which can be accesses using the `push_to_inputs` call. In this example, we push
a single random tensor into the device input queue. 

Finally, we're ready to run the device. Using the `run_inference` call, we tell PyBuda to compile the modules, 
start up the processes and devices, and run the given number of iterations through the inputs - a single one, in
this case. Since we didn't provide an explicit output queue for the outputs to be fed into asynchronously, this
call will block until all outputs are available, and will return them in an array of tensors.
  
