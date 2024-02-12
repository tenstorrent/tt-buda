PyBuda Training Pipeline
========================

Below is the minimum training pipeline example with a single Grayskull device running a PyBuda module, followed
by a CPU device calculating the loss and backpropagating it to Grayskull.

.. code-block:: python

  import pybuda
  import pybuda.op
  from pybuda import PyBudaModule

  # Create a simple matmul module
  class BudaLinear(PyBudaModule):

      def __init__(self, name):
          super().__init__(name)
          self.weights = pybuda.Parameter(32, 32, requires_grad=True)

      def forward(self, act):
          return pybuda.op.Matmul(self, "matmul", act, self.weights)

  # Create a device and model, and place it 
  tt0 = TTDevice("tt0", devtype=devtype)
  matmul0 = BudaLinear("matmul0")
  tt0.place_module(matmul0)

  act_dim = (1, 1, 32, 32)
  act = torch.rand(*act_dim).half()
  weight = torch.rand(*act_dim).half()
  matmul0.set_parameter("weights",  weight)
  
  # Add a CPU device to calculate loss
  cpu = CPUDevice("cpu0", optimizer_f = None, scheduler_f = None)
  identity = PyTorchModule("matmul0", torch.nn.Identity().half())
  cpu.place_module(identity)
  loss_module = PyTorchModule("l1loss", torch.nn.L1Loss())
  cpu.place_loss_module(loss_module)

  # Create a random target, and push to target queue
  target = torch.rand(32, 32).half()
  cpu.push_to_target_inputs(target)

  # Push inputs, and run
  tt0.push_to_inputs(act)
  pybuda.run_training(epochs=1, steps=1, accumulation_steps=1, micro_batch_count=1)
  

The first part of the code is similar to the :doc:`Hello World<first_model>` inference example - a PyBuda module
is defined and placed onto a Grayskull device.

At the moment, PyBuda modules cannot calculate final loss on device. So, we're going to create a CPU device and
assign an L1Loss module to it. By default, PyBuda will run devices in a pipeline in the order that they were created. An 
:func:`API<pybuda.set_device_pipeline>` exists to explicitly set this order, if needed. 

Then, some target value is needed by the training pipeline to calculate the loss, and the final device will
contain a target queue that we can push target data into. We push a random value in this example.

Finally, an input can be pushed into the first device, and training can be started with `run_training`. Run training
takes a number of parameters to specify loop counts for epochs, batches, and so on, which are further described in the 
API reference for :func:`pybuda.run_training`.

When training is done, PyBuda will automatically shut down all processes and devices. However, a `keep_alive` flag is
available to keep everything ready for another call, so it is possible to call `run_training` multiple times, with
different parameters.

