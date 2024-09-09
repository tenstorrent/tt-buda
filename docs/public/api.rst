API Reference
-------------

Python Runtime API
******************

.. automodule:: pybuda
.. autosummary::

   run_inference
   run_training
   initialize_pipeline
   run_forward
   run_backward
   run_optimizer
   get_parameter_checkpoint
   get_parameter_gradients
   update_device_parameters
   shutdown

.. automodule:: pybuda
   :members: run_inference, run_training, shutdown, initialize_pipeline, run_forward, run_backward, run_optimizer, get_parameter_checkpoint, get_parameter_gradients, update_device_parameters

C++ Runtime API
******************

The BUDA Backend used by Python Runtime can be optionally used stand-alone to run pre-compiled TTI models.

Configuration and Placement
***************************
.. autosummary::

   set_configuration_options
   set_epoch_break
   set_chip_break
   override_op_size
   detect_available_devices

.. automodule:: pybuda
   :members: set_configuration_options, set_epoch_break, set_chip_break, override_op_size, detect_available_devices

Operations
**********

.. currentmodule:: pybuda.op

General
=======

.. autosummary::

   Matmul
   Add
   Subtract
   Multiply
   ReduceSum
   ReduceAvg
   Constant
   Identity
   Buffer

.. automodule:: pybuda.op
   :members: Matmul, Add, Subtract, Multiply, ReduceSum, ReduceAvg, Constant, Identity, Buffer

Transformations
===============

.. autosummary::

   HSlice
   VSlice
   HStack
   VStack
   Reshape
   Index
   Select
   Pad
   Concatenate
   BinaryStack
   Heaviside

.. automodule:: pybuda.op
   :members: HSlice, VSlice, HStack, VStack, Reshape, Index, Select, Pad, Concatenate, BinaryStack, Heaviside
    
Activations
===========

.. autosummary::

   Relu
   Gelu
   Sigmoid

.. automodule:: pybuda.op
   :members: Relu, Gelu, Sigmoid

Math
====

.. autosummary::

   Exp
   Reciprocal
   Sqrt
   Log
   Abs
   Clip
   Max
   Argmax

.. automodule:: pybuda.op
   :members: Exp, Reciprocal, Sqrt, Log, Abs, Clip, Max, Argmax

Convolutions
============
    
.. autosummary::
   Conv2d
   Conv2dTranspose
   MaxPool2d
   AvgPool2d

.. automodule:: pybuda.op
   :members: Conv2d, Conv2dTranspose, MaxPool2d, AvgPool2d

NN
==

.. currentmodule:: pybuda.op.nn

.. autosummary::

   Softmax
   Layernorm

.. automodule:: pybuda.op.nn
   :members: Softmax, Layernorm

Module Types
************

.. currentmodule:: pybuda
.. autosummary::

   Module
   PyTorchModule
   TFModule
   OnnxModule
   PyBudaModule

.. autoclass:: pybuda::Module
   :members:

.. autoclass:: pybuda::PyTorchModule
   :members:

.. autoclass:: pybuda::TFModule
   :members:

.. autoclass:: pybuda::OnnxModule
   :members:

.. autoclass:: pybuda::PyBudaModule
   :members:

Device Types
************

.. autosummary::

   Device
   CPUDevice
   TTDevice

.. autoclass:: pybuda::Device
   :members:

.. autoclass:: pybuda::CPUDevice
   :members:

.. autoclass:: pybuda::TTDevice
   :members:

Miscellaneous
*************

.. autosummary::

   DataFormat
   MathFidelity

.. autoclass:: pybuda::DataFormat
   :members:

.. autoclass:: pybuda::MathFidelity
   :members:

