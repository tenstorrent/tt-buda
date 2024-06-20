User Guide
==========


Quick Start
-----------

Compiling and running a PyBuda workload is as easy as:

#. Downloading an off the shelf model from huggingface
#. Wrapping it in a PyBuda module
#. Create a tenstorrent device
#. Running a program on the currently bound device

.. code-block:: python

  import pybuda
  import torch
  from transformers import BertModel, BertConfig

  # Guard in the main module to avoid creating subprocesses recursively.
  if __name__ == "__main__":
      # Download the model from huggingface
      model = BertModel.from_pretrained("bert-base-uncased")

      # Wrap the pytorch model in a PyBuda module wrapper
      module = pybuda.PyTorchModule("bert_encoder", model.encoder)

      # Create a tenstorrent device
      tt0 = pybuda.TTDevice(
          "tt0",
          module=module,
          arch=pybuda.BackendDevice.Wormhole_B0,
          devtype=pybuda.BackendType.Silicon,
      )

      # Create an input tensor
      seq_len = 128
      input = torch.randn(1, seq_len, model.config.hidden_size)

      # Compile and run inference
      output_queue = pybuda.run_inference(inputs=[input])
      print(output_queue.get())


Framework Support
-----------------

Pybuda itself is a standalone ML framework and has an API heavily inspired by Pytorch.  That said, it is often more convenient to run models that have already been written using another major framework.  This is why we support a Pybuda backend for TVM which allows many popular frameworks to target our pybuda compiler. We support many major frameworks that TVM supports including:

.. list-table:: Framwork Support Matrix
  :header-rows: 1

  * - Framework
    - PyBuda Wrapper Class
    - Support
  * - Pybuda
    - ``pybuda.PyBudaModule``
    - Native Support
  * - Pytorch
    - ``pybuda.PyTorchModule``
    - Supported
  * - Tensorflow
    - ``pybuda.TFModule``
    - Supported
  * - Jax
    - ``pybuda.JaxModule``
    - Supported via ``jax2tf``
  * - Tensorflow Lite
    - ``pybuda.TFLiteModule``
    - Preliminary Support
  * - Onnx
    - ``pybuda.OnnxModule``
    - Preliminary Support
  * - Tensorflow GraphDef
    - ``pybuda.TFGraphDefModule``
    - Preliminary Support


PyBuda Introduction
-------------------

A typical PyBuda flow has 4 steps:

#. Define devices to run workload on
#. Place modules from the workload onto devices
#. Run workload
#. Retrieve results

PyBuda API and workflow is flexible enough that some of these steps can be merged, reordered, or skipped altogether, however it helps to work through this basic workflow to understand PyBuda concepts.

Devices
*******

PyBuda makes it easy to distribute a workload onto a heterogeneous set of devices available to you. This can be one or more 
Tenstorrent devices, CPUs, or GPUs. Each device that will be used to run your workflow needs to be declared by creating the appropriate
device type and giving it a unique name:

.. code-block:: python

   tt0 = TTDevice("tt0")
   cpu0 = CPUDevice("cpu0")

By default, the first available device of the appropriate type will be allocated, but out-of-order allocations can be done
using the the optional `id` field.

The order in which the devices are created is important, because they are automatically connected into a pipeline of devices, 
which will be explained in mode detail :ref:`further down<Using Multiple Tenstorrent Devices>`.

Each TTDevice can represent more than one hardware device, as described in :ref:`multi-device section<Multiple Devices>`.

Modules
*******

A typical ML/AI workload consists of modules. PyBuda supports modules that are written in various :ref:`frameworks<Framework Support>`, such
as PyTorch, Tensorflow, and so on, as well as modules written in native PyBuda. Currently, all modules run on Tenstorrent devices, but 
CPU and GPU devices can't run PyBuda modules.

To run a module on a device, it needs to be "placed" on it

.. code-block:: python

   tt0.place_module(mod)

This tells PyBuda that module ``mod`` needs to be compiled and executed on device ``tt0``. In this case, ``mod`` is a native PyBuda module. To
similarly place a PyTorch module onto a Tenstorrent device, the module must be wrapped in a :py:class:`PyTorchModule<pybuda.PyTorchModule>` wrapper:

.. code-block:: python

   tt0.place_module(PyTorchModule("linear", torch.nn.Linear(64, 64)))

Run Workload
************

The running of the workload involves two parts, usually done in parallel - data feeding, and actual running. 

To feed data into the device pipeline, simply push inputs into the first device:

.. code-block:: python

   tt0.push_to_inputs( (input0, input1) )

This can be done in a separate thread, through a data-loader, or any other common way of feeding data into a workload pipeline. It is important
that at least one set of inputs has been pushed into the first device before attempting to compile and run the workload, because PyBuda compiler
requires a set of sample inputs to determine graph shapes, data formats, and other properties.

PyBuda provides all-in-one APIs for compiling and running workloads, :py:func:`run_inference<pybuda.run_inference>` and 
:py:func:`run_training<pybuda.run_training>`, which both return a queue in which the results will be automatically pushed. 
For inference, and simple training setups, this is the simplest way to get up and running.

Alternatively, the models can be compiled in a separate step, using the :py:func:`initialize_pipeline<pybuda.initialize_pipeline>` call, 
which optionally takes sample inputs, if none have been pushed into the first device. Once the compilation has completed, the user 
can run :py:func:`run_forward<pybuda.run_forward>` pass through the pipeline for inference, or a loop of 
:py:func:`run_forward<pybuda.run_forward>`, :py:func:`run_backward<pybuda.run_backward>`, and :py:func:`run_optimizer<pybuda.run_optimizer>` 
calls to manually implement a training loop:

.. code-block:: python

    for step in range(1):
        for acc_step in range(1):

            pybuda.run_forward(input_count = 1)
            pybuda.run_backward(input_count = 1, zero_grad = (acc_step == 0))

        pybuda.run_optimizer(checkpoint=True)

CPU Fallback
************

If there are operators in the workload that are unsupported by PyBuda, the user can create a CPUDevice and place module containing those 
operators onto that CPUDevice. If enabled, PyBuda is capable of doing this automatically.

If a TTDevice contains unsupported operators, during compilation, the device will be split into multiple devices (TTDevice and CPUDevice). If
the CPUDevice is at the front of the pipeline (i.e. the unsupported ops are in the first half of the graph), any inputs pushed to the TTDevice
will be redirected to the correct CPUDevice. 

To enable CPU fallback:

.. code-block:: python

   compiler_cfg = pybuda.config._get_global_compiler_config()
   compiler_cfg.enable_tvm_cpu_fallback = True

Then place the module as normal:

.. code-block:: python

   tt0.place_module(PyTorchModule("workload", module_with_unsupported_ops))

Push inputs and run:

.. code-block:: python

   for i in range(5):
      tt0.push_to_inputs((input_tokens))
      output_q = pybuda.run_inference()
      output = output_q.get()

Results
*******

Typical inference runs produce inference outputs, and training runs produce parameter checkpoints. By default, PyBuda APIs for these workflows
create multiprocessing queues to which resulting tensors will be pushed to. Optionally, user can provide their own queues.

Reading these results is then as simple as popping data from the queues:

.. code-block:: python

    output_q = pybuda.run_inference(PyBudaTestModule("run_direct"), inputs=[(input1, input2)])
    output = output_q.get()

Output queues hold PyBuda tensors. For each PyBuda tensor, user can convert it back to original framework tensor using:

.. code-block:: python

    output_in_tf = output_q[0].to_framework("tensorflow")

Advanced training scenarios sometimes require accumulated gradients to be retrieved and analyzed. For those cases, PyBuda provides an 
:py::func:`API<pybuda.get_parameter_gradients>` that retrieves a dictionary of all currently accumulated gradients on a device. This can be used to 
debug or analyze data, or even run a manual optimizer and push new weights onto the device.

Saving and Loading Models
-------------------------

In a simple training workflow, a checkpoint interval can be set, and every N optimization steps the current device state will be pushed into 
the checkpoint queue. For more advanced use cases, a manual checkpoint can be retrieved using 
:py:func:`get_parameter_checkpoint<pybuda.get_parameter_checkpoint>` API, which returns a dictionary of all parameters on a 
device and their current values.

Such a dictionary can also be pushed back onto the device using :py:func:`update_device_parameters<pybuda.update_device_parameters>`.


TensTorrent Device Image (TTI): Saving/Loading
**********************************************

A Tenstorrent Image (TTI) is a standalone archive file that captures the entire compiled state of a 
model. The contents of the archive include device configuration, compiler configuration, compiled model artifacts,
backend build files (e.g. overlay and risc binaries), model parameter tensors. There can be multiple advantages
with leveraging the usage of a TTI archive:

1) Offline target compilation of models on arbitrary device targets (i.e. target device does not have to be present/available on the machine to compile and save a TTI).
2) Loading a TTI archive allows the user to skip any long front-end and backend compilations of models onto the device
   and directly begin executing the graph/module that was packaged in the `*.tti` after pushing inputs to queues.
3) TTI archives can be shared and loaded across different machines and environments.
4) When we save a TTI archive, we can configure the serialization format for the model parameters. This can be useful for
   scenarios where the user wants to save the model parameters in a tilized-binary format to avoid tilizing during model inference.
   By default the serialization format is pickle. To configure for alternate serialization formats, the user can set either: 
   `PYBUDA_TTI_BACKEND_FORMAT=1` or `PYBUDA_TTI_BACKEND_TILIZED_FORMAT=1` environment variables.

For example, from a machine without a silicon device, we can save a TTI archive intended to be deployed on a silicon device.
We need to configure the device type and architecture of the target device and compile the model to a TTI archive.
This can be done by invoking the `compile_to_image` method on  :py:class:`TTDevice<pybuda.TTDevice>`. 

.. code-block:: python

    tt0 = pybuda.TTDevice(
      name="tt0",
      arch=BackendDevice.Wormhole_B0,
      devtype=BackendType.Silicon
    )
    tt0.place_module(...)
    device_img: TTDeviceImage = tt0.compile_to_image(
        img_path="device_images/tt0.tti",
        training=training,
        sample_inputs=(...),
    )


This will create the archive file `device_images/tt0.tti`. The contents of a TTI file will contain:

.. code-block::

    /unzipped_tti_directory
    ├── device.json # Device state and compiled model metadata
    ├── <module-name>.yaml # netlist yaml
    ├── compile_and_runtime_config.json # compiler and runtime configurations
    ├── backend_build_binaries # backend build binaries
    │   ├── device_desc.yaml
    │   ├── cluster_desc.yaml
    │   ├── brisc
    │   ├── erisc
    │   ├── nrisc
    │   ├── hlks 
    │   ├── epoch_programs
    ├── tensors # directory containing serialized tensors
    ├── module_files # Python file containing the PybudaModule of the model

To load the TTI archive and inspect the contents:

.. code-block:: python

    device_img: TTDeviceImage = TTDeviceImage.load_from_disk("device_images/tt0.tti")

The :py:class:`TTDeviceImage<pybuda.TTDeviceImage>::info()` method provides a summary of contents of the TTI:

.. code-block:: python

    device_img.info()

.. code-block:: yaml

      Image Info...
      - Version Info:
            - pybuda_version: 0.1.220624+dev.f63c9d32
            - pybuda_commit: 7def2987
            - buda_backend_commit: f2fd0fa3
      - Device Name: tt0

      Device Info...
      - arch: BackendDevice.Grayskull
      - chip_ids: [0]
      - backend device type: BackendType.Silicon
      - grid size: [10, 12]
      - harvested rows: [0]

      Compilation Graph State...
      - training: False
      - ordered input shapes: [[1, 128, 128], [1, 1, 128, 128]]
      - ordered targets shapes: []


We can now configure :py:class:`TTDevice<pybuda.TTDevice>` by using our image object and execute directly on device:

.. code-block:: python

    img = TTDeviceImage.load_from_disk(img_path="device_images/tt0.tti")
    device = TTDevice.load_image(img=img)

    inputs = [torch.rand(shape) for shape in img.get_input_shapes()] # create tensors using shape info from img
    device.push_to_inputs(inputs) # push newly created input activation tensors to device
    output_q = pybuda.run_inference()


Create TTI: Targeting Supported Silicon Devices
***********************************************

In the example above, we saved a TTI file targeting a silicon device with default configuration (unharvested). There
are also convenience labels available that can be used to target specific silicon devices in our supported product spec.
The current support available is: {gs_e150, gs_e300, wh_n150, wh_n300}.

To target a specific silicon device, we can set the device type and architecture using :py:func:`set_configuration_options<pybuda.set_configuration_options>`.


.. code-block:: python

    pybuda.set_configuration_options(device_config="wh_n150")

    tt0 = pybuda.TTDevice(
      name="tt0",
      arch=BackendDevice.Wormhole_B0,
      devtype=BackendType.Silicon
    )
    tt0.place_module(...)
    device_img: TTDeviceImage = tt0.compile_to_image(
        img_path="device_images/tt0.tti",
        training=training,
        sample_inputs=(...),
    )


Create TTI: Targeting Custom Row-Harvested Silicon Devices
*********************************************************

We can also save a TTI file targeting a machine with silicon devices with harvested rows offline. 
The only difference from the above is we need to manually induce the harvested rows before saving TTI. 

We can set the harvested rows by invoking  :py:func:`set_configuration_options<pybuda.set_configuration_options>` with `harvested_rows` argument.
 
.. code-block:: python

    pybuda.set_configuration_options(harvested_rows=[1,11]) #manually harvest row 1 and 11 

    tt0 = pybuda.TTDevice("tt0",arch=BackendDevice.Grayskull, devtype=BackendType.Silicon)
    tt0.place_module(...)
    device_img: TTDeviceImage = tt0.compile_to_image(
        img_path="device_images/tt0.tti",
        training=training,
        sample_inputs=(...),
    )

The code snippet creates the TTI file targeting silicon devices with row 1 and 11 harvested.
Accordingly, part of the TTI file slightly changes as well:

.. code-block:: yaml
 
      Device Info...
      - arch: BackendDevice.Grayskull
      - chip_ids: [0]
      - backend device type: BackendType.Silicon
      - grid size: [8, 12]    # 2 rows are harvested from 10 rows
      - harvested rows: 2050  # indicates row 1 and 11 are harvested (in binary, 100000000010)
 
  
Note that only rows 1-5 and 7-11 are harvestable, and TTI loading will raise an error if the manually harvested rows in TTI does not match with that of the loaded silicon device.


Create TTI: Targeting Custom Device Descriptor
**************************************************

We can also save a TTI file targeting a machine with silicon devices with custom device descriptor (specified with file-path).
This can be done by setting the device descriptor using :py:func:`set_configuration_options<pybuda.set_configuration_options>` with `backend_device_descriptor_path` argument.
 
.. code-block:: python

    pybuda.set_configuration_options(backend_device_descriptor_path="<device-descriptor-path>/wormhole_b0_4x6.yaml")

    tt0 = pybuda.TTDevice("tt0",arch=BackendDevice.Wormhole_B0, devtype=BackendType.Silicon)
    tt0.place_module(...)
    device_img: TTDeviceImage = tt0.compile_to_image(
        img_path="device_images/tt0.tti",
        training=training,
        sample_inputs=(...),
    )

The device-descriptor used during the offline compilation process will be embedded in the TTI-archive.
This device-descriptor will be used to configure the device during the TTI-loading process.


Embedded TTI Loading
********************

Here's an example of loading a generic TTI model from C++ for environments that do not have a packaged Python interpreter.

.. code-block:: cpp

  #include <iostream>
  #include <memory>
  #include <vector>
  #include <experimental/filesystem>

  #include "tt_backend.hpp"
  #include "tt_backend_api.hpp"
  #include "tt_backend_api_types.hpp"
  #include "io_utils.h"

  namespace fs = std::experimental::filesystem; 

  int main(int argc, char **argv) {

      if (argc <= 1) {
          throw std::runtime_error("TTI path not specified on the command line");
      }
      else if (argc > 3) {
          throw std::runtime_error("Incorrect number of arguments specified to inference harness. Supported args: TTI_PATH NUM_INFERENCE_LOOPS");
      }

      // Define path to pre-compiled model and output artifacts
      std::string output_path = "tt_build/test_standalone_runtime";
      fs::create_directories(output_path);
      uint32_t inference_loops = 1;
      std::string model_path = argv[1];  // eg. "/home_mnt/software/spatial2/backend/binaries/CI_TTI_TEST_BINARIES_WH/bert.tti"

      if (argc == 3) {
          inference_loops = std::stoi(argv[2]);
      }

      // Create a pre-compiled model object and a backend object from it using default config
      std::shared_ptr<tt::tt_device_image> model = std::make_shared<tt::tt_device_image>(model_path, output_path);
      std::shared_ptr<tt_backend> backend = tt_backend::create(model, tt::tt_backend_config{});

      // The following code are organized into <runtime process> and <io process> sections
      // where the two processes can be running on different user spaces (e.g. host and soc)

      // <runtime process> - Initialize the backend
      if (backend->initialize() != tt::DEVICE_STATUS_CODE::Success) {
          throw std::runtime_error("Failed to initialize device");
      }

      // The following code must execute between initialize() and finish()
      for (uint32_t i = 0; i < inference_loops; i++) {
          // <io process> - Push a microbatch of inputs to device
          for (const std::string &name : model->get_graph_input_names()) {
              tt::tt_dram_io_desc io_desc = tt::io::utils::get_queue_descriptor(backend, name);
              tt::tt_PytorchTensorDesc tensor_desc = tt::io::utils::get_tensor_descriptor(name, model, io_desc);
              // Fill the tensor descriptor with data. We choose to allocate dummy memory using the TT backend for this tensor.
              // The user is free to use previously allocated memory, or use the backend to allocate memory that is then filled with actual data.
              tt::io::utils::fill_tensor_with_data(name, tensor_desc);
              // DMA the input tensor from host to device
              assert(tt::backend::push_input(io_desc, tensor_desc, false, 1) == tt::DEVICE_STATUS_CODE::Success);
              // Optional: Host memory management
              // - free releases storage on host (tensor data freed), since host is done with pushing data for this activation
              // - The user can choose not to free this memory and use it even after the data is in device DRAM 
              std::cout << "Pushed Input tensor " << name << " data ptr: " << tensor_desc.ptr << std::endl;
              assert(tt::backend::free_tensor(tensor_desc) == tt::DEVICE_STATUS_CODE::Success);
          }

          // <runtime process> - Run inference program, p_loop_count is the number of microbatches executed
          std::map<std::string, std::string> program_parameters = {{"$p_loop_count", "1"}};
          for (const auto& prog_name : backend -> get_programs()) {
              assert(backend->run_program(prog_name, program_parameters) == tt::DEVICE_STATUS_CODE::Success);
          }
          
          // <io process> - Pop a microbatch of outputs from device
          for (const std::string &name : model->get_graph_output_names()) {
              tt::tt_dram_io_desc io_desc = tt::io::utils::get_queue_descriptor(backend, name);
              tt::tt_PytorchTensorDesc tensor_desc = {};  // passed into get_tensor below to be populated

              // DMA the output tensor from device to host
              assert(tt::backend::get_output(io_desc, tensor_desc, false, 1) == tt::DEVICE_STATUS_CODE::Success);

              // Device memory management
              // - pop releases storage on device (queue entries popped), device can push more outputs to queue
              assert(tt::backend::pop_output(io_desc, false, 1) == tt::DEVICE_STATUS_CODE::Success);

              // Host memory management
              // - free releases storage on host (tensor data freed), host is done with the output data
              // - The user can choose not to free this memory and use it for downstream tasks 
              std::cout << "Got Output tensor " << name << " data ptr: " << tensor_desc.ptr << std::endl;
              assert(tt::backend::free_tensor(tensor_desc) == tt::DEVICE_STATUS_CODE::Success);
          }
      }
      // <runtime process> - Teardown the backend
      if (backend->finish() != tt::DEVICE_STATUS_CODE::Success) {
          throw std::runtime_error("Failed to shutdown device");
      }
      return 0;
  }


Pybuda Automatic Mixed Precision
--------------------------------------

Introduction
************

Automatic Mixed Precision (AMP) is a technique employed by Pybuda to leverage the hardware's ability to adjust precision along the hardware data path.
This technique mixes native support for low-precision block floating point (BFP) data format with higher precision data-formats (float16, bfloat16) 
to achieve accuracy results close or at parity relative to half precision.

There are several benefits enabling AMP:

- The ability make a trade at runtime between accuracy vs. performance based on application requirements.

- The ability to speed up compute-bound operations like matrix multiplication by reducing the cycles required.

- The ability to speed up memory-bound operations by reducing memory traffic between DRAM<->tensix or tensix<->tensix via on-chip interconnect.

- The ability to use half-precision/single-precision weights and run them in mixed precision without any changes to the model.


For additional details about our support for data formats, refer to :ref:`Data Formats<Data Formats>`. For additional details about Math Fidelity, refer to :ref:`Math Fidelity<Math Fidelity>`

Automatic Mixed Precision (AMP)
*******************************

During model compilation, the user will set the default configuration for precision and Math Fidelity of the model. This can be configured through the :py:func:`pybuda.set_configuration_options<pybuda.set_configuration_options>` API.

.. code-block:: python

   # Let's begin by setting the default single-precision configuration for the model:
   pybuda.set_configuration_options(
       default_df_override=pybuda.DataFormat.Float16_b,
       accumulate_df=pybuda.DataFormat.Float16_b,
       math_fidelity=pybuda.MathFidelity.HiFi3,
   )

AMP uses the global configuration set by the user and additionally enables a set of heuristics to determine the best precision for each operator in the model. These heuristics are based on empirical accuracy results. There are various switches uses to control the heuristics employed tune how aggressive these heuristics are. These switches are enabled by setting the following environment variables:

- PYBUDA_AMP_LIGHT=0: Disabled. No mixed precision settings are applied and the global configuration is used. The default MathFidelity is used for all operations.

- PYBUDA_AMP_LIGHT=1: Set the weights and bias inputs of MatMul operations to BFP8A/BFP8B and MathFidelity.HiFi2. For all other operations, the default MathFidelity provided by the user is used.

- PYBUDA_AMP_LIGHT=2: Set the weights and bias inputs of MatMul operations to BFP4A/BFP4B and MathFidelity.LoFi. For all other operations, the default MathFidelity provided by the user is used.


We also offer a switch to employ more aggressive mixed precision settings on select operators:

- PYBUDA_AMP_LEVEL=1: Set layernorm, softmax and fused operators to bfloat16 while keeping the rest of the model at BFP8B. For MatMul operations, HiFi2 is used, and for all other operations, the default MathFidelity provided by the user is used.


User-Defined Mixed Precision Configurations
*******************************************

With a recognition that precision tuning may differ across models, Pybuda provides an API for fine-grained control over the precision of arbitrary inputs, parameters and operators in the graph through: :py:func:`pybuda.config.configure_mixed_precision<pybuda.configure_mixed_precision>`.

Here's a simple example for adjusting the precision of select operators, inputs and parameters in the model:

.. code-block:: python

   import pybuda

   # Let's target matmul operators and set the accumulation data-format to bfloat16, output data-format to BFP8_b and math fidelity to HiFi2.
   pybuda.config.configure_mixed_precision(
    op_type="matmul",
    math_fidelity=pybuda.MathFidelity.HiFi2,
    accumulate_df=pybuda.DataFormat.Float16_b,
    output_df=pybuda.DataFormat.Bfp8_b,
   )
   
   # We'll also target the KQV matmul weights for attention modules and set them to lower precision
   pybuda.config.configure_mixed_precision(
    name_regex="layer.*.attention.self.(query|value|key).weight",
    output_df=pybuda.DataFormat.Bfp8_b,
   )

   # We'll also target the attention mask to set the data-format to BFP2_b.
   pybuda.config.configure_mixed_precision(
    name_regex="attention_mask",
    output_df=pybuda.DataFormat.Bfp2_b,
   )


User-Defined Intermediate Queues
********************************

Pybuda supports the ability to inspect/collect intermediate tensors at runtime to aid in the debug and model analysis of mixed precision configurations.
This enables a user to tag operations of interest during graph compilation and dedicated intermediate queues will be created to checkpoint intermediates.

Here is a simple example to (1) tag operations of interest and (2) fetch intermediates from device.

.. code-block:: python

   import torch
   import pybuda

   class PyBudaTestModule(pybuda.PyBudaModule):
       def __init__(self, name):
           super().__init__(name)
           self.weights1 = pybuda.Parameter(torch.rand(32, 32), requires_grad=True)
           self.weights2 = pybuda.Parameter(torch.rand(32, 32), requires_grad=True)

       def forward(self, x):
           matmul1 = pybuda.op.Matmul("matmul1", x, self.weights1)
           matmul1_gelu = pybuda.op.Gelu("gelu", matmul1)
           matmul2 = pybuda.op.Matmul("matmul2", matmul1_gelu, self.weights2)
           return matmul2

   # Guard in the main module to avoid creating subprocesses recursively.
   if __name__ == "__main__":
       # Configure Pybuda compilation options to include a list of operations to collect intermediate tensors
       tagged_operations = ["matmul1", "gelu"]
       pybuda.set_configuration_options(op_intermediates_to_save=tagged_operations)

       # Invoke the run_inference API to create device, compile and run module on device:
       output_q = pybuda.run_inference(PyBudaTestModule("test_module"), inputs=[torch.randn(1, 32, 32)])

       # After running inference, the intermediates queue will contain the ordered list of tagged intermediates
       intermediates_queue = pybuda.get_intermediates_queue()
       matmul1_tensor, gelu_tensor = intermediates_queue.get()

       # Print tensor values recorded from device inference
       print(matmul1_tensor)
       print(gelu_tensor)


Multiple Devices
----------------

Using Multiple Tenstorrent Devices
**********************************

PyBuda makes it easy to parallelize workloads onto multiple devices. A single :py:class:`TTDevice<pybuda.TTDevice>` can be used as a wrapper to any number of available 
Tenstorrent devices accessible to the host - either locally or through ethernet. The PyBuda compiler will then break up the workload over
assigned devices using either pipeline or model parallelism strategies, or a combination of both.

The easiest way to use all available hardware is to set ``num_chips`` parameter in :py:class:`TTDevice<pybuda.TTDevice>` to 0, which instructs it to auto-detect and use everything it can find. 
However, ``num_chips`` and ``chip_ids`` parameters can be used to select a subset of available hardware:

.. code-block:: python

   tt0 = TTDevice("tt0", num_chips=3) # Take first 3 available chips

.. code-block:: python

   tt0 = TTDevice("tt0", chip_ids[0, 2, 3]) # Skip chip id 1 and take 3 chips

See :py:class:`TTDevice<pybuda.TTDevice>` for more details.

Pybuda Multi-Model Support (Embedded Applications Only)
-------------------------------------------------------

Introduction
*******************

PyBuda allows users to merge several models into a single Tenstorrent Device Image, with minimal workflow overhead. The TTI can then be consumed by the C++ Backend and run on a Tenstorrent Device.

A typical process to generate and execute a Multi-Model workload is as follows:

**Compilation: Either Offline or on a Tenstorrent Device**

#. Generate TTIs for each model in the workload.
#. Run the Model-Merging tool to consolidate all models into a single TTI.

**Execution: On a Tenstorrent Device**

#. Spawn an application using the C++ backend APIs to deploy the workload contained in the TTI. An example application is provided in the `Embedded TTI Loading` section.

Fusing multiple independent models is well tested with several State of the Art models (including ViT, Mobilenet, ResNet50 ...). Supporting pipelined models is currently under active development.

Below, we describe the APIs and associated tools used to fuse models without any dependencies.

Usage
*****

Pybuda exposes two entry points for users to run the Model Merging Tool:

#. Command Line Interface to specify the list of models to merge along with optional arguments. These include parameters enabling/disabling certain optimizations.
#. Python API to be consumed by user applications. Usage of this API is very similar to the Command Line Tool.

**Command Line Interface**

.. code-block:: bash

  python3 pybuda/pybuda/tools/tti_merge.py [-h] [-mbl {dirname}] 
    [-mdl {models}] [-a {arch}]
    [-mml {filename}] [-scr] 
    [-dqo]

The following arguments are available when using `tti_merge.py`

.. list-table:: Table 1. TT-SMI optional arguments.
  :header-rows: 1

  * - Argument
    - Function
  * - -h, --help
    - Show help message and exit
  * - -mbl, --model_binaries_location
    - Relative path to where model TTIs are stored [Required]
  * - -mdl, --models
    - List of models to be merged (names must match TTI filenames) [Required]
  * - -a, --arch
    - Target Tenstorrent Architecture (default = wormhole_b0) [Optional]
  * - -mml, --merged_model_location
    - Relative path to where the Multi-Model TTI will be emitted (default = merged_model.tti) [Optional]
  * - -scr, --skip_channel_reallocation
    - Disable memory optimization that switches channels for queues when OOM during memory allocation (default = False) [Optional]
  * - -dqo, --dynamic_queue_overlap_off
    - Disable memory optimization allowing dynamic queues to overlap in memory channels (default = False) [Optional]
  
As an example, given the following directory structure in the Pybuda root directory:

.. code-block:: bash

  device_images_to_merge/
  ├-- bert_large.tti
  ├-- deit.tti
  ├-- hrnet.tti
  ├-- inception.tti
  ├-- mobilenet_v1.tti
  ├-- mobilenet_v2.tti
  ├-- mobilenet_v3.tti
  ├-- resnet.tti
  ├-- unet.tti
  ├-- vit.tti

The following command will generate a Multi-Model TTI (with memory optimizations enabled) and store it in `multi_model_workload.tti`:

.. code-block:: bash

  python3 pybuda/pybuda/tools/tti_merge.py -mbl device_images_to_merge/ -mdl bert_large deit hrnet inception mobilenet_v1 mobilenet_v2 mobilenet_v3 resnet unet vit -mml multi_model_workload.tti

**Python API**

This API provides identical functionality as the command line interface, for cases where the Model Merging step needs to be automated.

.. code-block:: python

  # API Declaration
  def merge_models(model_bin_location, models, arch = "wormhole_b0", merged_model_location = "", switch_chans_if_capacity_hit = True, overlap_dynamic_queues = True)

Here the arguments `switch_chans_if_capacity_hit` and `overlap_dynamic_queues` corresponds to memory optimizations, which are enabled my default.

The following Python code generates a Multi-Model TTI in a manner identical to the command listed in the previous section:

.. code-block:: python

  from pybuda.tools.tti_merge import merge_models

  model_binary_loc = "device_images_to_merge"
  models_to_merge = ["bert_large", "deit", "hrnet", "inception", "mobilenet_v1", "mobilenet_v2", "mobilenet_v3", "resnet", "unet", "vit"]
  target_arch = "wormhole_b0"
  merged_model_location = "multi_model_workload.tti"

  # Individual Model Generation Code Goes Here

  merge_models(model_binary_loc, models_to_merge, target_arch, merged_model_location)

**Memory Profiler**

During the model fusion process, the API presented above is responsible for performing memory reallocation. Users may be interested in the memory footprint of the fused model (both Device and Host DRAM).

To fulfill this requirement, the tool reports memory utilization post reallocation. An example using a model compiled for Wormhole (with 6 Device and up to 4 Host DRAM channels) is provided below.

.. code-block:: bash

Displaying memory footprint per DRAM channel (MB):
0 : 161.17
1 : 511.12
2 : 577.51
3 : 200.27
4 : 204.41
5 : 339.57
Displaying memory footprint per Host channel (MB):
0 : 132.88
1 : 0.0
2 : 0.0
3 : 0.0

TT-SMI
------

TT-SMI Introduction
*******************

TT-SMI is a command-line utility for monitoring and retrieving information about Tenstorrent devices. It provides two operating modes:

#. **Live Display**. Visualize live telemetry data, check on host info and device info.
#. **CLI**. Save a snapshot of telemetry into a .log file, dump telemetry into .csv/.xlsx, and perform warm resets on devices.

Usage
*****

.. code-block:: bash

  tt-smi [-h] [-v]
    [-t {default,dark}] [-nc]
    [-i [seconds]] [-s]
    [-f [filename]] [-d]
    [-dir [dirname]]
    [-dur seconds] [-wr]
    [-mr] [--external]

The following optional arguments are available when calling TT-SMI:

.. list-table:: Table 1. TT-SMI optional arguments.
  :header-rows: 1

  * - Argument
    - Function
  * - -h, --help
    - Show help message and exit
  * - -v, --version
    - Show program's version number and exit
  * - -t {default, dark}, --theme {default, dark}
    - Change color theme of live display. Use dark theme for light backgrounds
  * - -nc, --no-color
    - Remove color from output
  * - -i [seconds], --interval [seconds]
    - Change time interval between readings. Default: 0.5s
  * - -s, --snapshot
    - Take snapshot of current host and device information
  * - -f [filename], --filename [filename]
    - Change filename for snapshot. Default: tt-smi-snapshot\_\<timestamp\>
  * - -d, --dump
    - Dump device telemetry to file
  * - -dir [dirname], --dump\_dir [dirname]
    - Change directory path to dump to. Default: tt-smi-dump\_\<timestamp\>/
  * - -dur [seconds], --duration [seconds]
    - Set duration in seconds to dump for
  * - -wr, --warm\_reset
    - Warm reset board
  * - -mr, --mobo\_nb\_reset
    - Warm reset mobo
  * - --external
    - Run external version of TT-SMI

Some usage examples include:

- ``tt-smi`` (run live display)
- ``tt-smi -t dark`` (run live display with dark theme)
- ``tt-smi -nc`` (run live display with no color)
- ``tt-smi -i 0.01`` (run live display with interval between readings set to 0.01s)
- ``tt-smi -s`` (save snapshot to ./tt-smi-logs/tt-smi-snapshot\_\<timestamp\>)
- ``tt-smi -s -f my\_file.log`` (save snapshot to ./my\_file.log)
- ``tt-smi -d`` (dump until Enter is pressed and save to ./tt-smi-logs/tt-smi-dump\_\<timestamp\>/)
- ``tt-smi -d -dir my\_dir`` (dump until Enter is pressed and save in ./my\_dir/)
- ``tt-smi -d -dur 30`` (dump for 30 seconds and save to ./tt-smi-logs/tt-smi-dump\_\<timestamp\>/)
- ``tt-smi -d -dir my\_dir -dur 30`` (dump for 30 seconds and save in ./my\_dir/)
- ``tt-smi -wr`` (warm reset the board)
- ``tt-smi -mr`` (warm reset mobo)

Live Display Mode
*****************

The live display mode of TT-SMI shows several content boxes by default: the header, host info, compatibility check, device info, and footer.

The header displays TT-SMI version, the title, and the current datetime. The Host Info box provides some specs about the host system running TT-SMI, while the Compatibility Check box shows if device status and host system requirements are currently being met. The footer shows an abbreviated list of keyboard shortcuts.

The following live display key presses astomize the view and perform quick actions:

.. list-table:: Table 2. Live display keyboard shortcuts
  :header-rows: 1

  * - Action
    - Key Press
  * - Open device info tab
    - 1
  * - Open device telemetry tab
    - 2
  * - Open device firmware tab
    - 3
  * - Toggle left sidebar
    - t
  * - Toggle helper menu
    - h
  * - Toggle view of telemetry limits
    - w
  * - Toggle view of telemetry average
    - v
  * - Take snapshot
    - s
  * - Start/Stop telemetry dump
    - d

The device info tab shows some board info, chip info, DRAM status, and PCIe link status.

The device telemetry tab displays the following values in real-time: core voltage, current, and power, clock frequencies, core temperature, voltage regulator (VREG) temperature, inlet temperature, and outlet temperatures. Several values have expected maximum limits, shown in yellow bold text.

The device firmware versions tab shows data about the firmware on each device.

CLI Mode
********

Dump Telemetry
**************

To dump telemetry data for detected Tenstorrent devices to a spreadsheet format, use TT-SMI with the -d option. Unless a duration is specified with -dur, the telemetry dump will continue until the user presses Enter.

The resulting data is dumped into a .csv and an .xlsx file.

.. list-table:: Table 3. Telemetry dump output for one chip in spreadsheet format
  :header-rows: 1

  * - Timestamp (s)
    - Core Voltage (V)
    - AICLK (MHz)
    - ARCCLK (MHz)
    - AXICLK (MHz)
    - Core Current (A)
    - Core Power (W)
    - Core Temp (°C)
    - VREG Temp (°C)
    - Inlet Temp (°C)
    - Outlet Temp 1 (°C)
    - Outlet Temp 2 (°C)
  * - **0.502691**
    - 0.80
    - 810
    - 540
    - 900
    - 47.0
    - 37.0
    - 41.0
    - 43.0
    - 0.0
    - 0.0
    - 0.0
  * - **1.004782**
    - 0.80
    - 810
    - 540
    - 900
    - 47.0
    - 37.0
    - 40.8
    - 42.0
    - 0.0
    - 0.0
    - 0.0
  * - **1.506589**
    - 0.80
    - 810
    - 540
    - 900
    - 47.0
    - 37.0
    - 40.9
    - 43.0
    - 0.0
    - 0.0
    - 0.0
  * - **2.008529**
    - 0.80
    - 810
    - 540
    - 900
    - 47.0
    - 37.0
    - 40.9
    - 43.0
    - 0.0
    - 0.0
    - 0.0
  * - **2.510333**
    - 0.80
    - 810
    - 540
    - 900
    - 47.0
    - 37.0
    - 40.9
    - 43.0
    - 0.0
    - 0.0
    - 0.0
  * - **3.012129**
    - 0.80
    - 810
    - 540
    - 900
    - 47.0
    - 37.0
    - 40.8
    - 43.0
    - 0.0
    - 0.0
    - 0.0
  * - **3.513459**
    - 0.80
    - 810
    - 540
    - 900
    - 47.0
    - 38.0
    - 41.1
    - 43.0
    - 0.0
    - 0.0
    - 0.0
  * - **4.015265**
    - 0.80
    - 810
    - 540
    - 900
    - 47.0
    - 38.0
    - 41.0
    - 43.0
    - 0.0
    - 0.0
    - 0.0

Snapshot
********

Using TT-SMI with the -s option generates an output log file using telemetry and host data from a single moment in time.

Warm Reset
**********

TT-SMI can be used with the -wr option to warm reset Tenstorrent devices (reset without rebooting the host system). Users can enter the indices of any devices and press Enter to reset them.

Warm Reset Mobo
***************

With the -mr option, TT-SMI will warm reset the mobo. User will be prompt to enter the host name of the mobo and the ethernet port(s) connected on the mobo.

Examples of PyBuda use cases
----------------------------

.. literalinclude:: ../../pybuda/test/test_user.py
   :language: python
