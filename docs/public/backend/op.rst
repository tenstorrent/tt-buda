Buda Workload Overview
======================

A Buda workload is a collection of ML modules that are placed to run on one or more computer systems that each contain one or more Tenstorrent chips. Buda makes it easy to divide workload into modules
and spread over both small and large hardware topologies. 

A Buda module is represented as a graph of math operations (:ref:`Ops<Op>`). Writing out a module is very similar to how it is done in other frameworks, like PyTorch or TensorFlow: operations are
listed one by one in Python or C++ code, and outputs of operations are fed as inputs into others, creating compute graph connections.

An op is *placed* on to a grid of one or more Tensix cores, which are going be responsible for performing the computation. It is the model writer's responsibility to choose the size of this grid.
Typically, larger operations should be placed onto more cores than smaller ones in order to keep the computation balanced across all Tensix cores, but many other parameters come into play: placement,
routing, bandwidths, local memory availability, and so on. These will be covered in detail later on. 

Buda compiler will automatically place and connect operations in a graph. While it will do a good job of optimizing the placement and routing of operations in many cases, there will be situations
where manual intervention is needed to get the most performance out of hardware. Visualizations, reports, and APIs are available to users to get detailed feedback on expected performance, problem
areas and bottlenecks, and tools to fix them.

Commonly, the full module will not fit on available hardware, unless either the model is very small, or is being placed onto a large grid of Tenstorrent chips. In that case, the Buda compiler will
split the module into :ref:`epochs<Epoch>`. During an epoch, placement and routing between operations is static, and one or more input tensors are fed through this pipeline of operations until
outputs are produced at the end of the epoch. An epoch could run this way through either a single input or a large batch of inputs, buffering the outputs in local memory, DRAM, or on another chip. 

Within an epoch, operations are connected using :ref:`pipes<Pipe>` and :ref:`buffers<Buffer>`. Each operation defines its own input and output buffers based on its needs, and then creates pipes
between source operations and its input buffers. There is an understanding that the source operations produces data in a row-major tile order, evenly split along the grid of cores it was placed on. 
If this is not compatible with the consuming op, a :ref:`TM op` will need to be inserted to bring data into expected form.

Pipes provide for a flexible connectivity between operations, allowing for many kinds of topologies. See :ref:`Unicast`, :ref:`Multicast`, and :ref:`Gather` for more details.

Anatomy of a Buda op
--------------------

A functional Buda op has several responsibilities: it needs to define the math operation it performs, the buffering requirements for it input operands and its output(s), connectivity between itself
and the previous operations, and provide performance modelling feedback.

The math operation is defined through a high-level kernel (:ref:`HLK`). A typical HLK looks like simple single-threaded piece of code in which input operands are read, math performed on them, and
output written out. 

.. code-block:: cpp

  void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {

      for(int t = 0; t < args->per_core_tile_cnt; ++t)
      {
          hlk_acquire_dst(core_ptr, DstMode::Tile);

          // Wait for tiles on the input
          hlk_wait_tiles(core_ptr, HlkOperand::in0, 1);
          hlk_wait_tiles(core_ptr, HlkOperand::in1, 1);
          // Wait for space in output
          hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, 1);

          // Add and pack
          hlk_add_tile(core_ptr, HlkOperand::in0, HlkOperand::in1, 0, 0, 0);
          hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0);

          // Pop input and push to output 
          hlk_pop_tiles(core_ptr, HlkOperand::in0, 1);
          hlk_pop_tiles(core_ptr, HlkOperand::in1, 1);
          hlk_push_tiles(core_ptr, HlkOperand::out0, 1);

          hlk_release_dst(core_ptr, DstMode::Tile);
      }
  }

An HLK uses `hlk_` API to perform low-level actions in Tensix cores. These involve waiting for :ref:`tiles<Tile>` to arrive in input buffers, reading them into source operands, running math
primitives, and packing and pushing output data back to output operands. The API implementation is in :ref:`low-level kernels<LLK>`, which are highly optimized pieces of code that abstract away many
of the details of hardware programming. 

In addition to HLKs, a Buda op defines :ref:`buffers<Buffer>` and :ref:`pipes<Pipe>` needed to move data to and from the operation. The Buda op will create input buffers for each of its operands
based on its requirements. A :ref:`Streaming op` might only need room for two input tiles, while a :ref:`Blocking op` will need to store at least two blocks to be able to perform at 100%. An output
buffer should be sizes to contain at least one output :ref:`Block` to prevent deadlocks, but the actual sizing will depend on the performance requirements. 

Once the buffers are in places, the Buda op defines pipes to bring data in from incoming operations into its input buffers. Depending on the op requirements, it might need a set of :ref:`Unicast`
pipes (for example, an elementwise operation), or a :ref:`Multicast` if several of its cores will be using the same data (most common example of this is a matrix multiplication op). 

