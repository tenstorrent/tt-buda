Terminology
===========

Grayskull
---------
First-generation Tenstorrent chip

Wormhole
--------
Second-generation Tenstorrent chip

Tensix
------
High-performance compute core in Tenstorrent chips

NOC
---
Network-on-Chip

L1
---
Local memory storage (SRAM) in Tensix is commonly referred to as L1 memory. 

FPU
---
The dense tensor math unit in Tensix. It performs bulk tensor math operations, such as multiplication and addition, in various floating point formats.

SFPU
----
Tensix SIMD engine, used for various miscellaneous activation operations, such as exponents, square roots, softmax, topK, and others.

Unpacker
--------
Hardware unit that copies data from local memory to source registers inside :ref:`FPU<fpu>`. In the process of copying data, unpacker can also uncompress compressed data, and perform conversion
between various floating point formats.

Packer
------
Hardware unit that copies data from destination registers in FPU to L1 local memory. In the process of copying data, packer can also compress data, and perform format conversions.

Op
---
A Buda op (*operation*) is a single math operation, such as matrix multiply, convolution, or pool, performed on one or more tensor operands. If you have ever written a model in PyTorch or
TensorFlow, a Buda op is analogous to an operation in those frameworks. 

Epoch
-----
A collection of ops that fits onto one chip. In a typical workflow, epoch code will be copied onto a chip which will then compute a batch of one or more tensors before moving on to the next epoch. 

Buffer
------
A reserved location in local memory, DRAM, or host memory. Buffers are used either as destinations for operation outputs, sources for operation inputs, or temporary locations for intermediate data.

Pipe
----
A pipe is a connection between two or more buffers. Once established, a pipe will automatically copy data from source buffers into destination buffers, as space allows. 

Tile
----
The smallest unit of math and data transfer in Buda. It is a 32x32 matrix of data.

Streaming op
------------
A streaming operation is one that produces outputs as soon as inputs are available to it. It performs the math on a tile-by-tile basis, and therefore requires minimal buffering. Most elementwise
operations fall into this category.

Blocking op
-----------
A blocking operation is one that requires a block of tiles to be calculated. Matrix multiplications and reduce operations are typical examples of blocking ops.

Block
-----
A group of tiles on which a :ref:`blocking operation<blocking op>` is performed. 

TM op
-----
A tensor manipulation (TM) op is a tensor operation that modifies the order of tiles within the tensor, but does not modify the contents. Typically, this involves reshapes, squeezes, unsqueezes
and other explicit TM operations available in other frameworks, however, there is a class of TMs that are specific to Buda. Because Buda operations rely on input operands arriving in a particular
order and block shape, sometimes TM operations are needed to "rebuffer" incoming data into the expected order. These TMs can be manually inserted by the user, or automatically by the Buda
compiler.

Unicast
-------
A :ref:`pipe` between two buffers, where one copy of data is transferred from source to destination buffer.

Multicast
---------
A multicast is a :ref:`pipe` that has one source buffer but more than one destination. In this case, the NOC will use its multicasting ability to only send one copy of the data along its
connections, while reaching all destination buffers. It is critical to use multicast pipes whenever same data is broadcast to multiple cores of operations, or bandwidth bottlenecks will be created
on NOC links.

Gather
------
A gather :ref:`pipe` has multiple source buffers and one or more destination buffers. The gather pipe will create connections from source buffers to a single gather buffer, which will then be
unicast or broadcast to destinations.

HLK
---
An HLK (High-Level Kernel) is a piece of C++ Buda code that implements the op's behaviour. It typically contains one or more nested loops in which inputs are read into operand registers, some math
is performed, and results are written out. HLK is transpiled, using :ref:`HLKC`, into low-level C++ code that runs on individual RiscV cores. Users who implement new operations would write HLKs
to describe the desired functionality.

LLK
---
Low-Level Kernels are implementations of low-level Tensix and NOC API used in :ref:`HLKs<HLK>`. They use a mix of C++ and assembly to provide highly optimized and performant functionality, such
as data :ref:`unpacking<unpacker>`, :ref:`packing<packer>`, running math primitives, and so on. These are typically written by Tenstorrent as they require in-depth knowledge of Tenstorrent
hardware and low-level programming requirements. 

HLKC
----
HLK Compiler is a transpiler that convert single-threaded HLK into multi-threaded low-level RiscV code that separates data transfer and math into parallel threads for maximum utilization of
compute resources.
