# Every variable in subdir must be prefixed with subdir (emulating a namespace)
PYBUDA_CSRC_INCLUDES = \
	-Ipybuda/csrc \
	-Ithird_party/json \
	-Ithird_party/budabackend \
	-Ithird_party/budabackend/netlist \
	-I/usr/include/$(PYTHON_VERSION) \
	-isystem $(PYTHON_ENV)/lib/$(PYTHON_VERSION)/site-packages/torch/include \
 	-isystem $(PYTHON_ENV)/lib/$(PYTHON_VERSION)/site-packages/torch/include/torch/csrc/api/include

PYBUDA_CSRC_WARNINGS ?= -Wall -Wextra -Wno-pragmas
PYBUDA_CSRC_CFLAGS ?= $(CFLAGS_NO_WARN) $(PYBUDA_CSRC_WARNINGS) -DUTILS_LOGGER_PYTHON_OSTREAM_REDIRECT=1
BOOST_LIB_DIR = /usr/lib/x86_64-linux-gnu # use system installed boost
TORCH_LIB_DIR = $(PYTHON_ENV)/lib/$(PYTHON_VERSION)/site-packages/torch/lib

PYBUDA_CSRC_LIB = $(LIBDIR)/libpybuda_csrc.so

include pybuda/csrc/graph_lib/module.mk
include pybuda/csrc/shared_utils/module.mk
include pybuda/csrc/scheduler/module.mk
include pybuda/csrc/placer/module.mk
include pybuda/csrc/autograd/module.mk
include pybuda/csrc/balancer/module.mk
include pybuda/csrc/reportify/module.mk
include pybuda/csrc/backend_api/module.mk
include pybuda/csrc/pattern_matcher/module.mk
include pybuda/csrc/perf_model/module.mk
include pybuda/csrc/tt_torch_device/module.mk

ifndef BUDABACKEND_LIBDIR
$(error BUDABACKEND_LIBDIR not set)
endif

PYBUDA_CSRC_LDFLAGS = -Wl,-z,origin -Wl,-rpath,\$$ORIGIN/../python_env/lib/$(PYTHON_VERSION)/site-packages/torch/lib -Wl,-rpath,\$$ORIGIN/../budabackend/build/lib -Wl,-rpath,\$$ORIGIN/../../$(BUDABACKEND_LIBDIR) -lstdc++fs -lboost_serialization -ltorch -ltorch_cpu -lc10 -ltorch_python -lnet2pipe_tile_maps

PYBUDA_CSRC_SRCS = \
		pybuda/csrc/pybuda_bindings.cpp \
		pybuda/csrc/buda_passes.cpp \
		$(wildcard pybuda/csrc/passes/*.cpp) \
		pybuda/csrc/lower_to_buda/netlist.cpp \
		pybuda/csrc/lower_to_buda/queue.cpp \
		pybuda/csrc/lower_to_buda/graph.cpp \
		pybuda/csrc/lower_to_buda/op.cpp \
		pybuda/csrc/lower_to_buda/device.cpp \
		pybuda/csrc/lower_to_buda/debug.cpp \
		pybuda/csrc/lower_to_buda/program.cpp \
		pybuda/csrc/lower_to_buda/fused_op.cpp \
		pybuda/csrc/lower_to_buda/common.cpp

include pybuda/csrc/passes/tests/module.mk
include pybuda/csrc/balancer/tests/module.mk

PYBUDA_CSRC_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_SRCS:.cpp=.o))
PYBUDA_CSRC_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_SRCS:.cpp=.d))

PYBUDA_THIRD_PARTY_DEPS = $(SUBMODULESDIR)/third_party/pybind11.checkout

-include $(PYBUDA_CSRC_DEPS)

$(PYBUDA_CSRC_LIB): $(PYBUDA_CSRC_OBJS) $(PYBUDA_CSRC_GRAPH_LIB) $(PYBUDA_CSRC_AUTOGRAD) $(PYBUDA_CSRC_PATTERN_MATCHER_LIB) $(PYBUDA_CSRC_BALANCER_LIB) $(PYBUDA_CSRC_PLACER_LIB) $(PYBUDA_CSRC_SCHEDULER_LIB) $(PYBUDA_CSRC_REPORTIFY) $(PYBUDA_CSRC_BACKENDAPI_LIB) $(PYBUDA_CSRC_SHARED_UTILS_LIB) $(PYBUDA_CSRC_PERF_MODEL_LIB) $(PYBUDA_THIRD_PARTY_DEPS) $(PYBUDA_THIRD_PARTY_DEPS) $(PYBUDA_CSRC_TT_TORCH_DEVICE_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -L$(BOOST_LIB_DIR) -L$(TORCH_LIB_DIR) -o $@ $^ $(LDFLAGS) $(PYBUDA_CSRC_LDFLAGS)

$(PYTHON_ENV)/lib/$(PYTHON_VERSION)/site-packages/pybuda/_C.so: $(PYBUDA_CSRC_LIB)
	@mkdir -p $(PYTHON_ENV)/lib/$(PYTHON_VERSION)/site-packages/pybuda
	cp $^ $@
	touch -r $^ $@
	ln -sf ../../$(PYTHON_ENV)/lib/$(PYTHON_VERSION)/site-packages/pybuda/_C.so pybuda/pybuda/_C.so

$(OBJDIR)/pybuda/csrc/%.o: pybuda/csrc/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) $(PYBUDA_CSRC_INCLUDES) -c -o $@ $<

pybuda/csrc: $(PYBUDA_CSRC_LIB) third_party/budabackend/src/net2pipe ;
