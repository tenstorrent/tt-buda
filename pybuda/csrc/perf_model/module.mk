# Every variable in subdir must be prefixed with subdir (emulating a namespace)

PYBUDA_CSRC_PERF_MODEL_LIB = $(LIBDIR)/libperf_model.a
PYBUDA_CSRC_PERF_MODEL_SRCS = \
	pybuda/csrc/perf_model/graph.cpp \
	pybuda/csrc/perf_model/perf_model.cpp \
	pybuda/csrc/perf_model/event.cpp \
	pybuda/csrc/perf_model/trace.cpp \
	pybuda/csrc/perf_model/simulator.cpp

PYBUDA_CSRC_PERF_MODEL_INCLUDES = $(PYBUDA_CSRC_INCLUDES)

PYBUDA_CSRC_PERF_MODEL_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PERF_MODEL_SRCS:.cpp=.o))
PYBUDA_CSRC_PERF_MODEL_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PERF_MODEL_SRCS:.cpp=.d))



-include $(PYBUDA_CSRC_PERF_MODEL_DEPS)

PERF_MODEL_CSRC_CFLAGS = $(PYBUDA_CSRC_CFLAGS)

# Each module has a top level target as the entrypoint which must match the subdir name
pybuda/csrc/perf_model: $(PYBUDA_CSRC_PERF_MODEL_LIB)

$(PYBUDA_CSRC_PERF_MODEL_LIB): $(PYBUDA_CSRC_PERF_MODEL_OBJS) $(PYBUDA_CSRC_GRAPH_LIB) 
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/perf_model/%.o: pybuda/csrc/perf_model/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PERF_MODEL_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_PERF_MODEL_INCLUDES) -c -o $@ $<

include pybuda/csrc/perf_model/tests/module.mk
