# Every variable in subdir must be prefixed with subdir (emulating a namespace)
PYBUDA_CSRC_GRAPH_LIB = $(LIBDIR)/libgraph_lib.a
PYBUDA_CSRC_GRAPH_LIB_SRCS = \
	pybuda/csrc/graph_lib/defines.cpp \
	pybuda/csrc/graph_lib/edge.cpp \
	pybuda/csrc/graph_lib/graph.cpp \
	pybuda/csrc/graph_lib/node.cpp \
	pybuda/csrc/graph_lib/node_types.cpp \
	pybuda/csrc/graph_lib/shape.cpp \
	pybuda/csrc/graph_lib/utils.cpp \
	pybuda/csrc/graph_lib/python_bindings.cpp

PYBUDA_CSRC_GRAPH_LIB_INCLUDES = $(PYBUDA_CSRC_INCLUDES)

PYBUDA_CSRC_GRAPH_LIB_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_GRAPH_LIB_SRCS:.cpp=.o))
PYBUDA_CSRC_GRAPH_LIB_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_GRAPH_LIB_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_GRAPH_LIB_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
pybuda/csrc/graph_lib: $(PYBUDA_CSRC_GRAPH_LIB)

$(PYBUDA_CSRC_GRAPH_LIB): $(PYBUDA_CSRC_GRAPH_LIB_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/graph_lib/%.o: pybuda/csrc/graph_lib/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_GRAPH_LIB_INCLUDES) -c -o $@ $<

include pybuda/csrc/graph_lib/tests/module.mk
