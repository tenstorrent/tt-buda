# Every variable in subdir must be prefixed with subdir (emulating a namespace)

PYBUDA_CSRC_BALANCER_LIB = $(LIBDIR)/libbalancer.a
PYBUDA_CSRC_BALANCER_SRCS += \
	pybuda/csrc/balancer/balancer.cpp \
	pybuda/csrc/balancer/balancer_utils.cpp \
	pybuda/csrc/balancer/legalizer/constraints.cpp \
	pybuda/csrc/balancer/legalizer/graph_solver.cpp \
	pybuda/csrc/balancer/legalizer/legalizer.cpp \
	pybuda/csrc/balancer/types.cpp \
	pybuda/csrc/balancer/python_bindings.cpp \
	$(wildcard pybuda/csrc/balancer/policies/*.cpp)

PYBUDA_CSRC_BALANCER_INCLUDES = $(PYBUDA_CSRC_INCLUDES)

PYBUDA_CSRC_BALANCER_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_BALANCER_SRCS:.cpp=.o))
PYBUDA_CSRC_BALANCER_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_BALANCER_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_BALANCER_DEPS)

PYBUDA_CSRC_LOGGER_SRC     = third_party/budabackend/src/net2pipe/src/net2pipe_logger.cpp
PYBUDA_CSRC_LOGGER_OBJ     = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_LOGGER_SRC:.cpp=.o))
PYBUDA_CSRC_LOGGER_DEP     = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_LOGGER_SRC:.cpp=.d))

-include $(PYBUDA_CSRC_LOGGER_DEP)

# Each module has a top level target as the entrypoint which must match the subdir name
pybuda/csrc/balancer: $(PYBUDA_CSRC_BALANCER_LIB)

$(PYBUDA_CSRC_BALANCER_LIB): $(PYBUDA_CSRC_PLACER_LIB) $(PYBUDA_CSRC_BALANCER_OBJS) $(PYBUDA_CSRC_GRAPH_LIB)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/balancer/%.o: pybuda/csrc/balancer/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_BALANCER_INCLUDES) -c -o $@ $<
