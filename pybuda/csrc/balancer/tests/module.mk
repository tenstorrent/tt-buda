PYBUDA_CSRC_BALANCER_TESTS = $(TESTDIR)/pybuda/csrc/balancer/tests/balancer_unit_tests
PYBUDA_CSRC_BALANCER_TESTS_SRCS = \
	$(wildcard pybuda/csrc/balancer/tests/*.cpp) \

PYBUDA_CSRC_BALANCER_TESTS_INCLUDES = -Ipybuda/csrc/graph_lib $(PYBUDA_CSRC_BALANCER_INCLUDES)
PYBUDA_CSRC_BALANCER_TESTS_LDFLAGS = -lgtest -lgtest_main -lpthread -l$(PYTHON_VERSION) -lm -lbalancer

PYBUDA_CSRC_BALANCER_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_BALANCER_TESTS_SRCS:.cpp=.o))
PYBUDA_CSRC_BALANCER_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_BALANCER_TESTS_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_BALANCER_TESTS_DEPS)

pybuda/csrc/balancer/tests: $(PYBUDA_CSRC_BALANCER_TESTS)

$(PYBUDA_CSRC_BALANCER_TESTS): $(PYBUDA_CSRC_BALANCER_TESTS_OBJS) $(PYBUDA_CSRC_LIB) $(PYBUDA_CSRC_TILE_MAPS_LIB)
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(PYBUDA_CSRC_BALANCER_TESTS_LDFLAGS)

$(OBJDIR)/pybuda/csrc/balancer/tests/%.o: pybuda/csrc/balancer/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(PYBUDA_CSRC_BALANCER_TESTS_INCLUDES) -c -o $@ $<
