PYBUDA_CSRC_PASSES_TESTS = $(TESTDIR)/pybuda/csrc/passes/tests/passes_unit_tests
PYBUDA_CSRC_PASSES_TESTS_SRCS = \
	pybuda/csrc/balancer/tests/test_balancer_utils.cpp \
	$(wildcard pybuda/csrc/passes/tests/*.cpp)


PYBUDA_CSRC_PASSES_TESTS_INCLUDES = -Ipybuda/csrc/graph_lib $(PYBUDA_CSRC_INCLUDES)
PYBUDA_CSRC_PASSES_TESTS_LDFLAGS = -lstdc++fs -lgtest -lgtest_main -lpthread -l$(PYTHON_VERSION) -lm

PYBUDA_CSRC_PASSES_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PASSES_TESTS_SRCS:.cpp=.o))
PYBUDA_CSRC_PASSES_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PASSES_TESTS_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_PASSES_TESTS_DEPS)

pybuda/csrc/passes/tests: $(PYBUDA_CSRC_PASSES_TESTS)

# gcc + pybind causing segfault at the end of the tests
$(PYBUDA_CSRC_PASSES_TESTS): $(PYBUDA_CSRC_PASSES_TESTS_OBJS) $(PYBUDA_CSRC_LIB)
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(PYBUDA_CSRC_PASSES_TESTS_LDFLAGS)

$(OBJDIR)/pybuda/csrc/passes/tests/%.o: pybuda/csrc/passes/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(PYBUDA_CSRC_PASSES_TESTS_INCLUDES) -c -o $@ $<
