PYBUDA_CSRC_GRAPHLIB_TESTS = $(TESTDIR)/pybuda/csrc/graph_lib/tests/graphlib_unit_tests
PYBUDA_CSRC_GRAPHLIB_TESTS_SRCS = \
	pybuda/csrc/graph_lib/tests/test_graphlib_utils.cpp \
	pybuda/csrc/graph_lib/tests/test_graphlib.cpp

PYBUDA_CSRC_GRAPHLIB_TESTS_INCLUDES = $(PYBUDA_CSRC_INCLUDES)
PYBUDA_CSRC_GRAPHLIB_TESTS_LDFLAGS = -lgtest -lgtest_main -lpthread -l$(PYTHON_VERSION) -lm

PYBUDA_CSRC_GRAPHLIB_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_GRAPHLIB_TESTS_SRCS:.cpp=.o))
PYBUDA_CSRC_GRAPHLIB_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_GRAPHLIB_TESTS_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_GRAPHLIB_TESTS_DEPS)

pybuda/csrc/graph_lib/tests: $(PYBUDA_CSRC_GRAPHLIB_TESTS)

$(PYBUDA_CSRC_GRAPHLIB_TESTS): $(PYBUDA_CSRC_GRAPHLIB_TESTS_OBJS) $(PYBUDA_CSRC_LIB)
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(PYBUDA_CSRC_GRAPHLIB_TESTS_LDFLAGS)

$(OBJDIR)/pybuda/csrc/graph_lib/tests/%.o: pybuda/csrc/graph_lib/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(PYBUDA_CSRC_GRAPHLIB_TESTS_INCLUDES) -c -o $@ $<
