PYBUDA_CSRC_PERF_MODEL_TESTS = $(TESTDIR)/pybuda/csrc/perf_model/tests/perf_model_unit_tests
PYBUDA_CSRC_PERF_MODEL_TESTS_SRCS = \
	pybuda/csrc/perf_model/tests/simulator_tests.cpp \
	pybuda/csrc/perf_model/tests/gtest_main.cpp

PYBUDA_CSRC_PERF_MODEL_TESTS_INCLUDES = $(PYBUDA_CSRC_PERF_MODEL_INCLUDES)
PYBUDA_CSRC_PERF_MODEL_TESTS_LDFLAGS = -lstdc++fs -lgtest -lgtest_main -lpthread -l$(PYTHON_VERSION) -lm

PYBUDA_CSRC_PERF_MODEL_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PERF_MODEL_TESTS_SRCS:.cpp=.o))
PYBUDA_CSRC_PERF_MODEL_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PERF_MODEL_TESTS_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_PERF_MODEL_TESTS_DEPS)

pybuda/csrc/perf_model/tests: $(PYBUDA_CSRC_PERF_MODEL_TESTS)

$(PYBUDA_CSRC_PERF_MODEL_TESTS): $(PYBUDA_CSRC_PERF_MODEL_TESTS_OBJS) $(PYBUDA_CSRC_PERF_MODEL_LIB) $(PYBUDA_CSRC_GRAPH_LIB)
	@mkdir -p $(@D)
	$(CXX) $(PERF_MODEL_CSRC_CFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(PYBUDA_CSRC_PERF_MODEL_TESTS_LDFLAGS)

$(OBJDIR)/pybuda/csrc/perf_model/tests/%.o: pybuda/csrc/perf_model/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PERF_MODEL_CSRC_CFLAGS) $(CXXFLAGS) $(PYBUDA_CSRC_PERF_MODEL_TESTS_INCLUDES) -c -o $@ $<
