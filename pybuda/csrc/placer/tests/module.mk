PYBUDA_CSRC_PLACER_TESTS = $(TESTDIR)/pybuda/csrc/placer/tests/placer_unit_tests
PYBUDA_CSRC_PLACER_TESTS_SRCS = \
	pybuda/csrc/placer/tests/unit_tests.cpp \
	pybuda/csrc/placer/tests/dram.cpp \
	pybuda/csrc/placer/tests/gtest_main.cpp

PYBUDA_CSRC_PLACER_TESTS_INCLUDES = $(PYBUDA_CSRC_PLACER_INCLUDES)
PYBUDA_CSRC_PLACER_TESTS_LDFLAGS = -lstdc++fs -lgtest -lgtest_main -lpthread -l$(PYTHON_VERSION) -lm

PYBUDA_CSRC_PLACER_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PLACER_TESTS_SRCS:.cpp=.o))
PYBUDA_CSRC_PLACER_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PLACER_TESTS_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_PLACER_TESTS_DEPS)

pybuda/csrc/placer/tests: $(PYBUDA_CSRC_PLACER_TESTS)

$(PYBUDA_CSRC_PLACER_TESTS): $(PYBUDA_CSRC_PLACER_TESTS_OBJS) $(PYBUDA_CSRC_LIB)
	@mkdir -p $(@D)
	$(CXX) $(PLACER_CSRC_CFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(PYBUDA_CSRC_PLACER_TESTS_LDFLAGS)

$(OBJDIR)/pybuda/csrc/placer/tests/%.o: pybuda/csrc/placer/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PLACER_CSRC_CFLAGS) $(CXXFLAGS) $(PYBUDA_CSRC_PLACER_TESTS_INCLUDES) -c -o $@ $<
