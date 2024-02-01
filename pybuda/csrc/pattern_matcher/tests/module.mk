PYBUDA_CSRC_PATTERN_MATCHER_TESTS = $(TESTDIR)/pybuda/csrc/pattern_matcher/tests/pattern_matcher_unit_tests
PYBUDA_CSRC_PATTERN_MATCHER_TESTS_SRCS = \
	pybuda/csrc/pattern_matcher/tests/unit_tests.cpp \
	pybuda/csrc/pattern_matcher/tests/gtest_main.cpp

PYBUDA_CSRC_PATTERN_MATCHER_TESTS_INCLUDES = $(PYBUDA_CSRC_PATTERN_MATCHER_INCLUDES) -I./boost_test_graphs
PYBUDA_CSRC_PATTERN_MATCHER_TESTS_LDFLAGS = -lstdc++fs -lgtest -lgtest_main -lpthread -l$(PYTHON_VERSION) -L./third_party/boost/stage/lib -lboost_serialization -lm

PYBUDA_CSRC_PATTERN_MATCHER_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PATTERN_MATCHER_TESTS_SRCS:.cpp=.o))
PYBUDA_CSRC_PATTERN_MATCHER_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PATTERN_MATCHER_TESTS_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_PATTERN_MATCHER_TESTS_DEPS)

pybuda/csrc/pattern_matcher/tests: $(PYBUDA_CSRC_PATTERN_MATCHER_TESTS)

$(PYBUDA_CSRC_PATTERN_MATCHER_TESTS): $(PYBUDA_CSRC_PATTERN_MATCHER_TESTS_OBJS) $(PYBUDA_CSRC_PATTERN_MATCHER_LIB)
	@mkdir -p $(@D)
	$(CXX) $(PATTERN_MATCHER_CSRC_CFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(PYBUDA_CSRC_PATTERN_MATCHER_TESTS_LDFLAGS)

$(OBJDIR)/pybuda/csrc/pattern_matcher/tests/%.o: pybuda/csrc/pattern_matcher/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PATTERN_MATCHER_CSRC_CFLAGS) $(CXXFLAGS) $(PYBUDA_CSRC_PATTERN_MATCHER_TESTS_INCLUDES) -c -o $@ $<
