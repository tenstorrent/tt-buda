# Every variable in subdir must be prefixed with subdir (emulating a namespace)

PYBUDA_CSRC_PATTERN_MATCHER_LIB = $(LIBDIR)/libpattern_matcher.a
PYBUDA_CSRC_PATTERN_MATCHER_SRCS = \
	pybuda/csrc/pattern_matcher/pattern_matcher.cpp \
	pybuda/csrc/pattern_matcher/boost_lowering.cpp \
	pybuda/csrc/pattern_matcher/python_bindings.cpp

PYBUDA_CSRC_PATTERN_MATCHER_INCLUDES = $(PYBUDA_CSRC_INCLUDES)

PYBUDA_CSRC_PATTERN_MATCHER_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PATTERN_MATCHER_SRCS:.cpp=.o))
PYBUDA_CSRC_PATTERN_MATCHER_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PATTERN_MATCHER_SRCS:.cpp=.d))



-include $(PYBUDA_CSRC_PATTERN_MATCHER_DEPS)

PATTERN_MATCHER_CSRC_CFLAGS = $(PYBUDA_CSRC_CFLAGS)

# Each module has a top level target as the entrypoint which must match the subdir name
pybuda/csrc/pattern_matcher: $(PYBUDA_CSRC_PATTERN_MATCHER_LIB)

$(PYBUDA_CSRC_PATTERN_MATCHER_LIB): $(PYBUDA_CSRC_GRAPH_LIB) $(PYBUDA_CSRC_PATTERN_MATCHER_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/pattern_matcher/%.o: pybuda/csrc/pattern_matcher/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PATTERN_MATCHER_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_PATTERN_MATCHER_INCLUDES) -c -o $@ $<

include pybuda/csrc/pattern_matcher/tests/module.mk
