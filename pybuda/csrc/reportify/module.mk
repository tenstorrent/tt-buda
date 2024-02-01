PYBUDA_CSRC_REPORTIFY = ${LIBDIR}/libreportify.a
PYBUDA_CSRC_REPORTIFY_SRCS = \
	pybuda/csrc/reportify/reportify.cpp \
	pybuda/csrc/reportify/paths.cpp \
	pybuda/csrc/reportify/to_json.cpp

PYBUDA_CSRC_REPORTIFY_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_REPORTIFY_SRCS:.cpp=.o))
PYBUDA_CSRC_REPORTIFY_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_REPORTIFY_SRCS:.cpp=.d))

PYBUDA_CSRC_REPORTIFY_INCLUDES = $(PYBUDA_CSRC_INCLUDES)

-include $(PYBUDA_CSRC_REPORTIFY_DEPS)

pybuda/csrc/reportify: $(PYBUDA_CSRC_REPORTIFY)

$(PYBUDA_CSRC_REPORTIFY): $(PYBUDA_CSRC_REPORTIFY_OBJS) $(PYBUDA_CSRC_GRAPH_LIB)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/reportify/%.o: pybuda/csrc/reportify/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_REPORTIFY_INCLUDES) -c -o $@ $<

