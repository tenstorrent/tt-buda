PYBUDA_CSRC_AUTOGRAD = ${LIBDIR}/libautograd.a
PYBUDA_CSRC_AUTOGRAD_SRCS = \
	pybuda/csrc/autograd/autograd.cpp \
	pybuda/csrc/autograd/binding.cpp \
	pybuda/csrc/autograd/python_bindings.cpp

PYBUDA_CSRC_AUTOGRAD_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_AUTOGRAD_SRCS:.cpp=.o))
PYBUDA_CSRC_AUTOGRAD_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_AUTOGRAD_SRCS:.cpp=.d))

PYBUDA_CSRC_AUTOGRAD_INCLUDES = $(PYBUDA_CSRC_INCLUDES)

-include $(PYBUDA_CSRC_AUTOGRAD_DEPS)

pybuda/csrc/autograd: $(PYBUDA_CSRC_AUTOGRAD)

$(PYBUDA_CSRC_AUTOGRAD): $(PYBUDA_CSRC_AUTOGRAD_OBJS) $(PYBUDA_CSRC_GRAPH_LIB)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/autograd/%.o: pybuda/csrc/autograd/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_AUTOGRAD_INCLUDES) -c -o $@ $<

