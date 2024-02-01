# Every variable in subdir must be prefixed with subdir (emulating a namespace)

PYBUDA_CSRC_SCHEDULER_LIB = $(LIBDIR)/libscheduler.a
PYBUDA_CSRC_SCHEDULER_SRCS = \
	pybuda/csrc/scheduler/scheduler.cpp \
	pybuda/csrc/scheduler/longest_path.cpp \
	pybuda/csrc/scheduler/utils.cpp \
	pybuda/csrc/scheduler/interactive_scheduler.cpp \
	pybuda/csrc/scheduler/python_bindings.cpp

PYBUDA_CSRC_SCHEDULER_INCLUDES = $(PYBUDA_CSRC_INCLUDES)

PYBUDA_CSRC_SCHEDULER_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_SCHEDULER_SRCS:.cpp=.o))
PYBUDA_CSRC_SCHEDULER_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_SCHEDULER_SRCS:.cpp=.d))



-include $(PYBUDA_CSRC_SCHEDULER_DEPS)

SCHEDULER_CSRC_CFLAGS = $(PYBUDA_CSRC_CFLAGS)

# Each module has a top level target as the entrypoint which must match the subdir name
pybuda/csrc/scheduler: $(PYBUDA_CSRC_SCHEDULER_LIB)

$(PYBUDA_CSRC_SCHEDULER_LIB): $(PYBUDA_CSRC_SCHEDULER_OBJS) $(PYBUDA_CSRC_GRAPH_LIB) 
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/scheduler/%.o: pybuda/csrc/scheduler/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(SCHEDULER_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_SCHEDULER_INCLUDES) -c -o $@ $<

