# Every variable in subdir must be prefixed with subdir (emulating a namespace)

PYBUDA_CSRC_PLACER_LIB = $(LIBDIR)/libplacer.a
PYBUDA_CSRC_PLACER_SRCS = \
	pybuda/csrc/placer/allocator_utils.cpp \
	pybuda/csrc/placer/best_fit_allocator.cpp \
	pybuda/csrc/placer/chip_id_assignment.cpp \
	pybuda/csrc/placer/dram.cpp \
	pybuda/csrc/placer/dram_logger.cpp \
	pybuda/csrc/placer/dram_allocator.cpp \
	pybuda/csrc/placer/evaluator.cpp \
	pybuda/csrc/placer/grid_placer.cpp \
	pybuda/csrc/placer/host_memory.cpp \
	pybuda/csrc/placer/host_memory_allocator.cpp \
	pybuda/csrc/placer/interactive_placer.cpp \
	pybuda/csrc/placer/lowering_utils.cpp \
	pybuda/csrc/placer/lower_to_placer.cpp \
	pybuda/csrc/placer/placer.cpp \
	pybuda/csrc/placer/pre_epoch_passes.cpp \
	pybuda/csrc/placer/post_epoch_passes.cpp \
	pybuda/csrc/placer/python_bindings.cpp \
	pybuda/csrc/placer/utils.cpp

PYBUDA_CSRC_PLACER_INCLUDES = $(PYBUDA_CSRC_INCLUDES)

PYBUDA_CSRC_PLACER_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PLACER_SRCS:.cpp=.o))
PYBUDA_CSRC_PLACER_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_PLACER_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_PLACER_DEPS)

PLACER_CSRC_CFLAGS = $(PYBUDA_CSRC_CFLAGS)

# Each module has a top level target as the entrypoint which must match the subdir name
pybuda/csrc/placer: $(PYBUDA_CSRC_PLACER_LIB)

$(PYBUDA_CSRC_PLACER_LIB):  $(PYBUDA_CSRC_PLACER_OBJS) $(PYBUDA_CSRC_GRAPH_LIB) $(PYBUDA_CSRC_SCHEDULER_LIB) $(PYBUDA_CSRC_GRAPH_LIB)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/placer/%.o: pybuda/csrc/placer/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PLACER_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_PLACER_INCLUDES) -c -o $@ $<

include pybuda/csrc/placer/tests/module.mk
