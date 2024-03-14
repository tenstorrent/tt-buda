# Every variable in subdir must be prefixed with subdir (emulating a namespace)

PYBUDA_CSRC_SHARED_UTILS_LIB = $(LIBDIR)/libsharedutils.a
PYBUDA_CSRC_SHARED_UTILS_SRCS += \
	pybuda/csrc/shared_utils/placement_printer.cpp \
	pybuda/csrc/shared_utils/pretty_table.cpp \
	pybuda/csrc/shared_utils/sparse_matmul_utils.cpp \
	pybuda/csrc/shared_utils/string_extension.cpp

PYBUDA_CSRC_SHARED_UTILS_INCLUDES = $(PYBUDA_CSRC_INCLUDES)

PYBUDA_CSRC_SHARED_UTILS_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_SHARED_UTILS_SRCS:.cpp=.o))
PYBUDA_CSRC_SHARED_UTILS_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_SHARED_UTILS_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_SHARED_UTILS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
pybuda/csrc/shared_utils: $(PYBUDA_CSRC_SHARED_UTILS_LIB)

$(PYBUDA_CSRC_SHARED_UTILS_LIB): $(PYBUDA_CSRC_SHARED_UTILS_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/shared_utils/%.o: pybuda/csrc/shared_utils/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_SHARED_UTILS_INCLUDES) -c -o $@ $<
