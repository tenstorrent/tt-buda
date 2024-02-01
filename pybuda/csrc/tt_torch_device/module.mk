PYBUDA_CSRC_TT_TORCH_DEVICE_LIB = ${LIBDIR}/libtt_torch_device.a
PYBUDA_CSRC_TT_TORCH_DEVICE_SRCS = \
	pybuda/csrc/tt_torch_device/tt_device.cpp \
	pybuda/csrc/tt_torch_device/torch_device_impl.cpp \
	pybuda/csrc/tt_torch_device/python_bindings.cpp

PYBUDA_CSRC_TT_TORCH_DEVICE_INCLUDES = $(PYBUDA_CSRC_INCLUDES)

PYBUDA_CSRC_TT_TORCH_DEVICE_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_TT_TORCH_DEVICE_SRCS:.cpp=.o))
PYBUDA_CSRC_TT_TORCH_DEVICE_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_TT_TORCH_DEVICE_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_TT_TORCH_DEVICE_DEPS)

pybuda/csrc/tt_torch_device: $(PYBUDA_CSRC_TT_TORCH_DEVICE_LIB);

$(PYBUDA_CSRC_TT_TORCH_DEVICE_LIB): $(PYBUDA_CSRC_TT_TORCH_DEVICE_OBJS) $(PYBUDA_CSRC_BACKENDAPI_LIB)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/tt_torch_device/%.o: pybuda/csrc/tt_torch_device/%.cpp python_env
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) $(PYBUDA_CSRC_TT_TORCH_DEVICE_INCLUDES) -c -o $@ $<
