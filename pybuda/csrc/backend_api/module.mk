# Every variable in subdir must be prefixed with subdir (emulating a namespace)

BACKEND_CONFIG ?= release
BACKEND_ARCH_NAME ?= grayskull
BACKEND_CC ?= gcc
BACKEND_CXX ?= g++

BACKEND_INCLUDES =  -Ithird_party/budabackend

BUDABACKEND_LIBDIR = third_party/budabackend/build/lib
BUDABACKEND_LIB = $(BUDABACKEND_LIBDIR)/libtt.so
BUDABACKEND_DEVICE = $(BUDABACKEND_LIBDIR)/libdevice.so
BUDABACKEND_NET2PIPE = third_party/budabackend/build/bin/net2pipe
BUDABACKEND_PIPEGEN = third_party/budabackend/build/bin/pipegen2
BUDABACKEND_BLOBGEN = third_party/budabackend/build/bin/blobgen2

PYBUDA_CSRC_BACKENDAPI_LIB = $(LIBDIR)/libbackend_api.a
PYBUDA_CSRC_BACKENDAPI_SRCS += \
	pybuda/csrc/backend_api/backend_api.cpp \
	pybuda/csrc/backend_api/arch_type.cpp

PYBUDA_CSRC_BACKENDAPI_INCLUDES = $(PYBUDA_CSRC_INCLUDES) $(BACKEND_INCLUDES)

PYBUDA_CSRC_BACKENDAPI_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_BACKENDAPI_SRCS:.cpp=.o))
PYBUDA_CSRC_BACKENDAPI_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_BACKENDAPI_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_BACKENDAPI_DEPS)

third_party/budabackend/clean:
	#ROOT=third_party/budabackend $(MAKE) -C third_party/budabackend clean
	cd third_party/budabackend; rm -rf build
	rm -f $(SUBMODULESDIR)/third_party/budabackend.build

third_party/budabackend: $(SUBMODULESDIR)/third_party/budabackend.build ;

DEVICE_VERSIM_INSTALL_ROOT ?= third_party/budabackend
$(SUBMODULESDIR)/third_party/budabackend.build: $(SUBMODULESDIR)/third_party/budabackend.checkout
	CC=$(BACKEND_CC) CXX=$(BACKEND_CXX) CONFIG=$(BACKEND_CONFIG) ARCH_NAME=$(BACKEND_ARCH_NAME) DEVICE_VERSIM_INSTALL_ROOT=$(DEVICE_VERSIM_INSTALL_ROOT) ROOT=$(PWD)/third_party/budabackend $(MAKE) -C third_party/budabackend backend build_hw dbd
	touch $@

.PHONY: third_party/budabackend/netlist_analyzer
third_party/budabackend/netlist_analyzer:
	CONFIG=$(BACKEND_CONFIG) ARCH_NAME=$(BACKEND_ARCH_NAME) DEVICE_VERSIM_INSTALL_ROOT=$(DEVICE_VERSIM_INSTALL_ROOT) ROOT=$(PWD)/third_party/budabackend $(MAKE) -C third_party/budabackend netlist_analyzer/tests

$(BUDABACKEND_DEVICE): third_party/budabackend ;
$(BUDABACKEND_LIB):  third_party/budabackend ;
$(BUDABACKEND_NET2PIPE): third_party/budabackend ;
$(BUDABACKEND_PIPEGEN): third_party/budabackend ;
$(BUDABACKEND_BLOBGEN): third_party/budabackend ;

third_party/budabackend/src/net2pipe: $(BUDABACKEND_NET2PIPE) $(BUDABACKEND_PIPEGEN) $(BUDABACKEND_BLOBGEN) ;

# Each module has a top level target as the entrypoint which must match the subdir name
pybuda/csrc/backend_api: $(PYBUDA_CSRC_BACKENDAPI_LIB) $(BUDABACKEND_LIB) $(BUDABACKEND_DEVICE) $(PYBUDA_CSRC_SHARED_UTILS_LIB) ;

$(PYBUDA_CSRC_BACKENDAPI_LIB): $(PYBUDA_CSRC_BACKENDAPI_OBJS) $(BUDABACKEND_LIB) $(BUDABACKEND_DEVICE) $(PYBUDA_CSRC_SHARED_UTILS_LIB)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/backend_api/%.o: pybuda/csrc/backend_api/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_BACKENDAPI_INCLUDES) -c -o $@ $<

