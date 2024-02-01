PYBUDA_CSRC_BALANCER_TESTS = $(TESTDIR)/pybuda/csrc/balancer/tests/balancer_unit_tests
PYBUDA_CSRC_BALANCER_TESTS_SRCS = \
	$(wildcard pybuda/csrc/balancer/tests/*.cpp) \

PYBUDA_CSRC_BALANCER_TESTS_INCLUDES = -Ipybuda/csrc/graph_lib $(PYBUDA_CSRC_BALANCER_INCLUDES)
PYBUDA_CSRC_BALANCER_TESTS_LDFLAGS = -lgtest -lgtest_main -lpthread -l$(PYTHON_VERSION) -lm

PYBUDA_CSRC_BALANCER_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_BALANCER_TESTS_SRCS:.cpp=.o))
PYBUDA_CSRC_BALANCER_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_BALANCER_TESTS_SRCS:.cpp=.d))

# BBE TILE MAPS
PYBUDA_CSRC_TILE_MAPS_LIB     = $(LIBDIR)/libnet2pipe_tile_maps.a
PYBUDA_CSRC_TILE_MAPS_INCLUDES = \
	$(INCDIR)/net2pipe/tile_maps.h \
	$(INCDIR)/net2pipe/tile_maps_common.h \
	$(INCDIR)/net2pipe/net2pipe_logger.h
PYBUDA_CSRC_TILE_MAPS_SRC     = third_party/budabackend/src/net2pipe/src/tile_maps.cpp
PYBUDA_CSRC_TILE_MAPS_OBJ     = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_TILE_MAPS_SRC:.cpp=.o))
PYBUDA_CSRC_TILE_MAPS_DEP     = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_TILE_MAPS_SRC:.cpp=.d))

-include $(PYBUDA_CSRC_TILE_MAPS_DEP)
-include $(PYBUDA_CSRC_BALANCER_TESTS_DEPS)

pybuda/csrc/balancer/tests: $(PYBUDA_CSRC_BALANCER_TESTS)

$(INCDIR)/net2pipe/%.h: third_party/budabackend/src/net2pipe/inc/%.h
	@mkdir -p $(@D)
	cp $^ $@

$(PYBUDA_CSRC_TILE_MAPS_OBJ): $(PYBUDA_CSRC_TILE_MAPS_INCLUDES) $(PYBUDA_CSRC_TILE_MAPS_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(STATIC_LIB_FLAGS) -I$(INCDIR)/net2pipe -c -o $@ $(PYBUDA_CSRC_TILE_MAPS_SRC)

$(PYBUDA_CSRC_LOGGER_OBJ): $(PYBUDA_CSRC_TILE_MAPS_INCLUDES) $(PYBUDA_CSRC_LOGGER_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(STATIC_LIB_FLAGS) -I$(INCDIR)/net2pipe -c -o $@ $(PYBUDA_CSRC_LOGGER_SRC)

$(PYBUDA_CSRC_TILE_MAPS_LIB): $(PYBUDA_CSRC_TILE_MAPS_OBJ) $(PYBUDA_CSRC_LOGGER_OBJ)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(PYBUDA_CSRC_BALANCER_TESTS): $(PYBUDA_CSRC_BALANCER_TESTS_OBJS) $(PYBUDA_CSRC_LIB) $(PYBUDA_CSRC_TILE_MAPS_LIB)
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(PYBUDA_CSRC_BALANCER_TESTS_LDFLAGS)

$(OBJDIR)/pybuda/csrc/balancer/tests/%.o: pybuda/csrc/balancer/tests/%.cpp $(PYBUDA_CSRC_TILE_MAPS_INCLUDES)
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(PYBUDA_CSRC_BALANCER_TESTS_INCLUDES) -c -o $@ $<
