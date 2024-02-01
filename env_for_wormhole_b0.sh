export LD_LIBRARY_PATH=versim/wormhole_b0/lib:versim/wormhole_b0/lib/ext
export BACKEND_ARCH_NAME=wormhole_b0
export ARCH_NAME=wormhole_b0

if command -v bear >/dev/null 2>&1; then
    bear make
else
    make
fi
source build/python_env/bin/activate
source third_party/tvm/enable.sh
set +e
