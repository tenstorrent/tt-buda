export LD_LIBRARY_PATH=third_party/confidential_tenstorrent_modules/versim/wormhole/lib:third_party/confidential_tenstorrent_modules/versim/wormhole/lib/ext
export BACKEND_ARCH_NAME=wormhole
export ARCH_NAME=wormhole

if command -v bear >/dev/null 2>&1; then
    bear make
else
    make
fi
source build/python_env/bin/activate
source third_party/tvm/install.sh
source third_party/tvm/enable.sh
set +e
