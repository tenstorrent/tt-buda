export BACKEND_ARCH_NAME=grayskull
export ARCH_NAME=grayskull

if command -v bear >/dev/null 2>&1; then
    bear make
else
    make
fi
source build/python_env/bin/activate
source third_party/tvm/enable.sh
set +e
