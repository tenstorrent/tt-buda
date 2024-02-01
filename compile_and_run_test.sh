#!/bin/bash

export BACKEND_ARCH_NAME=wormhole_b0
git reset --hard
git submodule update --init --recursive
make pybuda
if [ ! $? -eq 0 ]
then
    echo "Make pybuda failed"
    make clean_no_python
    cd third_party/budabackend
    git clean -fxd
    cd ../..
    make pybuda
    if [ ! $? -eq 0 ]
    then
        echo "Clean build failed bad rev 1"
        return 1
    fi
fi
source build/python_env/bin/activate
source third_party/tvm/install.sh
source third_party/tvm/enable.sh
echo "Evaluating $*"
eval "$*"
if [ ! $? -eq 0 ]
then
    echo "Test failed"
    exit 1
fi

echo "All passed"
exit 0
