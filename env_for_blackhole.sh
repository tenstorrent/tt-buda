#!/bin/bash

export LD_LIBRARY_PATH=versim/blackhole/lib:versim/blackhole/lib/ext
export BACKEND_ARCH_NAME=blackhole
export ARCH_NAME=blackhole

SUPPORTS_BEAR=$(command -v bear >/dev/null 2>&1 && echo true || echo false)

if [ "$SUPPORTS_BEAR" = "true" ] && [ "$TT_BUDA_GENERATE_DEPS" = "1" ]; then
    bear -- make
    echo "Compilation database generated with bear"
else
    make
fi

RET=$?
if [[ $RET -eq 0 ]]; then
    source build/python_env/bin/activate
    source third_party/tvm/enable.sh
else
    echo -e "\e[31mPybuda build failed!\e[0m"
    return $RET
fi

set +e
