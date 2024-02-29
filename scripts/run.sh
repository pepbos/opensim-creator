#!/usr/bin/env bash

set -xeuo pipefail
WORKSPACE=$(dirname $(dirname $(realpath "$0")))
echo "WORKSPACE = $WORKSPACE"

# Uncomment for reconfiguring.
# export OSC_BASE_BUILD_TYPE=RelWithDebInfo
# env -C $WORKSPACE CC=clang-14 CXX=clang++-14 OSC_BUILD_CONCURRENCY=$(nproc) ./scripts/build_debian-buster.sh

env -C $WORKSPACE cmake --build osc-build -j$(nproc)

env -C $WORKSPACE ln -sf osc-build/compile_commands.json compile_commands.json

env -C $WORKSPACE ./osc-build/apps/osc/osc
