#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/invariant_solver.prototxt 

