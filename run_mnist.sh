#!/bin/bash
if [[ $1 == "mnist_71" ]] || [[ $1 == "all" ]]; then
python -u src/experiments/binary/test_slope_mnist.py -pair="7-1" -clf="svm"  -trigger_size=3
python -u src/experiments/binary/test_slope_mnist.py -pair="7-1" -clf="logistic"  -trigger_size=3
python -u src/experiments/binary/test_slope_mnist.py -pair="7-1" -clf="ridge"  -trigger_size=3
python -u src/experiments/binary/test_slope_mnist.py -pair="7-1" -clf="svm-rbf"  -trigger_size=3
fi

if [[ $1 == "mnist_30" ]] || [[ $1 == "all" ]]; then
python -u src/experiments/binary/test_slope_mnist.py -pair="3-0" -clf="svm"  -trigger_size=3
python -u src/experiments/binary/test_slope_mnist.py -pair="3-0" -clf="logistic"  -trigger_size=3
python -u src/experiments/binary/test_slope_mnist.py -pair="3-0" -clf="ridge"  -trigger_size=3
python -u src/experiments/binary/test_slope_mnist.py -pair="3-0" -clf="svm-rbf"  -trigger_size=3
fi

if [[ $1 == "mnist_52" ]] || [[ $1 == "all" ]]; then
python -u src/experiments/binary/test_slope_mnist.py -pair="5-2" -clf="svm"  -trigger_size=3
python -u src/experiments/binary/test_slope_mnist.py -pair="5-2" -clf="logistic"  -trigger_size=3
python -u src/experiments/binary/test_slope_mnist.py -pair="5-2" -clf="ridge"  -trigger_size=3
python -u src/experiments/binary/test_slope_mnist.py -pair="5-2" -clf="svm-rbf"  -trigger_size=3
fi


if [[ $1 == "double_trigger" ]]; then
python -u src/experiments/binary/test_slope_mnist.py -pair="7-1" -clf="svm"  -trigger_size=6
python -u src/experiments/binary/test_slope_mnist.py -pair="7-1" -clf="logistic"  -trigger_size=6
python -u src/experiments/binary/test_slope_mnist.py -pair="7-1" -clf="ridge"  -trigger_size=6
python -u src/experiments/binary/test_slope_mnist.py -pair="7-1" -clf="svm-rbf"  -trigger_size=6
fi
