#!/bin/bash
if [[ $1 == "cifar_60" ]] || [[ $1 == "all" ]]; then
python -u src/experiments/binary/test_slope_cifar.py -pair="6-0" -clf="svm" -trigger_size=8
python -u src/experiments/binary/test_slope_cifar.py -pair="6-0" -clf="logistic" -trigger_size=8
python -u src/experiments/binary/test_slope_cifar.py -pair="6-0" -clf="ridge" -trigger_size=8
python -u src/experiments/binary/test_slope_cifar.py -pair="6-0" -clf="svm-rbf" -trigger_size=8
fi

if [[ $1 == "cifar_25" ]] || [[ $1 == "all" ]]; then
python -u src/experiments/binary/test_slope_cifar.py -pair="2-5" -clf="svm" -trigger_size=8
python -u src/experiments/binary/test_slope_cifar.py -pair="2-5" -clf="logistic" -trigger_size=8
python -u src/experiments/binary/test_slope_cifar.py -pair="2-5" -clf="ridge" -trigger_size=8
python -u src/experiments/binary/test_slope_cifar.py -pair="2-5" -clf="svm-rbf" -trigger_size=8
fi

if [[ $1 == "cifar_09" ]] || [[ $1 == "all" ]]; then
python -u src/experiments/binary/test_slope_cifar.py -pair="0-9" -clf="svm" -trigger_size=8
python -u src/experiments/binary/test_slope_cifar.py -pair="0-9" -clf="logistic" -trigger_size=8
python -u src/experiments/binary/test_slope_cifar.py -pair="0-9" -clf="ridge" -trigger_size=8
python -u src/experiments/binary/test_slope_cifar.py -pair="0-9" -clf="svm-rbf" -trigger_size=8
fi


if [[ $1 == "double_trigger" ]]; then
python -u src/experiments/binary/test_slope_cifar.py -pair="6-0" -clf="svm" -trigger_size=16
python -u src/experiments/binary/test_slope_cifar.py -pair="6-0" -clf="logistic" -trigger_size=16
python -u src/experiments/binary/test_slope_cifar.py -pair="6-0" -clf="ridge" -trigger_size=16
python -u src/experiments/binary/test_slope_cifar.py -pair="6-0" -clf="svm-rbf" -trigger_size=16
fi
