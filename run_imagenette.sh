#!/bin/bash
if [[ $1 == "imagenette_60" ]] || [[ $1 == "all" ]]; then
python -u src/experiments/binary/test_slope_imagenette.py -pair="6-0" -clf="svm" -trigger_type="invisible"
python -u src/experiments/binary/test_slope_imagenette.py -pair="6-0" -clf="logistic" -trigger_type="invisible"
python -u src/experiments/binary/test_slope_imagenette.py -pair="6-0" -clf="ridge" -trigger_type="invisible"
python -u src/experiments/binary/test_slope_imagenette.py -pair="6-0" -clf="svm-rbf" -trigger_type="invisible"
fi

if [[ $1 == "imagenette_25" ]] || [[ $1 == "all" ]]; then
python -u src/experiments/binary/test_slope_imagenette.py -pair="2-5" -clf="svm" -trigger_type="invisible"
python -u src/experiments/binary/test_slope_imagenette.py -pair="2-5" -clf="logistic" -trigger_type="invisible"
python -u src/experiments/binary/test_slope_imagenette.py -pair="2-5" -clf="ridge" -trigger_type="invisible"
python -u src/experiments/binary/test_slope_imagenette.py -pair="2-5" -clf="svm-rbf" -trigger_type="invisible"
fi

if [[ $1 == "imagenette_09" ]] || [[ $1 == "all" ]]; then
python -u src/experiments/binary/test_slope_imagenette.py -pair="0-9" -clf="svm" -trigger_type="invisible"
python -u src/experiments/binary/test_slope_imagenette.py -pair="0-9" -clf="logistic" -trigger_type="invisible"
python -u src/experiments/binary/test_slope_imagenette.py -pair="0-9" -clf="ridge" -trigger_type="invisible"
python -u src/experiments/binary/test_slope_imagenette.py -pair="0-9" -clf="svm-rbf" -trigger_type="invisible"
fi


if [[ $1 == "imagenette_full" ]] || [[ $1 == "all" ]]; then
python -u src/experiments/nn/backdoor_slope.py
fi
