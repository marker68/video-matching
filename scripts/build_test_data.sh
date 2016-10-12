#!/usr/bin/env bash

for i in `seq 1 3`; do
    python keras/create_data.py create_test_${i}.yaml
done
