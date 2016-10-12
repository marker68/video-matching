#!/bin/bash

./scripts/ranking.sh ./model/y2t_mean_5x64x64_v2t_weights.21-0.01-0.01.hdf5 a model/d2v_yt2t+tvv2t_100.model data/a_tvv2t_test_yt2t.h5 10 64
./scripts/ranking.sh ./model/y2t_mean_5x64x64_v2t_weights.21-0.01-0.01.hdf5 a model/d2v_yt2t+tvv2t_100.model data/b2_tvv2t_test_yt2t.h5 10 64

