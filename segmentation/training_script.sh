#!/bin/bash

# 8 bands (All + NDVI) composition
nohup python deeplab.py --composition 8 7 6 5 4 3 2 1 \
--composition_name "All+NDVI" \
--batch_size 32 \
--encoder "efficientnet-b0" \
--experiment_name "Encoder_Comparison"

wait

nohup python deeplab.py --composition 8 7 6 5 4 3 2 1 \
--composition_name "All+NDVI" \
--batch_size 32 \
--encoder "resnet34" \
--experiment_name "Encoder_Comparison"

wait

nohup python deeplab.py --composition 8 7 6 5 4 3 2 1 \
--composition_name "All+NDVI" \
--batch_size 32 \
--encoder "resnet101" \

# 3 bands (651) composition
nohup python deeplab.py --composition 6 5 1 \
--composition_name "651" \
--batch_size 32 \
--encoder "efficientnet-b0" \
--experiment_name "Encoder_Comparison"

wait

nohup python deeplab.py --composition 6 5 1 \
--composition_name "651" \
--batch_size 32 \
--encoder "resnet34" \
--experiment_name "Encoder_Comparison"

wait

nohup python deeplab.py --composition 6 5 1 \
--composition_name "651" \
--batch_size 32 \
--encoder "resnet101" \
--experiment_name "Encoder_Comparison"

wait

# Single band (6) composition
nohup python deeplab.py --composition 6 \
--composition_name "6" \
--batch_size 32 \
--encoder "efficientnet-b0" \
--experiment_name "Encoder_Comparison"

wait

nohup python deeplab.py --composition 6 \
--composition_name "6" \
--batch_size 32 \
--encoder "resnet34" \
--experiment_name "Encoder_Comparison"

wait

nohup python deeplab.py --composition 6 \
--composition_name "6" \
--batch_size 32 \
--encoder "resnet101" \
--experiment_name "Encoder_Comparison"

wait