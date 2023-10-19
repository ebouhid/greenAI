nohup python deeplab.py --composition 6 5 1 \
--composition_name "651" \
--batch_size 64 \
--encoder "resnet101" \
--experiment_name "Encoder_Comparison"

nohup python deeplab.py --composition 6 5 1 \
--composition_name "6" \
--batch_size 64 \
--encoder "resnet34" \
--experiment_name "Encoder_Comparison"

nohup python deeplab.py --composition 6 5 1 \
--composition_name "6" \
--batch_size 64 \
--encoder "resnet101" \
--experiment_name "Encoder_Comparison"