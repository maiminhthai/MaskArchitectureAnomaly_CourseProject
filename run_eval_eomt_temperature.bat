@echo off
call conda activate eomt

echo Running Evaluation with Temperature Scaling for eomt_cityscapes.bin

python eval/evalAnomalyEomt_temperature.py --input Validation_Dataset\FS_LostFound_full\images\*.png --ckpt_path trained_eomt\eomt_cityscapes.bin --result_file temperature_result\results_eomt_maxlogit_temp.txt
python eval/evalAnomalyEomt_temperature.py --input Validation_Dataset\fs_static\images\*.jpg --ckpt_path trained_eomt\eomt_cityscapes.bin --result_file temperature_result\results_eomt_maxlogit_temp.txt
python eval/evalAnomalyEomt_temperature.py --input Validation_Dataset\RoadAnomaly\images\*.jpg --ckpt_path trained_eomt\eomt_cityscapes.bin --result_file temperature_result\results_eomt_maxlogit_temp.txt
python eval/evalAnomalyEomt_temperature.py --input Validation_Dataset\RoadAnomaly21\images\*.png --ckpt_path trained_eomt\eomt_cityscapes.bin --result_file temperature_result\results_eomt_maxlogit_temp.txt
python eval/evalAnomalyEomt_temperature.py --input Validation_Dataset\RoadObsticle21\images\*.webp --ckpt_path trained_eomt\eomt_cityscapes.bin --result_file temperature_result\results_eomt_maxlogit_temp.txt

echo All evaluations completed.
