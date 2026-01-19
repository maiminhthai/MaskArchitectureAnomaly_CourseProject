@echo off
call conda activate py36

echo Running Evaluation with Temperature Scaling for erfnet_pretrained.path

python eval/evalAnomaly_temperature.py --input Validation_Dataset\FS_LostFound_full\images\*.png --loadDir trained_models\ --result_file temperature_result\results_erfnet_MSP_temp.txt
python eval/evalAnomaly_temperature.py --input Validation_Dataset\fs_static\images\*.jpg --loadDir trained_models\ --result_file temperature_result\results_erfnet_MSP_temp.txt
python eval/evalAnomaly_temperature.py --input Validation_Dataset\RoadAnomaly\images\*.jpg --loadDir trained_models\ --result_file temperature_result\results_erfnet_MSP_temp.txt
python eval/evalAnomaly_temperature.py --input Validation_Dataset\RoadAnomaly21\images\*.png --loadDir trained_models\ --result_file temperature_result\results_erfnet_MSP_temp.txt
python eval/evalAnomaly_temperature.py --input Validation_Dataset\RoadObsticle21\images\*.webp --loadDir trained_models\ --result_file temperature_result\results_erfnet_MSP_temp.txt

echo All evaluations completed.