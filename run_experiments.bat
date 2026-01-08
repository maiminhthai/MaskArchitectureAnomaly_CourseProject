@echo off
call conda activate eomt

echo Running Experiment 1: Standard Resolution (1024x1024)
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\eomt_cityscapes.bin --result_dir result-1024x1024
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=0.005-no_load.ckpt --result_dir result-1024x1024
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=0.01-no_load.ckpt --result_dir result-1024x1024
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=0.04-no_load.ckpt --result_dir result-1024x1024
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=0.3-no_load.ckpt --result_dir result-1024x1024
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=0.7-no_load.ckpt --result_dir result-1024x1024
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=1.2-no_load.ckpt --result_dir result-1024x1024

echo Running Experiment 2: Upscale Resolution (1024x1024-upscale)
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\eomt_cityscapes.bin --result_dir result-1024x1024-upscale
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=0.005-no_load-upscale.ckpt --result_dir result-1024x1024-upscale
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=0.01-no_load-upscale.ckpt --result_dir result-1024x1024-upscale
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=0.04-no_load-upscale.ckpt --result_dir result-1024x1024-upscale
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=0.3-no_load-upscale.ckpt --result_dir result-1024x1024-upscale
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=0.7-no_load-upscale.ckpt --result_dir result-1024x1024-upscale
python run_eval_experiment.py --dataset_root .\Validation_Dataset\ --ckpt_path .\trained_eomt\no_lora-t=1.2-no_load-upscale.ckpt --result_dir result-1024x1024-upscale

echo All experiments completed.
