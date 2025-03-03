pip install -r requirements.txt 
pip install dvc
pip install dvc-s3
pip install hydra-optuna-sweeper --upgrade
pip install hydra_colorlog --upgrade
pip install hydra-joblib-launcher --upgrade
pip install aim

dvc remote modify --local s3_data_store access_key_id <access_key_id>

dvc remote modify --local s3_data_store secret_access_key <secret_access_key>

dvc pull

python3 src/train.py  --multirun  hparam=mobilenet_v3_food_classifier_hparam experiment=mobilenet_v3_food_classifier   +trainer.log_every_n_steps=1  logger.aim.run_name=optuna_food_classiftier_mobilenetv3_run