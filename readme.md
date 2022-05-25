
# steps
1. installation
- conda env create --file environment.yaml
- conda activate fog_dissipation

2. download dataset
- aws configure # id, password 동일하게 입력
- aws configure set default.s3.signature_version s3v4
- aws --endpoint-url http://oldgpu:9000 s3 cp s3://seafog-dissipation/data data/clean --recursive

3. running
- python steps/step1_report_calc_stat.py
stat summary 계산

- python steps/step2_report_pre_importance.py
pre importance 계산

- python steps/step3_train_ag.py
autogluon automl 모델 돌리는 스텝. 뒤에 옵션 없으면 conf/config.yaml 파일 안의 station, pred_hour 만 돌아감
python steps/step3_train_ag.py --multirun station=SF_0002,SF_0003,SF_0004,SF_0005,SF_0006,SF_0007,SF_0008,SF_0009 pred_hour=1,2,3
하면 다 돌릴 수 있음

- python steps/step4_post_train.py
위에  결과가 data/result/ 폴더에 저장되는데, 그 것들 다 읽어서 table로 만들어주는 코드







