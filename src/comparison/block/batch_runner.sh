#!/bin/bash
#conda activate rapids-21.06
#which python
#python clustering_metrics_other_baselines.py
nohup /home/zhangli/mydisk-2t/miniconda3/envs/rapids-21.06/bin/python -u clustering_metrics_other_baselines.py --which_model=0 >r1_trimmed_model_trimmed_dataset0.log 2>1&
nohup /home/zhangli/mydisk-2t/miniconda3/envs/rapids-21.06/bin/python -u clustering_metrics_other_baselines.py --which_model=1 >r1_trimmed_model_trimmed_dataset1.log 2>1&
nohup /home/zhangli/mydisk-2t/miniconda3/envs/rapids-21.06/bin/python -u clustering_metrics_other_baselines.py --which_model=2 >r1_trimmed_model_trimmed_dataset2.log 2>1&
nohup /home/zhangli/mydisk-2t/miniconda3/envs/rapids-21.06/bin/python -u clustering_metrics_other_baselines.py --which_model=3 >r1_trimmed_model_trimmed_dataset3.log 2>1&
nohup /home/zhangli/mydisk-2t/miniconda3/envs/rapids-21.06/bin/python -u clustering_metrics_other_baselines.py --which_model=4 >r1_trimmed_model_trimmed_dataset4.log 2>1&
nohup /home/zhangli/mydisk-2t/miniconda3/envs/rapids-21.06/bin/python -u clustering_metrics_other_baselines.py --which_model=5 >r1_trimmed_model_trimmed_dataset5.log 2>1&