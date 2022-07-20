## **Neural Basis Model (NBM) and Scalable Polynomial Additive Model (SPAM)**

Training and evaluating NBM and SPAM for interpretable machine learning.

### Library Setup

To setup the library run:
```
git clone git@github.com:facebookresearch/nbm-spam.git
cd nbm-spam/
conda create --name nbm_spam python=3.9
conda activate nbm_spam
pip install -r requirements.txt
pip install -e .
```

Test whether the setup was succesful:
```
(nbm_spam) ~/nbm-spam$ python
Python 3.9.12 (main, Apr  5 2022, 06:56:58)
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import nbm_spam
>>>
```

### Neural Basis Model (NBM)

> **Note**: Optimal hyper-parameters were tuned on a server with 8 GPUs. If you want to run on, e.g. single GPU, change `gpus:8` to `gpus:1` in `nbm_spam/config/mode/local.yaml`. Note also that using less GPUs makes the global batch size smaller, and it might require adjusting the learning rate to reach the same performance as reported in the paper.

Run NBM training on CA Housing:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=california_housing.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/ca_housing/nbm/  ++datamodule.max_epochs=1000  ++datamodule.batch_size=1024  ++tabular_prediction_module.model=nbm  ++tabular_prediction_module='{learning_rate:0.001956,weight_decay:1.568e-05,model_params:{nary:null,num_bases:100,hidden_dims:[256,128,128],num_subnets:1,dropout:0.0,bases_dropout:0.05,batchnorm:True,output_penalty:0.0001439}}'
```

Run NBM order 2 training on CA Housing:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=california_housing.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/ca_housing/nb2m/  ++datamodule.max_epochs=1000  ++datamodule.batch_size=1024  ++tabular_prediction_module.model=nbm  ++tabular_prediction_module='{learning_rate:0.001902,weight_decay:7.483e-09,model_params:{nary:[1,2],num_bases:200,hidden_dims:[256,128,128],num_subnets:1,dropout:0.0,bases_dropout:0.05,batchnorm:True,output_penalty:1.778e-06}}'
```

Run NBM training on CoverType:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=covtype.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/covtype/nbm/  ++datamodule.max_epochs=500  ++datamodule.batch_size=1024  ++tabular_prediction_module.model=nbm  ++tabular_prediction_module='{learning_rate:0.0199,weight_decay:5.931e-07,model_params:{nary:null,num_bases:100,hidden_dims:[256,128,128],num_subnets:1,dropout:0.0,bases_dropout:0.0,batchnorm:True,output_penalty:0.05533}}'
```

Run NBM order 2 training on CoverType:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=covtype.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/covtype/nb2m/  ++datamodule.max_epochs=500  ++datamodule.batch_size=512  ++tabular_prediction_module.model=nbm  ++tabular_prediction_module='{learning_rate:0.002681,weight_decay:1.66e-07,model_params:{nary:[1,2],num_bases:200,hidden_dims:[256,128,128],num_subnets:1,dropout:0.0,bases_dropout:0.00,batchnorm:True,output_penalty:0.001545}}'
```

Run NBM with sparse optimization training on Newsgroups:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=newsgroups.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/newsgroups/nbm/  ++datamodule.max_epochs=500  ++datamodule.batch_size=512  ++tabular_prediction_module.model=nbm_sparse  ++tabular_prediction_module='{learning_rate:0.0003133,weight_decay:1.593e-08,model_params:{nary:null,num_bases:100,hidden_dims:[256,128,128],dropout:0.1,bases_dropout:0.3,batchnorm:True,output_penalty:4.578,nary_ignore_input:0.0}}'
```

### Scalable Polynomial Additive Model (SPAM)

> **Note**: Optimal hyper-parameters were tuned on a server with 8 GPUs. If you want to run on, e.g. single GPU, change `gpus:8` to `gpus:1` in `nbm_spam/config/mode/local.yaml`. Note also that using less GPUs makes the global batch size smaller, and it might require adjusting the learning rate to reach the same performance as reported in the paper.


Run SPAM order 2 training on CA Housing:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=california_housing.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/ca_housing/spam  ++datamodule.batch_size=1024  ++datamodule.max_epochs=1000  ++tabular_prediction_module.model=spam  ++tabular_prediction_module='{learning_rate:0.05661,weight_decay:1.365e-8,model_params:{ranks:[400],dropout:0.216}}'
```

Run SPAM order 3 training on CA Housing:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=california_housing.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/ca_housing/spam  ++datamodule.batch_size=1024  ++datamodule.max_epochs=1000  ++tabular_prediction_module.model=spam  ++tabular_prediction_module='{learning_rate:0.05377,weight_decay:0.00001888,model_params:{ranks:[800,200],dropout:0.4319}}'
```

Run SPAM order 2 training on CoverType:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=covtype.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/covtype/spam  ++datamodule.batch_size=1024  ++datamodule.max_epochs=1000  ++tabular_prediction_module.model=spam  ++tabular_prediction_module='{learning_rate:0.007006,weight_decay:3.448e-7,model_params:{ranks:[600],dropout:0.1246}}'
```

Run SPAM order 3 training on CoverType:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=covtype.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/covtype/spam  ++datamodule.batch_size=1024  ++datamodule.max_epochs=1000  ++tabular_prediction_module.model=spam  ++tabular_prediction_module='{learning_rate:0.008929,weight_decay:9.783e-11,model_params:{ranks:[400,800],dropout:0}}'
```

Run SPAM order 2 training on Newsgroups:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=newsgroups.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/newsgroups/spam  ++datamodule.batch_size=1024  ++datamodule.max_epochs=1000  ++tabular_prediction_module.model=spam  ++tabular_prediction_module='{learning_rate:0.05161,weight_decay:7.178e-12,model_params:{ranks:[200],dropout:0.486}}'
```

Run SPAM order 3 training on Newsgroups:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=newsgroups.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/newsgroups/spam  ++datamodule.batch_size=1024  ++datamodule.max_epochs=1000  ++tabular_prediction_module.model=spam  ++tabular_prediction_module='{learning_rate:0.00007742,weight_decay:5.006e-11,model_params:{ranks:[400,100],dropout:0.4905}}'
```

### Neural Basis Model with Scalable Polynomial Additive Model (NBM-SPAM)
> **Note**: This is just an example run command, NOT optimal hyper-parameters. It will be updated soon.

Run NBM-SPAM order 2 training on CA Housing:
```
python nbm_spam/train_tabular.py -m  --config-path=config  --config-name=california_housing.yaml  +mode=local  hydra.sweep.dir=$HOME/local_runs/ca_housing/nbm/  ++datamodule.max_epochs=1000  ++datamodule.batch_size=1024  ++tabular_prediction_module.model=nbm  ++tabular_prediction_module='{learning_rate:0.001,weight_decay:1e-07,model_params:{nary:null,num_bases:100,hidden_dims:[256,128,128],num_subnets:1,dropout:0.0,bases_dropout:0.05,batchnorm:True,output_penalty:0.0002,polynomial:{ranks:[400],dropout:0.5}}}'
```

### References

Neural Basis Model (NBM) [[pdf](https://arxiv.org/pdf/2205.14120.pdf)]:
```bibtex
@article{radenovic2022neural,
  title={Neural Basis Models for Interpretability},
  author={Radenovic, Filip and Dubey, Abhimanyu and Mahajan, Dhruv},
  journal={arXiv:2205.14120},
  year={2022}
}
```

Scalable Polynomial Additive Model (SPAM) [[pdf](https://arxiv.org/pdf/2205.14108.pdf)]:
```bibtex
@article{dubey2022scalable,
  title={Scalable Interpretability via Polynomials},
  author={Dubey, Abhimanyu and Radenovic, Filip and Mahajan, Dhruv},
  journal={arXiv:2205.14108},
  year={2022}
}
```

### License
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
