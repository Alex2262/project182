"""Load the repository's inherited YAML configs without obsolete Quinine pins."""
import argparse
from copy import deepcopy
from pathlib import Path
import yaml
from munch import Munch


def merge(base,override):
    result=deepcopy(base)
    for key,value in override.items():
        result[key]=merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else deepcopy(value)
    return result


def read_yaml(path,seen=()):
    path=Path(path).resolve()
    if path in seen:raise ValueError('Cyclic YAML inheritance')
    value=yaml.safe_load(path.read_text()) or {}
    parents=value.pop('inherit',[])
    if isinstance(parents,str):parents=[parents]
    result={}
    for parent in parents:
        result=merge(result,read_yaml(path.parent/parent,(*seen,path)))
    return merge(result,value)


def load_config(path):
    defaults={'test_run':False,'model':{'pilot':False,'state_size':32},
              'training':{'num_tasks':None,'num_training_examples':None,'task_kwargs':{},
                          'batch_size':64,'learning_rate':.0003,'train_steps':1000,
                          'save_every_steps':1000,'keep_every_steps':-1,'resume_id':None},
              'wandb':{'project':'project182','entity':'alexander-tian-ucb','notes':'',
                       'name':None,'log_every_steps':100}}
    conf=merge(defaults,read_yaml(path))
    for section in ('model','training','wandb'):
        if not isinstance(conf[section],dict):raise ValueError(f'{section} must be a mapping')
    for key in ('n_dims','n_positions','n_embd','n_layer','n_head','family'):
        if key not in conf['model']:raise ValueError(f'Missing model.{key}')
    if conf['training']['learning_rate']<=0:raise ValueError('Learning rate must be positive')
    return Munch.fromDict(conf)


def parse_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--test-run',action='store_true')
    args=parser.parse_args()
    conf=load_config(args.config)
    if args.test_run:conf.test_run=True
    return conf
