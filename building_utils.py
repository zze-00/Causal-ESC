# coding=utf-8

import json
import os
import logging
import torch
from os.path import join

from models import models
from transformers import (AutoTokenizer, AutoModel, AutoConfig)
from torch.distributed import get_rank

logger = logging.getLogger(__name__)


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def build_model(only_toker=False, checkpoint=None, config_name = 'strat' ):
    
    if not os.path.exists(f'./CONFIG/{config_name}.json'):
        raise ValueError
    
    with open(f'./CONFIG/{config_name}.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if 'model_name' not in config or 'pretrained_model_path' not in config:
        raise ValueError
    
    toker = AutoTokenizer.from_pretrained(config['pretrained_model_path'])
    if only_toker:
        toker.add_special_tokens({'cls_token': '[CLS]'})
        # if 'expanded_vocab' in config:
        #     toker.add_tokens(config['expanded_vocab'], special_tokens=True)
        
        return toker  # 加strategy token 到toker vocab里 ：54944 + 8
    
    Model = models[config['model_name']]
    model = Model.from_pretrained(config['pretrained_model_path'])
    if config.get('custom_config_path', None) is not None:
        model = Model(AutoConfig.from_pretrained(config['custom_config_path']))
    
    if 'gradient_checkpointing' in config:
        setattr(model.config, 'gradient_checkpointing', config['gradient_checkpointing'])
    
    if 'expanded_vocab' in config:
        toker.add_tokens(config['expanded_vocab'], special_tokens=True)
    model.tie_tokenizer(toker)
    
    if checkpoint is not None:
        if local_rank == -1 or get_rank() == 0:
            logger.info('loading finetuned model from %s' % checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    
    return toker, model

