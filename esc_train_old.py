# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import datetime
import json
import logging
import os
import sys
import time
from os.path import join

import numpy as np
import torch
import tqdm
from torch import Tensor
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed

from utils.building_utils import boolean_string
from esc_dataset_old import BaseDataset
from torch.utils.data import DataLoader
from model import Model
from eval_utils import eval_model_loss


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000
CUDA_VISIBLE_DEVICES=0
#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_path', type=str, default= "/model/zhuoer/BlenderBot_small")
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=5, help="how many training epochs")
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

parser.add_argument('--pretrained_model', type=str, default= 'blenderbot_small90M')
parser.add_argument('--window', type=int, default=8)
parser.add_argument('--in_channels', type=int, default=512)
parser.add_argument('--num_relations', type=int, default=4)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=3)

parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--valid_steps", type=int, default=200,help="how many optim steps between validations")
parser.add_argument('--max_grad_norm', type=float, default=1.0)

args = parser.parse_args()

init_args_dict = vars(args).copy()

logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
args.device, args.n_gpu = device, n_gpu

logger.info('initializing cuda...')
torch.tensor([1.], device=args.device)

set_seed(args.seed)


logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))


#########################################################################
# Prepare Data Set
##########################################################################

toker = AutoTokenizer.from_pretrained(args.pretrained_model_path)
toker.add_special_tokens({'cls_token': '[CLS]'}) 

trainset = BaseDataset(dataset_type='train', window=args.window, toker=toker)
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=trainset.collate_fn_batch)

valset = BaseDataset(dataset_type='test',window=args.window, toker=toker)
val_loader = DataLoader(valset, batch_size=args.batch_size*2, shuffle=False, num_workers=0, collate_fn=valset.collate_fn_batch)

# for batch in train_loader:
#     batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}   

if args.num_epochs is not None:
    args.num_optim_steps = args.num_epochs * (len(trainset) // args.batch_size +
                                              int(len(trainset) % args.batch_size != 0)) # 1 or 0

#########################################################################
# Prepare Model and Optimizer
#########################################################################
# config = BlenderbotSmallConfig.from_pretrained(args.pretrained_model_path)
# model = Model(config=config, encoder_type=args.pretrained_model,\
#               in_channels=args.in_channels,out_channels=args.in_channels//args.heads, \
#               num_relations=args.num_relations, heads=args.heads, num_layers=args.num_layers)
model = Model.from_pretrained(args.pretrained_model_path, \
              in_channels=args.in_channels,out_channels=args.in_channels//args.heads, \
              num_relations=args.num_relations, heads=args.heads, num_layers=args.num_layers)

model.tie_tokenizer(toker)
logger.info('deploying model...')
model.to(args.device)


model_parameters = list(filter(lambda p: p.requires_grad, model.parameters())) #所有的参数
# 过滤出那些requires_grad属性为True的参数，
total_params = sum([np.prod(p.size()) for p in model_parameters])
# 模型的 requires_grad属性为True 总参数量。

logger.info('Number of parameter = {}'.format(total_params))

param_optimizer = list(model.named_parameters()) #所有的参数及名称
no_decay = ['bias', 'ln', 'LayerNorm.weight']   # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
#列表，包含了两个字典，每个字典定义了一组模型参数和其对应的权重衰减值。

optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_optim_steps
)

#########################################################################
# Training !
##########################################################################

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
output_dir = join(f'/data/zhuoer/ESC/simple_graph/my_result_old/',
                  f'{timestamp}.{args.learning_rate}.{args.batch_size}.{n_gpu}gpu')

os.makedirs(output_dir, exist_ok=True)
print(output_dir)
with open(join(output_dir, 'args.json'), 'w', encoding='utf-8') as f:
    json.dump(init_args_dict, f, ensure_ascii=False, indent=2)
# args.json，其中包含了init_args_dict的内容

train_logger = open(join(output_dir, 'train_log.csv'), 'a+', buffering=1)
eval_logger = open(join(output_dir, 'eval_log.csv'), 'a+', buffering=1)
print('epoch,global_step,step,tmp_loss,tmp_ppl,mean_loss,mean_ppl,epoch_time', file=train_logger)
print('epoch,global_step,step,freq_loss,freq_ppl,stra_acc', file=eval_logger)

global_step = 0
step = 0
epoch = 0

if args.pbar:
    pbar = tqdm.tqdm(total=args.num_optim_steps, desc=f"training")
    # 创建一个进度条对象。
    # total参数设置了进度条的总步数，通常对应于训练的总步骤数量。
    # desc参数用于设置进度条的描述文本，通常是"training"。
 
while True:
    model.train()
    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    train_start_time_epoch = time.time()
    for i,batch in enumerate(train_loader):
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        # batch.update({'global_step': global_step})
        # batch.update({'epoch': epoch})
        # batch.update({'warmup_steps': args.warmup_steps})
        outputs = model(**batch)
        
        loss = outputs.pop('all') # 每个词的loss
        ppl = outputs.pop('ppl')
        
        if 'input_ids' in batch:
            input_ids = batch['input_ids']
        elif 'tgt_input_ids' in batch:
            input_ids = batch['tgt_input_ids']
        else:
            assert 'src_input_ids' in batch
            input_ids = batch['src_input_ids']
        
        loss = loss / (args.batch_size / input_ids.shape[0])
        
        loss.backward()
        
        tmp_loss = float(loss.item()) * (args.batch_size / input_ids.shape[0])
        tr_loss += tmp_loss
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        mean_loss = tr_loss / nb_tr_steps # 每个词的loss / 步
        
        if ppl.item() < INF:
            tmp_ppl = ppl.item()
        else:
            tmp_ppl = mean_ppl
        tr_ppl += tmp_ppl
        mean_ppl = tr_ppl / nb_tr_steps

        # gradient update
        step += 1
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        epoch_time = time.time() - train_start_time_epoch
        if pbar is not None:
            pbar_str = ''
            for k, v in outputs.items():
                pbar_str += f"{k}: {v.item():.2f} "
            pbar_str += f"ppl: {mean_ppl:.2f} epoch: {epoch}"
            
            pbar.set_postfix_str(pbar_str)
            pbar.update(1)
    
        print(f'{epoch+1},{global_step},{step},{tmp_loss},{tmp_ppl},{mean_loss},{mean_ppl},{epoch_time}', file=train_logger)
        
        if (step+1) % CACHE_EMPTY_STEP == 0:
            torch.cuda.empty_cache()
    
        if epoch>0 and step % args.valid_steps == 0:
        # if step % args.valid_steps == 0:
            torch.save(model.state_dict(), join(output_dir, f'step-{step}.bin'))
            # toker.save_vocabulary(output_dir) 
            model.config.to_json_file(join(output_dir, f'config.json'))

            eval_loss, eval_ppl, _, _, _, stra_acc = eval_model_loss(
                model = model,
                eval_dataloader = val_loader,
                epoch_id = epoch,
                step_id = step,
                infer = False,
                args = args,
            )
            print(f'{epoch},{global_step},{step},{eval_loss},{eval_ppl},{stra_acc}', file=eval_logger)
            # logger.info('current learning rate: '+ str(optimizer.param_groups[0]['lr']))
            model.train()

    epoch += 1
    if args.num_epochs is not None and epoch == args.num_epochs:
        break

if pbar is not None:
    pbar.close()
train_logger.close()
eval_logger.close()
