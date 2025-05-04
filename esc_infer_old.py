# coding=utf-8

import argparse
import json
import logging
import os
# import nltk
# nltk.download('punkt')
import numpy as np
import torch
import tqdm
from torch import Tensor

from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer
from esc_dataset_old import BaseDataset
from torch.utils.data import DataLoader
from model import Model
from eval_utils import eval_model_loss

from inputters.inputter_utils import _norm
from metric.myMetrics import Metric
from utils.building_utils import boolean_string


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def cut_seq_to_eos(sentence, eos, remove_id=None):
    if remove_id is None:
        remove_id = [-1]
    sent = []
    for s in sentence:
        if s in remove_id:
            continue
        if s != eos:
            sent.append(s)
        else:
            break
    return sent


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_path', type=str, default= "/model/zhuoer/BlenderBot_small")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--load_checkpoint", '-c', type=str, default= "/data/zhuoer/ESC/simple_graph/my_result_old/2024-03-11084349.3e-05.8.1gpu/step-4400.bin")
parser.add_argument("--infer_batch_size", type=int, default=2)

parser.add_argument('--window', type=int, default=8)
parser.add_argument('--in_channels', type=int, default=512)
parser.add_argument('--num_relations', type=int, default=4)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=3)

parser.add_argument('--only_encode', action='store_true', help='only do encoding')
parser.add_argument('--only_generate', action='store_true', help='do not conduct evaluations')
parser.add_argument('--chinese', action='store_true', help='chinese language')
parser.add_argument('--add_nlg_eval', action='store_false', help='add nlg-eval')

parser.add_argument("--min_length", type=int, default=10)
parser.add_argument("--max_length", type=int, default=40)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_k", type=int, default=30)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument("--length_penalty", type=float, default=1.0)
parser.add_argument("--repetition_penalty", type=float, default=1.03)
parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
args.device, args.n_gpu = device, n_gpu

logger.info('initializing cuda...')
_ = torch.tensor([1.], device=args.device)

set_seed(args.seed)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

toker = AutoTokenizer.from_pretrained(args.pretrained_model_path)
toker.add_special_tokens({'cls_token': '[CLS]'})

model = Model.from_pretrained(args.pretrained_model_path, \
              in_channels=args.in_channels,out_channels=args.in_channels//args.heads, \
              num_relations=args.num_relations, heads=args.heads, num_layers=args.num_layers)
model.tie_tokenizer(toker)
logger.info('loading finetuned model from %s' % args.load_checkpoint)
model.load_state_dict(torch.load(args.load_checkpoint, map_location=torch.device('cpu')))

logger.info('deploying model...')
model.to(args.device)

model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

model.eval()

pad = toker.pad_token_id
if pad is None:
    pad = toker.eos_token_id
    assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
bos = toker.bos_token_id
if bos is None:
    bos = toker.cls_token_id
    assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
eos = toker.eos_token_id
if eos is None:
    eos = toker.sep_token_id
    assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
generation_kwargs = {
    'max_length': args.max_length,
    'min_length': args.min_length,
    'do_sample': True if (args.top_k > 0 or args.top_p < 1) else False,
    'temperature': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'num_beams': args.num_beams,
    'num_return_sequences': args.num_return_sequences,
    'length_penalty': args.length_penalty,
    'repetition_penalty': args.repetition_penalty,
    'no_repeat_ngram_size': args.no_repeat_ngram_size,
    'encoder_no_repeat_ngram_size': args.no_repeat_ngram_size if model.config.is_encoder_decoder else None,
    'pad_token_id': pad,
    'bos_token_id': bos,
    'eos_token_id': eos,
}
print(json.dumps(generation_kwargs, indent=2, ensure_ascii=False))

testset = BaseDataset(dataset_type='test',window=args.window, toker=toker)
infer_loader = DataLoader(testset, batch_size=args.infer_batch_size, shuffle=False, num_workers=0, collate_fn=testset.collate_fn_batch_infer)

metric_res = {}
if not args.only_encode and not args.only_generate:
    test_loader = DataLoader(testset, batch_size=args.infer_batch_size, shuffle=False, num_workers=0, collate_fn=testset.collate_fn_batch)
    infer_loss, infer_ppl, _, pointwise_loss, pointwise_sample,stra_acc = eval_model_loss(
        model=model,
        eval_dataloader=test_loader,
        epoch_id=0,
        infer=True,
        args=args,
    )
    assert len(pointwise_loss) == len(pointwise_sample)
    metric_res['perplexity'] = float(np.exp(infer_loss))
    metric_res['stra_acc'] = stra_acc
    
    ptr = 0
    
if not args.only_generate:
    metric = Metric(toker)
    
res = []
decode = lambda x: _norm(toker.decode(x))
for batch,responses in tqdm.tqdm(infer_loader, total=len(infer_loader),desc="Inferring"):
    batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
    batch.update(generation_kwargs)
    generations = model.generate(**batch)

    if not args.only_encode:
        generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]   # response id ****

    posts = []
    references = []
    for idx,cls in enumerate(batch['cls_indices']):
        posts.append(toker.decode(batch['input_ids'][idx][cls[-2]+1:cls[-1]].tolist()))
        references.append(toker.decode(responses[idx][:-1]))    
    
    for idx in range(len(batch['cls_indices'])):
        p = posts[idx]    # 不参与计算，直接输出
        r = references[idx]    # 真实
        if not args.only_encode:   # True
            if args.num_return_sequences > 1:    # 输出多句  [r1, r2, ...]
                g = []
                for gg in generations[idx * args.num_return_sequences: (idx+1) * args.num_return_sequences]:
                    g.append(gg)
            else:
                g = generations[idx]    # r
            
            if not args.only_generate and args.num_return_sequences == 1:
                ref, gen = [r], toker.decode(g) if not isinstance(g[0], list) else toker.decode(g[0])   # 人类的句子
                metric.forword(ref, gen, chinese=args.chinese)    # 生成各类指标
            
            if isinstance(g[0], list):   
                g  = [decode(gg) for gg in g]
            else:
                g = decode(g)
            
            tmp_res_to_append = {'sample_id': 0, 'post': p, 'response': r, 'generation': g}    # 记录
            #print('> context:   ', p)
            #print('> generation:', g)
        else:    # 不用管
            tmp_res_to_append = {'sample_id': 0, 'post': p, 'response': r}
        #print(json.dumps(tmp_res_to_append, indent=4, ensure_ascii=False))
        
        if not args.only_encode and not args.only_generate:
            ptr_loss = pointwise_loss[ptr]
            ptr_sample = pointwise_sample[ptr]
            turn_loss = ptr_loss / ptr_sample
            turn_ppl = np.exp(turn_loss)
            tmp_res_to_append['token_num'] = ptr_sample
            tmp_res_to_append['loss'] = turn_loss
            tmp_res_to_append['ppl'] = turn_ppl
            ptr += 1
            
        res.append(tmp_res_to_append)
    
    #raise EOFError
    
if not args.only_encode and not args.only_generate:
    assert ptr == len(pointwise_loss)

checkpoint_dir_path = '/'.join(args.load_checkpoint.split('/')[:-1])
checkpoint_name = args.load_checkpoint.split('/')[-1]
if not args.only_encode:
    save_dir = f'{checkpoint_dir_path}/res_{checkpoint_name}_k.{args.top_k}' \
                f'_p.{args.top_p}_b.{args.num_beams}_t.{args.temperature}_lp.{args.length_penalty}' \
                f'_rp.{args.repetition_penalty}_ng.{args.no_repeat_ngram_size}'
else:
    save_dir = f'{checkpoint_dir_path}/res_{checkpoint_name}'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

with open(os.path.join(save_dir, f'gen.json'), 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=2, sort_keys=False)

with open(os.path.join(save_dir, f'gen.txt'), 'w') as f:
    for line in res:
        f.write(json.dumps(line, ensure_ascii=False) + '\n')

metric_res_list = None
if not args.only_encode and not args.only_generate:
    metric_res_list = {}
    closed_res = metric.close()
    metric_res.update(closed_res[0])
    metric_res_list.update(closed_res[1])

if not args.only_generate:
    with open(os.path.join(save_dir, f'metric.json'), 'w') as f:
        json.dump(metric_res, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    if metric_res_list is not None:
        with open(os.path.join(save_dir, f'metric_list.json'), 'w') as f:
            json.dump(metric_res_list, f)





