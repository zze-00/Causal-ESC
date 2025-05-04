# coding=utf-8

import argparse
import json
import re
import pickle
import multiprocessing as mp
import tqdm
from inputters import inputters
from building_utils import build_model
import json
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

parser = argparse.ArgumentParser()

parser.add_argument('--train_input_file', type=str, default='/data/zhuoer/ESC/MISC_DATA/trainWithStrategy_short.tsv')
parser.add_argument('--emo_input_file', type=str, default='/data/zhuoer/ESC/MISC_DATA/train_emotion.json')
parser.add_argument('--config_name', type=str, default= 'strat')
parser.add_argument('--Emoberta', type=str, default= '/model/zhuoer/Emoberta')
parser.add_argument('--max_input_length', type=int, default=511, help='discard data longer than this')
parser.add_argument('--max_decoder_input_length', type=int, default=40, help='discard data longer than this')
parser.add_argument('--single_processing', action='store_false', help='do not use multiprocessing')
parser.add_argument('--window', type=int, default=8, help='graph window')


args = parser.parse_args()

toker = build_model(only_toker=True, config_name = args.config_name) 

# def _norm(s): #去除多余的空格字符，并将单词之间的多个空格合并成一个
#     return ' '.join(s.strip().split())

emo_mapping_dict = {'neutral': 0, 'joy':1,'angry': 2,'sadness' :3 ,'disgust' : 4,'fear': 5}
strat_mapping_dict = {"Question": 0, "Restatement or Paraphrasing": 1, "Reflection of feelings":2,\
                      "Self-disclosure": 3, "Affirmation and Reassurance": 4, "Providing Suggestions" : 5,\
                      "Information" : 6, "Others": 7}

def _norm_text(text):
    _, r, t, *toks = text.strip().split()
    try:
        r = int(r)
        t = int(t)
        toks = ' '.join(toks[:len(toks)])
    except Exception as e:
        raise e
    return r, toks

def convert_data_to_inputs(data, toker, emotions): # convert_data_to_ids, 打包成batch
    # emo_toker = AutoTokenizer.from_pretrained(emo_pretained_model)    
    # model = AutoModelForSequenceClassification.from_pretrained(emo_pretained_model)
    # emo_pip = pipeline (task = "sentiment-analysis", model = model, tokenizer = emo_toker)
    
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x)) # 转成ids

    context = []
    stra_id_his = []
    emotion_id_his = []
    role_id = []

    ctx_res = data.split('EOS')
    _, response = _norm_text(ctx_res[-1])
    strat_id = strat_mapping_dict[response.split("[")[1].split("]")[0]]
    response = process(re.sub(r'\[.*?\]', '', response).strip())  
    for idx,ctx in enumerate(ctx_res[:-1]):
        role, ctx = _norm_text(ctx)
        role_id.append(role)
        
        if role == 1:
            context.append(process(re.sub(r'\[.*?\]', '', ctx).strip()))
            ctx_strat_id = strat_mapping_dict[ctx.split("[")[1].split("]")[0]]
            stra_id_his.append(ctx_strat_id)
            emotion_id_his.append(None)
        else:
            context.append(process(ctx))
            emotion = emotions[idx]
            assert emotion in ["neutral","joy","angry","sadness","disgust","fear"], 'emotion run out of range!'
            emo_label = emo_mapping_dict[emotion]
            emotion_id_his.append(emo_label)
            stra_id_his.append(None)

    stra_id_his.append(strat_id)
    emotion_id_his.append(None)
    role_id.append(1)
    
    res = {
            'context': context,
            'response': response,
            'strat_id': strat_id,
            'stra_id_his': stra_id_his,
            'emotion_id_his':emotion_id_his,
            'role_id':role_id,
        }
    
    assert len(emotion_id_his)==len(stra_id_his)==len(context)+1==len(role_id)==len(emotions)
        
    return res


def featurize( # 处理成输入transformer格式
    bos, eos, cls,
    context,
    response,strat_id,
    stra_id_his,emotion_id_his,role_id,
    args,
):
    ctx = []
    if len(context) >= args.window :
        for i in range(args.window - 1, 0 ,-1):
            ctx.extend(context[-i] + [cls])
    else:
        ctx = [c + [cls] for c in context]
        ctx = sum(ctx, [])
    
    input_ids = [cls] + ctx[-args.max_input_length:] # 其截断为最多max_input_length 个标记,去掉前面的。
    cls_indices = [index for index, value in enumerate(input_ids) if value == cls]
    assert len(cls_indices) <= args.window

    labels = response[:args.max_decoder_input_length]+ [eos]# 前，+1是<eos>
    decoder_input_ids = [bos] + labels[:-1] #去掉eos_id
    assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]
          

    # input_ids : <cls>____<cls>____<cls>____<cls>
    # decoder_input_ids : <bos> _________ 
    # labels :_________<eos> 

    stra_id_his = stra_id_his[-len(cls_indices):]
    emotion_id_his = emotion_id_his[-len(cls_indices):]
    role_id = role_id[-len(cls_indices):]

    res_dict = {
        'input_ids' : input_ids,
        'cls_indices' : cls_indices,
        'decoder_input_ids' : decoder_input_ids,
        'labels' : labels,
        'strat_id' : strat_id,
        'stra_id_his': stra_id_his,
        'emotion_id_his' : emotion_id_his,
        'role_id' : role_id,
    }

    return res_dict

def convert_inputs_to_features(inputs, toker, args):
    if len(inputs) == 0:
        return []
    
    # max_input_length = args.max_input_length
    # max_decoder_input_length = args.max_decoder_input_length
    
    bos = toker.bos_token_id
    eos = toker.eos_token_id
    cls = toker.cls_token_id
    
    feature = featurize(bos, eos, cls, \
                    inputs['context'], inputs['response'], inputs['strat_id'], \
                    inputs['stra_id_his'], inputs['emotion_id_his'], inputs['role_id'],\
                    args)
    return feature 
    
def process_data(line,emotions):
    # data = json.loads(line)
    inputs = convert_data_to_inputs(
        data=line,
        toker=toker,
        emotions=emotions
    ) 
    features = convert_inputs_to_features(
        inputs=inputs,
        toker=toker,
        args=args
    ) #处理成 transformer 结构的输入

    return features

with open(args.train_input_file) as f:
    reader = f.readlines()

with open(args.emo_input_file, "r", encoding="utf-8") as f:
    emotion = json.load(f)

assert len(reader)==len(emotion)

processed_data = []
if args.single_processing:
    for line, emotion in tqdm.tqdm(zip(reader, emotion),total=len(reader),desc="Pkling"):
        features = process_data(line,emotion['emotions'])
        processed_data.append(features) # 所有的dialogue中的批放到 processed_data 中
else:
    pass

print('saving...')
data_path = '/data/zhuoer/ESC/simple_graph/old_data/train_'+ str(args.max_input_length) \
            +'_'+ str(args.max_decoder_input_length) + '_' + str(args.window) + '.pkl'
print(data_path)

with open(data_path, 'wb') as file:
    pickle.dump(processed_data, file)

data_path = '/data/zhuoer/ESC/simple_graph/old_data/train_'+ str(args.max_input_length) \
            +'_'+ str(args.max_decoder_input_length) + '_' + str(args.window) + '.pkl'
train_data = pickle.load(open(data_path, 'rb'), encoding='utf-8')
print("hello")