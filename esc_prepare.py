# coding=utf-8

import argparse
import json
import pickle
import multiprocessing as mp
import tqdm
from inputters import inputters
from building_utils import build_model
import json
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

parser = argparse.ArgumentParser()

parser.add_argument('--train_input_file', type=str, default= '/data/zhuoer/ESC/valid.txt')
parser.add_argument('--config_name', type=str, default= 'strat')
parser.add_argument('--Emoberta', type=str, default= '/model/zhuoer/Emoberta')
parser.add_argument('--max_input_length', type=int, default=511, help='discard data longer than this')
parser.add_argument('--max_decoder_input_length', type=int, default=40, help='discard data longer than this')
parser.add_argument('--single_processing', action='store_false', help='do not use multiprocessing')
parser.add_argument('--window', type=int, default=8, help='graph window')


args = parser.parse_args()

toker = build_model(only_toker=True, config_name = args.config_name) 

def _norm(s): #去除多余的空格字符，并将单词之间的多个空格合并成一个
    return ' '.join(s.strip().split())

emo_mapping_dict = {'neutral': 0, 'joy':1,'surprise' : 2,'anger': 3,'sadness' :4 ,'disgust' : 5,'fear': 6}
strat_mapping_dict = {"Question": 0, "Restatement or Paraphrasing": 1, "Reflection of feelings":2,\
                      "Self-disclosure": 3, "Affirmation and Reassurance": 4, "Providing Suggestions" : 5,\
                      "Information" : 6, "Others": 7}

def convert_data_to_inputs(data, toker, emo_pretained_model): # convert_data_to_ids, 打包成batch
    emo_toker = AutoTokenizer.from_pretrained(emo_pretained_model)    
    model = AutoModelForSequenceClassification.from_pretrained(emo_pretained_model)
    emo_pip = pipeline (task = "sentiment-analysis", model = model, tokenizer = emo_toker)
    
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x)) # 转成ids

    dialog = data['dialog']
    inputs = []
    context = []
    stra_id_his = []
    emotion_id_his = []
    role_id = []
    dia_turn = []
    
    for i in range(len(dialog)):
        text = _norm(dialog[i]['text']) # 当前utter
        text = process(text) # 当前utter_to_ids
        dia_turn.append(i)

        if dialog[i]['speaker'] == 'usr':
            emotion = emo_pip(dialog[i]['text'])[0]['label']
            assert emotion in ["neutral","joy","surprise","anger","sadness","disgust","fear"], 'emotion run out of range!'
            emo_label = emo_mapping_dict[emotion]
            emotion_id_his.append(emo_label)
            stra_id_his.append(None)
            role_id.append(1)

        if dialog[i]['speaker'] == 'sys':
            strat_id = strat_mapping_dict[dialog[i]['strategy']]
            stra_id_his.append(strat_id)
            emotion_id_his.append(None)
            role_id.append(0)
        
        if i > 0 and dialog[i]['speaker'] == 'sys': #不是第一句话
            res = {
                'context': context.copy(),
                'response': text,
                'strat_id': strat_id,
                'stra_id_his': stra_id_his.copy(),
                'emotion_id_his':emotion_id_his.copy(),
                'role_id':role_id.copy(),
                'dia_turn':dia_turn.copy()
            }
            
            inputs.append(res) # 遇到一个response 打包一次

        # if dialog[i]['speaker'] == 'sys':
        #     text = [strat_id] + text

        context = context + [text] # 整个对话的context   
        
        assert len(emotion_id_his)==len(stra_id_his)==len(context)==len(role_id)==len(dia_turn)

    return inputs


def featurize( # 处理成输入transformer格式
    bos, eos, cls,
    context,
    response,strat_id,
    stra_id_his,emotion_id_his,role_id,dia_turn,
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
    dia_turn = dia_turn[-len(cls_indices):]

    res_dict = {
        'input_ids' : input_ids,
        'cls_indices' : cls_indices,
        'decoder_input_ids' : decoder_input_ids,
        'labels' : labels,
        'strat_id' : strat_id,
        'stra_id_his': stra_id_his,
        'emotion_id_his' : emotion_id_his,
        'role_id' : role_id,
        'dia_turn' : dia_turn
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
    
    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        # feat = {}
        feat = featurize(bos, eos, cls, \
                         ipt['context'], ipt['response'], ipt['strat_id'], \
                         ipt['stra_id_his'], ipt['emotion_id_his'],ipt['role_id'],ipt['dia_turn'],\
                         args)
        features.append(feat)
    return features 
    
def process_data(line):
    data = json.loads(line)
    inputs = convert_data_to_inputs(
        data=data,
        toker=toker,
        emo_pretained_model=args.Emoberta
    ) 
    features = convert_inputs_to_features(
        inputs=inputs,
        toker=toker,
        args=args
    ) #处理成 transformer 结构的输入

    return features

with open(args.train_input_file) as f:
    reader = f.readlines()

processed_data = []
if args.single_processing:
    for dia_id,features in enumerate(map(process_data, tqdm.tqdm(reader, total=len(reader)))):
        for f in features:
            assert dia_id < 910, 'dia_id out of range!'
            f['dia_id'] = dia_id
        processed_data.extend(features) # 所有的dialogue中的批放到 processed_data 中
else:
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for dia_id,features in enumerate(pool.imap(process_data, tqdm.tqdm(reader, total=len(reader)))):
            for f in features:
                assert dia_id < 910, 'dia_id out of range!'
                f['dia_id'] = dia_id
            processed_data.extend(features)

print('saving...')
data_path = '/data/zhuoer/ESC/simple_graph/original/valid_'+ str(args.max_input_length) \
            +'_'+ str(args.max_decoder_input_length) + '_' + str(args.window) + '.pkl'
print(data_path)

with open(data_path, 'wb') as file:
    pickle.dump(processed_data, file)

# train_data = pickle.load(open(data_path, 'rb'), encoding='utf-8')
print("hello")