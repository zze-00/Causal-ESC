import argparse
import json
import pickle
import json


from sentence_transformers import SentenceTransformer as ST

parser = argparse.ArgumentParser()
parser.add_argument('--train_input_file', type=str, default='/data/zhuoer/ESC/MISC_DATA/devWithStrategy_short.tsv')
parser.add_argument('--emo_input_file', type=str, default='/data/zhuoer/ESC/MISC_DATA/test_emotion.json')
parser.add_argument('--situation_input_file', type=str, default='/data/zhuoer/ESC/MISC_DATA/devSituation.txt')

parser.add_argument('--Emoberta', type=str, default= '/model/zhuoer/Emoberta')
args = parser.parse_args()

# with open(args.train_input_file, "r", encoding="utf-8") as f:
#     df_trn = f.read().split("\n")

with open(args.train_input_file) as f:
    reader = f.readlines()

# with open(args.situation_input_file, "r", encoding="utf-8") as f:
#     sit_list = f.readlines()

# with open(args.emo_input_file, "r", encoding="utf-8") as f:
#     emotion = json.load(f)


# for idx, (reader, emotion) in enumerate(zip(reader, emotion[:-1])):
#     u = reader[idx]
#     e = emotion[idx]['emotions']

def _norm(s): #去除多余的空格字符，并将单词之间的多个空格合并成一个
    return ' '.join(s.strip().split())

def process_situation():
    with open(args.situation_input_file, "r", encoding="utf-8") as f:
        sit_list = f.readlines()
    assert len(reader)==len(sit_list)
    situations = []
    for s in sit_list:
        s = _norm(s)
        situations.append(s)
    return situations 

SBert = ST('/model/zhuoer/mpnet')
situations = process_situation()
situation_embeddings = SBert.encode(situations, show_progress_bar = True, convert_to_numpy = False) #(910,768)
pickle.dump(situation_embeddings, open('/data/zhuoer/ESC/simple_graph/old_data/valid_situation.pkl', 'wb'))