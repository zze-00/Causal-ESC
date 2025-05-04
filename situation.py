# pip install sentence_transformers -i https://mirrors.aliyun.com/pypi/simple

import argparse
import json
import pickle
import multiprocessing as mp
import tqdm
from building_utils import build_model
import json
import tqdm
from sentence_transformers import SentenceTransformer as ST

parser = argparse.ArgumentParser()
parser.add_argument('--train_input_file', type=str, default= '/data/zhuoer/ESC/train.txt')
parser.add_argument('--test_input_file', type=str, default= '/data/zhuoer/ESC/test.txt')
parser.add_argument('--valid_input_file', type=str, default= '/data/zhuoer/ESC/valid.txt')
args = parser.parse_args()

def _norm(s): #去除多余的空格字符，并将单词之间的多个空格合并成一个
    return ' '.join(s.strip().split())

problem_dict = {'ongoing depression': 0, 
                'job crisis': 1, 
                'breakup with partner': 2, 
                'problems with friends': 3, 
                'academic pressure': 4, 
                'Sleep Problems': 5, 
                'Procrastination': 6, 
                'Alcohol Abuse': 7, 
                'Appearance Anxiety': 8, 
                'conflict with parents': 9, 
                'Issues with Children': 10, 
                'Issues with Parents': 11, 
                'School Bullying': 12}

with open(args.train_input_file) as f:
    reader = f.readlines()

def process_situation(line):
    data = json.loads(line)
    p = problem_dict[data['problem_type']]
    s = _norm(data['situation'])
    return p,s 
    
problems = []
situations = []
for p,s in map(process_situation, tqdm.tqdm(reader, total=len(reader))):
    problems.append(p)
    situations.append(s)

SBert = ST('/model/zhuoer/mpnet')
situation_embeddings = SBert.encode(situations, show_progress_bar = True, convert_to_numpy = False) #(910,768)

# pickle.dump(problems, open('/data/zhuoer/ESC/simple_graph/original/valid_problem.pkl', 'wb'))
pickle.dump(situation_embeddings, open('/data/zhuoer/ESC/simple_graph/original/train_situation.pkl', 'wb'))



