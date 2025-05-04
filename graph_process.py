import pickle
import numpy as np
import argparse
import json
import os
import sys
sys.path.append(os.getcwd())
import tqdm


def get_com_graph_multi_speaker(spk_list, max_hip=1):
    edge_index = []
    edge_types = []
    for conv in tqdm.tqdm(spk_list, total=len(spk_list),desc="Commense Edge Construction"):
        edge_idx = []
        edge_type = []
        num_spk = np.unique(conv)
        for i in range(len(conv)):
            xwant_hip = 0
            owant_hip = [0 for spk in range(np.max(num_spk) + 1)]
            xintent_hip = 0
            s_i = conv[i]
            j = i - 1
            pre_edge_idx = []
            pre_edge_type = []
            while j >= 0:
                if i==len(conv)-1:
                    break
                if conv[j] == s_i and xintent_hip < max_hip:
                    pre_edge_idx.append([i, j])
                    pre_edge_type.append('xIntent')
                    xintent_hip += 1
                if xintent_hip == max_hip:
                    break
                j -= 1
            for k in range(len(pre_edge_idx) - 1, -1, -1):
                edge_idx.append(pre_edge_idx[k])
                edge_type.append(pre_edge_type[k])
            # if i<(len(conv)-1):
            #     edge_idx.append([i, i])
            #     edge_type.append('xEffect')
            j = i + 1
            while j < len(conv):
                if conv[j] == s_i and xwant_hip < max_hip:
                    edge_idx.append([i, j])
                    edge_type.append('xWant')
                    xwant_hip += 1
                if conv[j] != s_i:
                    if owant_hip[conv[j]] < max_hip:
                        edge_idx.append([i, j])
                        edge_type.append('oWant')
                        owant_hip[conv[j]] += 1
                j += 1
        edge_index.append(np.transpose(np.array(edge_idx)))
        edge_types.append(edge_type)
    graph = {'edge_index': edge_index, 'edge_type': edge_types}
    return graph
# edge_index : 训练集中所有对话， 每段对话的连接.transpose
# edge_types ： 对应的 relation type

    
parser = argparse.ArgumentParser()
    
parser.add_argument('-train_path', type=str, default='/data/zhuoer/ESC/simple_graph/original/train_511_40_8.pkl')
parser.add_argument('-valid_path', type=str, default='/data/zhuoer/ESC/simple_graph/original/valid_511_40_8.pkl')
parser.add_argument('-test_path', type=str, default='/data/zhuoer/ESC/simple_graph/original/test_511_40_8.pkl')
parser.add_argument('-hip', type=int, default=2)

args = parser.parse_args()

def get_role_list(data_path):
    train_data = pickle.load(open(data_path, 'rb'), encoding='utf-8')
    role = []
    for data_i in train_data:
        role.append(data_i['role_id'])
    return role

# get_role_list(args.valid_path)

train_com_graph = get_com_graph_multi_speaker(get_role_list(args.train_path), args.hip)
valid_com_graph = get_com_graph_multi_speaker(get_role_list(args.valid_path), args.hip)
test_com_graph = get_com_graph_multi_speaker(get_role_list(args.test_path), args.hip)

pickle.dump(train_com_graph, open('/data/zhuoer/ESC/simple_graph/original/train_graph_511_40_8_hip'+ str(args.hip) +'-xEffect.pkl', 'wb'))
pickle.dump(valid_com_graph, open('/data/zhuoer/ESC/simple_graph/original/valid_graph_511_40_8_hip'+ str(args.hip) +'-xEffect.pkl', 'wb'))
pickle.dump(test_com_graph, open('/data/zhuoer/ESC/simple_graph/original/test_graph_511_40_8_hip'+ str(args.hip) +'-xEffect.pkl', 'wb'))

# data = pickle.load(open('/data/zhuoer/ESC/graph/com_graph_hip'+ str(args.hip) +'.pkl', 'rb'), encoding='utf-8')


