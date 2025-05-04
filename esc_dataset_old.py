import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import pickle


class BaseDataset(Dataset):
    def __init__(self, dataset_type, window, toker: AutoTokenizer):
        super(BaseDataset, self).__init__()
        assert dataset_type in ['train', 'valid', 'test'], 'ESC support train valid and test'
        self.dataset_type = dataset_type

        self.window = window

        self.toker = toker
        if dataset_type == 'train':
            data_path = '/data/zhuoer/ESC/simple_graph/old_data/train_511_40_8.pkl'
            graph_path = '/data/zhuoer/ESC/simple_graph/old_data/train_graph_511_40_8_hip2.pkl'
            edge_path = '/data/zhuoer/ESC/Comet_re_old/ESC' + '_edge_attr_' + dataset_type + '.pkl'
            # problem_path = '/data/zhuoer/ESC/simple_graph/old_data/train_problem.pkl'
            situation_path = '/data/zhuoer/ESC/simple_graph/old_data/train_situation.pkl'
        elif dataset_type == 'valid':
            data_path = '/data/zhuoer/ESC/simple_graph/old_data/valid_511_40_8.pkl'
            graph_path = '/data/zhuoer/ESC/simple_graph/old_data/valid_graph_511_40_8_hip2.pkl'
            edge_path = '/data/zhuoer/ESC/Comet_re_old/ESC' + '_edge_attr_' + dataset_type + '.pkl'
            # problem_path = '/data/zhuoer/ESC/simple_graph/old_data/valid_problem.pkl'
            situation_path = '/data/zhuoer/ESC/simple_graph/old_data/valid_situation.pkl'
        elif dataset_type == 'test':
            data_path = '/data/zhuoer/ESC/simple_graph/old_data/test_511_40_8.pkl'
            graph_path = '/data/zhuoer/ESC/simple_graph/old_data/test_graph_511_40_8_hip2.pkl'
            edge_path = '/data/zhuoer/ESC/Comet_re_old/ESC' + '_edge_attr_' + dataset_type + '.pkl'
            # problem_path = '/data/zhuoer/ESC/simple_graph/old_data/test_problem.pkl'
            situation_path = '/data/zhuoer/ESC/simple_graph/old_data/test_situation.pkl'
      
        self.data = pickle.load(open(data_path, 'rb'), encoding='utf-8')
        self.graph = pickle.load(open(graph_path, 'rb'), encoding='utf-8')
        self.edge_repre = pickle.load(open(edge_path, 'rb'), encoding='utf-8')
        # self.problem = pickle.load(open(problem_path, 'rb'), encoding='utf-8')
        self.situation = pickle.load(open(situation_path, 'rb'), encoding='utf-8')


    def __getitem__(self, item):

        input_ids = self.data[item]['input_ids']
        cls_indices = self.data[item]['cls_indices']
        decoder_input_ids = self.data[item]['decoder_input_ids']
        labels = self.data[item]['labels']
        strat_id = self.data[item]['strat_id']

        stra_id_his = self.data[item]['stra_id_his']
        stra_id_his = [-100 if x == None else x for x in stra_id_his]

        emotion_id_his = self.data[item]['emotion_id_his']
        emotion_id_his = [-100 if x == None else x for x in emotion_id_his]

        role_id = self.data[item]['role_id']
        # dia_turn = self.data[item]['dia_turn']
        # dia_id = self.data[item]['dia_id']
        edge_index = self.graph['edge_index'][item] # (2,x)

        edge_type = self.graph['edge_type'][item] # len(list) = x 
        edge_type_i = []
        edge_type_mapping = {'xEffect': 0 ,'xIntent': 1 ,'xWant': 2 ,'oWant': 3 }
        for et in edge_type:
            edge_type_i.append(edge_type_mapping[et]) 

        edge_repre = []
        for i,et in enumerate(edge_type):
            if len(self.edge_repre[item])>self.window:
                edge_repre.append(self.edge_repre[item][-self.window:][edge_index[0,i]][et])
            else:
                edge_repre.append(self.edge_repre[item][edge_index[0,i]][et])

        # p = self.problem[dia_id]
        s = self.situation[item]

        res_dict = {
            'input_ids' : input_ids,
            'cls_indices' : cls_indices,
            'decoder_input_ids' : decoder_input_ids,
            'labels' : labels,
            'strat_id' : strat_id,
            'stra_id_his': stra_id_his,
            'emotion_id_his' : emotion_id_his,
            'role_id' : role_id,
            # 'dia_turn' : dia_turn,
            # 'dia_id' : dia_id,
            'edge_index' : edge_index,
            'edge_type' : edge_type_i,
            'edge_repre' : edge_repre,
            # 'problem' : p,
            'situation' : s,
        }
        return res_dict

    def __len__(self):
        return len(self.data)
    
    def collate_fn_batch(self, features):
        pad = self.toker.pad_token_id
        
        input_ids = pad_sequence([torch.tensor(f['input_ids'], dtype=torch.long) for f in features],
                          batch_first=True, padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1.] * len(f['input_ids']), dtype=torch.float) for f in features],
                          batch_first=True, padding_value=0.)

        # count = 0
        # batch_cls_indices = []
        # for f in features:
        #     cls_indices = [x + y for x, y in zip(f['cls_indices'], [count]*len(f['cls_indices']))]
        #     batch_cls_indices.append(cls_indices)
        #     count = count + len(f['input_ids'])
        # batch_cls_indices = sum(batch_cls_indices,[])
        cls_indices = [f['cls_indices'] for f in features]

        strat_id = [f['strat_id'] for f in features]

        stra_id_his = [f['stra_id_his'] for f in features]
        
        emotion_id_his = [f['emotion_id_his'] for f in features]

        count = 0
        batch_edge_index = []
        for f in features:
            edge_index = torch.tensor(f['edge_index'])
            acc_i = torch.full(edge_index.size(), count)
            edge_index = edge_index + acc_i
            batch_edge_index .append(edge_index)
            count = count + len(f['cls_indices'])
        batch_edge_index = torch.cat(batch_edge_index,dim=1)
        
        edge_type = torch.tensor(sum([f['edge_type'] for f in features],[]))

        edge_repre = torch.cat([torch.tensor(f['edge_repre']) for f in features],dim=0)

        
        decoder_input_ids = pad_sequence([torch.tensor(f['decoder_input_ids'], dtype=torch.long) for f in features],
                            batch_first=True, padding_value=pad)
        decoder_attention_mask = pad_sequence([torch.tensor([1.] * len(f['decoder_input_ids']), dtype=torch.float) for f in features],
                        batch_first=True, padding_value=0.)
        labels = pad_sequence([torch.tensor(f['labels'], dtype=torch.long) for f in features],
                            batch_first=True, padding_value=-100)
        
        # problems = [f['problem'] for f in features]

        situations =  torch.stack([f['situation'] for f in features], dim=0)
 

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'cls_indices' : cls_indices,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'strat_id': strat_id,
            'stra_id_his': stra_id_his,
            'emotion_id_his': emotion_id_his,
            'edge_index' : batch_edge_index,
            'edge_type' : edge_type,
            'edge_repre' : edge_repre,
            'problems' : None,
            'situations' : situations
            }
        return res
    
    def collate_fn_batch_infer(self, features):
        pad = self.toker.pad_token_id
        
        input_ids = pad_sequence([torch.tensor(f['input_ids'], dtype=torch.long) for f in features],
                          batch_first=True, padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1.] * len(f['input_ids']), dtype=torch.float) for f in features],
                          batch_first=True, padding_value=0.)

        # count = 0
        # batch_cls_indices = []
        # for f in features:
        #     cls_indices = [x + y for x, y in zip(f['cls_indices'], [count]*len(f['cls_indices']))]
        #     batch_cls_indices.append(cls_indices)
        #     count = count + len(f['input_ids'])
        # batch_cls_indices = sum(batch_cls_indices,[])
        cls_indices = [f['cls_indices'] for f in features]

        strat_id = [f['strat_id'] for f in features]

        stra_id_his = [f['stra_id_his'] for f in features]
        
        emotion_id_his = [f['emotion_id_his'] for f in features]

        count = 0
        batch_edge_index = []
        for f in features:
            edge_index = torch.tensor(f['edge_index'])
            acc_i = torch.full(edge_index.size(), count)
            edge_index = edge_index + acc_i
            batch_edge_index .append(edge_index)
            count = count + len(f['cls_indices'])
        batch_edge_index = torch.cat(batch_edge_index,dim=1)
        
        edge_type = torch.tensor(sum([f['edge_type'] for f in features],[]))

        edge_repre = torch.cat([torch.tensor(f['edge_repre']) for f in features],dim=0)

        decoder_input_ids = torch.tensor([[f['decoder_input_ids'][0]] for f in features], dtype=torch.long)
        decoder_attention_mask = None
        labels = None
        responses = [f['labels'] for f in features]
        # sample_ids = [f['dia_id'] for f in features]

        # problems = [f['problem'] for f in features]
        situations =  torch.stack([f['situation'] for f in features], dim=0)

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'cls_indices' : cls_indices,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'strat_id': strat_id,
            'stra_id_his': stra_id_his,
            'emotion_id_his': emotion_id_his,
            'edge_index' : batch_edge_index,
            'edge_type' : edge_type,
            'edge_repre' : edge_repre,
            'problems' : None,
            'situations' : situations
            }
        return res, responses # , sample_ids

    
