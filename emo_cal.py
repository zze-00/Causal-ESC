import pickle

data_path = '/data/zhuoer/ESC/simple_graph/original/train_511_40_8.pkl'
data = pickle.load(open(data_path, 'rb'), encoding='utf-8')

def count_element_in_list(e, lst):  
    return lst.count(e)

result = []
for d in data:
    result.extend(d['emotion_id_his'])

print(f"情绪 neutral : {count_element_in_list(0,result)}")
print(f"情绪 joy : {count_element_in_list(1,result)}")
print(f"情绪 surprise : {count_element_in_list(2,result)}")
print(f"情绪 anger : {count_element_in_list(3,result)}")
print(f"情绪 sadness : {count_element_in_list(4,result)}")
print(f"情绪 disgust : {count_element_in_list(5,result)}")
print(f"情绪 fear : {count_element_in_list(6,result)}")

