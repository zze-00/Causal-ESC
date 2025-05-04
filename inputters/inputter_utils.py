# coding=utf-8
import gzip
import json
import os
import math
import random
import pickle
from functools import partial
from torch.utils.data import DataLoader, Sampler


def _norm(s): #去除多余的空格字符，并将单词之间的多个空格合并成一个
    return ' '.join(s.strip().split())


class BucketSampler(Sampler):
    """
    this sampler will sort data by sequence length
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size], # ids分桶
                          key=lambda i: self._lens[i], reverse=True) # 子列表的元素将根据其与 self._lens 中的值的大小降序排列。
                   for i in range(0, len(ids), self._bucket_size)] # 生成的数列将从0开始，每隔self._bucket_size个元素增加一个值，直到不超过len(ids)为止
        batches = [bucket[i:i+self._batch_size] #分桶后分批（batches）
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast: # 确保只有满足self._batch_size的批次被保留，而不满足条件的批次将被丢弃。
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size] #它遍历之前生成的所有批次 batches，并只保留那些长度等于 self._batch_size 的批次。
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size) # 桶的数量
                        + [len(self._lens) % self._bucket_size]) # 将完整桶的大小列表与最后一个桶的大小合并成一个列表
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes) # 数据集的总批次
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)


class BucketingDataLoader(object):
    def __init__(self, toker, feature_dataset, batch_size,
                 bucket=100, shuffle=True, **kwargs):
        assert 'inputter_name' in kwargs
        assert 'config_name' in kwargs
        inputter_name = kwargs.pop('inputter_name')
        config_name = kwargs.pop('config_name')
        with open(f'./DATA/{inputter_name}.{config_name}/data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.toker = toker
        self.feature_dataset = feature_dataset
        self.batch_size = batch_size
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle

    def __iter__(self):
        trunc_chunk = []
        lens = []
        for feat in self.data:
            trunc_chunk.append(feat)
            lens.append(feat.input_len)

        dataset = self.feature_dataset(trunc_chunk)
        sampler = BucketSampler(lens, self.bucket_size, self.batch_size,
                                droplast=True, shuffle=self.shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=0,  # can test multi-worker
                            collate_fn=partial(self.feature_dataset.collate, toker=self.toker))
        yield from loader

    def __len__(self):
        return len(self.data)


class DistributedBucketingDataLoader(BucketingDataLoader):
    """ distributed version """
    def __init__(self, rank, num_replica, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.num_replica = num_replica
        self.data = self.data[self.rank::self.num_replica]

