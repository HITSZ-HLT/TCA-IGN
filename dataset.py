import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import random
from pandas import DataFrame


class IEMOCAPDataset(Dataset):
    tk = 0
    def __init__(self, dataset_name = 'IEMOCAP', split = 'train', speaker_vocab=None, label_vocab=None, args = None, tokenizer = None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('data/%s/%s_data_roberta.json.feature'%(dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)

        topic_n = self.args.topic

        # for MELD #################################################################
        # _, _, _, \
        # roberta1, roberta2, roberta3, roberta4, \
        # _, trainIds, testIds, validIds \
        # = pickle.load(open('data/MELD/meld_features_roberta.pkl', 'rb'), encoding='latin1')
        #################################################################

        Topic = np.load('data/{}_{}.npy'.format(dataset_name, topic_n))

        if split == 'train':
            xIntent, xAttr, xNeed, xWant, xEffect, xReact, oWant, oEffect, oReact \
                = pickle.load(open('data/{}/{}_features_comet1.pkl'.format(dataset_name, dataset_name), 'rb'), encoding='latin1')
            # keys = [x for x in trainIds]
        if split == 'dev':
            xIntent, xAttr, xNeed, xWant, xEffect, xReact, oWant, oEffect, oReact \
                = pickle.load(open('data/{}/{}_features_comet2.pkl'.format(dataset_name, dataset_name), 'rb'), encoding='latin1')
            # keys = [x for x in validIds]
        if split == 'test':
            xIntent, xAttr, xNeed, xWant, xEffect, xReact, oWant, oEffect, oReact \
                = pickle.load(open('data/{}/{}_features_comet3.pkl'.format(dataset_name, dataset_name), 'rb'), encoding='latin1')
            # keys = [x for x in testIds]


        # one-dim (to) two-dim topics #####################################################
        newTopic = []
        for i in range(len(raw_data)):
            newTopic_tmp = []
            for j in range(len(raw_data[i])):
                newTopic_tmp.append(Topic[IEMOCAPDataset.tk])
                IEMOCAPDataset.tk += 1
            newTopic.append(newTopic_tmp)
        ##################################################################################

        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for item, d in enumerate(raw_data):
            utterances = []
            labels = []
            speakers = []
            features = []
            for i,u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers':speakers,
                'features': features,
                'xWant': xWant[item],
                'xEffect': xEffect[item],
                'xReact': xReact[item],
                'oWant': oWant[item],
                'oEffect': oEffect[item],
                'oReact': oReact[item],
                'xIntent': xIntent[item],
                'xAttr': xAttr[item],
                'xNeed': xNeed[item],
                'topic': newTopic[item]
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances'], \
               torch.FloatTensor(self.data[index]['xWant']), \
               torch.FloatTensor(self.data[index]['xEffect']), \
               torch.FloatTensor(self.data[index]['xReact']), \
               torch.FloatTensor(self.data[index]['oWant']), \
               torch.FloatTensor(self.data[index]['oEffect']), \
               torch.FloatTensor(self.data[index]['oReact']), \
               torch.FloatTensor(self.data[index]['xIntent']), \
               torch.FloatTensor(self.data[index]['xAttr']), \
               torch.FloatTensor(self.data[index]['xNeed']), \
               torch.LongTensor(self.data[index]['topic'])

    def __len__(self):
        return self.len

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                get_local_pred = False
                get_global_pred = False
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s and not get_local_pred:
                        get_local_pred = True
                        a[i,j] = 1
                    elif speaker[j] != s and not get_global_pred:
                        get_global_pred = True
                        a[i,j] = 1
                    if get_global_pred and get_local_pred:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        #print(1111, self.args.windowp)
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):
                    #if cnt < 1:          
                    a[i,j] = 1
                    # if random.random() < 0.01:
                    #         a[i,j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt==self.args.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)


    def get_adj_v1_aadj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        aadj = []
        #print(1111, self.args.windowp)
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):
                    #if cnt < 1:          
                    a[i,j] = 1
                    # if random.random() < 0.01:
                    #         a[i,j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt==self.args.windowp_aadj:
                            break
            aadj.append(a)
        return torch.stack(aadj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i,j] = 1
                        # if random.random() < 0.1: # 69.03
                        #     s[i,j] = 2
                        s_onehot[i,j,1] = 1
                    else:
                        s_onehot[i,j,0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first = True) # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first = True, padding_value = -1) # (B, N )
        adj = self.get_adj_v1([d[2] for d in data], max_dialog_len)
        aadj = self.get_adj_v1_aadj([d[2] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first = True, padding_value = -1)
        utterances = [d[4] for d in data]
        xWant, xEffect, xReact = pad_sequence([d[5] for d in data], batch_first = True), pad_sequence([d[6] for d in data], batch_first = True), pad_sequence([d[7] for d in data], batch_first = True)
        oWant, oEffect, oReact = pad_sequence([d[8] for d in data], batch_first = True), pad_sequence([d[9] for d in data], batch_first = True), pad_sequence([d[10] for d in data], batch_first = True)
        xIntent, xAttr, xNeed = pad_sequence([d[11] for d in data], batch_first = True), pad_sequence([d[12] for d in data], batch_first = True), pad_sequence([d[13] for d in data], batch_first = True)

        topics = pad_sequence([torch.LongTensor(d[14]) for d in data], batch_first = True, padding_value = -1)
        adj2 = self.get_adj_v1([d[14] for d in data], max_dialog_len)
        s_mask2, s_mask_onehot2 = self.get_s_mask([d[14] for d in data], max_dialog_len)
    
        return feaures, labels, adj, aadj, s_mask, s_mask_onehot,lengths, speakers, utterances, xWant, xEffect, xReact, oWant, oEffect, oReact, xIntent, xAttr, xNeed, adj2, s_mask2, topics
