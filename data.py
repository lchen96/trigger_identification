import re
import random
import json
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

import pytorch_lightning as pl
from nltk.tokenize import TweetTokenizer
import emoji

tweet_tokenizer = TweetTokenizer()

# load tools for preprocessning text
with open("./tools/abbreviation.json", "r", encoding="utf-8") as f:
    abbre_dict = dict(json.load(f))

class TriggerDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data
        self.cids = pd.unique(data['cid']).tolist()
        self.cid_mid_dict = {cid: data[data['cid'] == cid].sort_values(by=['cid', 'time'])['mid'].tolist() for cid in self.cids}

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, index):
        cid = self.cids[index]
        mids = self.cid_mid_dict.get(cid, [])
        data_cascade = self.data.loc[mids].to_dict(orient='records')
        return data_cascade


class TriggerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_file = args.data_file
        self.batch_size = args.batch_size
        self.val_type = args.val_type
        self.split_seed = args.split_seed
        self.test_event = args.test_event
        self.test = args.test
        self.balance = args.balance
        self.num_workers = args.num_workers
        self.args = args

    def _load_dataset(self):
        print('Loading dataset...')
        df = pd.read_csv(self.data_file)
        # reserve useful fields
        fields = ['mid', 'cid', 'pid', 'event',
                  'time', 'content', 'trigger', 'verify']
        df = df[fields]
        # convert ID fields into string type to avoid incoherence in np/pd/torch
        df['cid'] = df['cid'].apply(str)
        df['mid'] = df['mid'].apply(str)
        df['pid'] = df['pid'].apply(str)
        # reset dataframe index
        df.index = df['mid']
        df.index.name = None
        self.dataset = df

    def _split_train_val_test(self):
        '''Split all the cids into train/val/test set'''
        print('Spliting dataset...')
        df = self.dataset
        events = list(pd.unique(df['event']))
        val_type, seed, test_event = self.val_type, self.split_seed, self.test_event
        if val_type == 'RANDOM':  # randomly split with label-stratified sampling
            cids_val, cids_test, cids_train = [], [], []
            random.seed(seed)
            for i in range(3):
                for event in events:
                    df_temp = df[(df['verify'] == i) & (df['event']==event)]
                    cids_temp = pd.unique(df_temp['cid']).tolist()
                    random.shuffle(cids_temp)
                    val_num, test_num = int(len(cids_temp)*0.1), int(len(cids_temp)*0.1)
                    c1 = cids_temp[:val_num]
                    c2 = cids_temp[val_num:val_num+test_num]
                    c3 = cids_temp[val_num+test_num:]
                    cids_val.extend(c1)
                    cids_test.extend(c2)
                    cids_train.extend(c3)
        if val_type == 'LOEO':   # leave-one-event-out split
            df_new = df.copy(deep=True)
            events = ['sydneysiege', 'ottawashooting', 'ferguson', 'germanwings','charliehebdo']
            test_event = events[test_event]
            cids_train, cids_val, cids_test = [], [], []
            for i in range(3):
                df_temp = df[(df['verify'] == i) & (df['event']==test_event)]
                cids_temp = pd.unique(df_temp['cid']).tolist()
                random.seed(seed)
                random.shuffle(cids_temp)
                val_num = int(len(cids_temp)*0.5)
                c1 = cids_temp[:val_num]
                c2 = cids_temp[val_num:]
                cids_val.extend(c1)
                cids_test.extend(c2)
            train_event = [event for event in events if event!=test_event]
            if self.balance:        
                class_cid_dict = {i:list(pd.unique(df[(df['event'].isin(train_event)) & (df['verify']==i)]['cid'])) for i in range(3)}
                class_num_dict = {i:len(cids) for i, cids in class_cid_dict.items()}
                max_class_num = max(class_num_dict.values())
                cids_train = []
                for i, cid in class_cid_dict.items():
                    if max_class_num/len(cid) >= 1.8:
                        repeat_num = max_class_num//len(cid)-1 if max_class_num//len(cid)>=2 else 1
                        for j in range(repeat_num):
                            df_temp  = df[df['cid'].isin(cid)].copy(deep=True)
                            df_temp['cid'] = df_temp['cid'].apply(lambda x:f'{x}_{j}')
                            df_temp['mid'] = df_temp['mid'].apply(lambda x:f'{x}_{j}')
                            df_temp['pid'] = df_temp['pid'].apply(lambda x:f'{x}_{j}' if x!='None' else x)
                            df_temp.index = df_temp['mid']
                            df_temp.index.name = None
                            df_new = df_new.append(df_temp)
                class_cid_dict = {i:list(pd.unique(df_new[(df_new['event'].isin(train_event)) & (df_new['verify']==i)]['cid'])) for i in range(3)}
                class_num_dict = {i:len(cids) for i, cids in class_cid_dict.items()}  
                cids_train = [cid for cids in class_cid_dict.values() for cid in cids]
                self.dataset = df_new
            else:
                cids_train = list(pd.unique(df[df['event'].isin(train_event)]['cid']))
        random.seed(seed)
        random.shuffle(cids_train)
        self.cids_train = cids_train
        self.cids_val = cids_val
        self.cids_test = cids_test

    def _split_statistics(self):
        '''Analyze label distribution'''
        df = self.dataset
        cids_train, cids_val, cids_test = self.cids_train, self.cids_val, self.cids_test
        events = ['sydneysiege', 'ottawashooting',
                  'ferguson', 'germanwings', 'charliehebdo']
        des = ['train', 'val', 'test']
        split_info = {}
        # compute statasitcal features
        for i, cids in enumerate([cids_train, cids_val, cids_test]):
            dfm = df[df['cid'].isin(cids)]    # all the included messages
            dfs = df[df['mid'].isin(cids)]    # only the source messages
            cas_size_avg = len(dfm)/len(cids)  # averaged cascade size
            event_dist = dfs['event'].value_counts(
            ).to_dict()     # event distribution
            event_dist = '/'.join([str(event_dist.get(event, 0))
                                  for event in events])
            # label distribution of triggers
            label_t_dist = dfm['trigger'].value_counts().to_dict()
            label_t_dist = '/'.join([str(label_t_dist.get(label, 0))
                                    for label in range(4)])
            # label distribution of verifying
            label_v_dist = dfs['verify'].value_counts().to_dict()
            label_v_dist = '/'.join([str(label_v_dist.get(label, 0))
                                    for label in range(3)])
            split_info[f'event_dist_{des[i]}'] = event_dist
            split_info[f'label_t_dist_{des[i]}'] = label_t_dist
            split_info[f'label_v_dist_{des[i]}'] = label_v_dist
        self.split_info = split_info

    def _clean_word(self, word):
        if word and word[0] != "@":
            # split with capitalized letter
            word = re.sub(r'([a-z]+|\d+)([A-Z])', r'\1 \2', word)
            word = word.lower()
        ## extend the abbreviated words
        word = " ".join([abbre_dict.get(sub_word, 0) if abbre_dict.get(sub_word, 0) else sub_word for sub_word in word.split(" ")])
        return word

    def _clean_sentence(self, sentence):
        '''function to clean single sentence'''
        sentence = re.sub('[hH]ttp\S+|www\.\S+', 'HTTPURL', sentence)  # remove url        
        sentence = re.sub('<.*?>+', '', sentence) # remove html tags
        sentence = re.sub('@\S*', '@USER', sentence) # remove @
        sentence = emoji.demojize(sentence) # convert emoji into text                   
        sentence = ' '.join(tweet_tokenizer.tokenize(sentence))      
        sentence = ' '.join([self._clean_word(word) for word in sentence.split()])   
        sentence = re.sub('\s[0-9]+\s', '', sentence) # remove numbers   
        sentence = re.sub('[\.\+\-\?\'\\,/$%&#:;^_`{|}~><“”]', '', sentence) # remove special tokens    
        return sentence

    def _clean_text(self, text_field):
        print('Cleaning text...')
        clean_text_field = f'{text_field}_clean'
        self.dataset[clean_text_field] = self.dataset[text_field].apply(
            self._clean_sentence)

    def setup(self, stage=None):
        self._load_dataset()
        self._split_train_val_test()            # split cids into train/val/test set
        self._split_statistics()                # analyze and record split statistics
        self._clean_text(text_field='content')  # clean text

    @staticmethod
    def collate_fn(item_list):
        flatten_item_list = [x for item in item_list for x in item]
        df_item = pd.DataFrame(flatten_item_list)
        batch = df_item.to_dict(orient='list')
        batch = {k: default_collate(v) for k, v in batch.items()}
        return batch

    def train_dataloader(self):
        df_train = self.dataset[self.dataset['cid'].isin(self.cids_train)]
        return DataLoader(
            dataset=TriggerDataset(df_train),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.args.test:
            df_val = self.dataset[self.dataset['cid'].isin(self.cids_test)]
        else:
            df_val = self.dataset[self.dataset['cid'].isin(self.cids_val)]
        return DataLoader(
            dataset=TriggerDataset(df_val),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        df_test = self.dataset[self.dataset['cid'].isin(self.cids_test)]
        return DataLoader(
            dataset=TriggerDataset(df_test),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

