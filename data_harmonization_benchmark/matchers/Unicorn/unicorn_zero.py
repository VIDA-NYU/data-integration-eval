
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer, DebertaTokenizer, XLNetTokenizer, DistilBertTokenizer
import torch

import os
import random
import csv
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from unicorn.model.encoder import (BertEncoder, MPEncoder, DistilBertEncoder, DistilRobertaEncoder, DebertaBaseEncoder, DebertaLargeEncoder,
                   RobertaEncoder, XLNetEncoder)
from unicorn.model.matcher import  Classifier, MOEClassifier
from unicorn.model.moe import MoEModule
from unicorn.trainer import pretrain, evaluate
from unicorn.utils.utils import get_data, init_model, AttrDict
from unicorn.dataprocess import predata, dataformat
from unicorn.utils import param

csv.field_size_limit(500 * 1024 * 1024)


class TrainApp():
    def __init__(
                self,
                pretrain=False,
                seed=42,
                train_seed=42,
                load=False,
                model="bert",
                max_seq_length=128,
                max_grad_norm=1.0,
                clip_value=0.01,
                batch_size=32,
                pre_epochs=10,
                pre_log_step=10,
                log_step=10,
                c_learning_rate=3e-5,
                num_cls=5,
                num_tasks=2,
                resample=0,
                modelname="UnicornZero",
                ckpt="",
                num_data=1000,
                num_k=2,
                scale=20,
                wmoe=1,
                expertsnum=15,
                size_output=768,
                units=1024,
                shuffle=0,
                load_balance=0,
                usecase_path=None,
        ):
            self.pretrain = pretrain
            self.seed = seed
            self.train_seed = train_seed
            self.load = load
            self.model = model
            self.max_seq_length = max_seq_length
            self.max_grad_norm = max_grad_norm
            self.clip_value = clip_value
            self.batch_size = batch_size
            self.pre_epochs = pre_epochs
            self.pre_log_step = pre_log_step
            self.log_step = log_step
            self.c_learning_rate = c_learning_rate
            self.num_cls = num_cls
            self.num_tasks = num_tasks
            self.resample = resample
            self.modelname = modelname
            self.ckpt = ckpt
            self.num_data = num_data
            self.num_k = num_k
            self.scale = scale
            self.wmoe = wmoe
            self.expertsnum = expertsnum
            self.size_output = size_output
            self.units = units
            self.shuffle = shuffle
            self.load_balance = load_balance
            self.usecase_path = usecase_path

    def get_args(self):
        d = AttrDict()
        d.update(dict(
            pretrain=self.pretrain,
            seed=self.seed,
            train_seed=self.train_seed,
            load=self.load,
            model=self.model,
            max_seq_length=self.max_seq_length,
            max_grad_norm=self.max_grad_norm,
            clip_value=self.clip_value,
            batch_size=self.batch_size,
            pre_epochs=self.pre_epochs,
            pre_log_step=self.pre_log_step,
            log_step=self.log_step,
            c_learning_rate=self.c_learning_rate,
            num_cls=self.num_cls,
            num_tasks=self.num_tasks,
            resample=self.resample,
            modelname=self.modelname,
            ckpt=self.ckpt,
            num_data=self.num_data,
            num_k=self.num_k,
            scale=self.scale,
            wmoe=self.wmoe,
            expertsnum=self.expertsnum,
            size_output=self.size_output,
            units=self.units,
            shuffle=self.shuffle,
            load_balance=self.load_balance
        ))
        return d

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(seed)

    def main(self):
        # argument setting
        print("=== Argument Setting ===")
        print("experts", self.expertsnum)
        print("encoder: " + str(self.model))
        print("max_seq_length: " + str(self.max_seq_length))
        print("batch_size: " + str(self.batch_size))
        print("epochs: " + str(self.pre_epochs))
        self.set_seed(self.train_seed)

        if self.model in ['roberta', 'distilroberta']:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        if self.model == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        if self.model == 'mpnet':
            tokenizer = AutoTokenizer.from_pretrained('all-mpnet-base-v2')
        if self.model == 'deberta_base':
            tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        if self.model == 'deberta_large':
            tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large')
        if self.model == 'xlnet':
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        if self.model == 'distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                
        if self.model == 'bert':
            encoder = BertEncoder()
        if self.model == 'mpnet':
            encoder = MPEncoder()
        if self.model == 'deberta_base':
            encoder = DebertaBaseEncoder()
        if self.model == 'deberta_large':
            encoder = DebertaLargeEncoder()
        if self.model == 'xlnet':
            encoder = XLNetEncoder()
        if self.model == 'distilroberta':
            encoder = DistilRobertaEncoder()
        if self.model == 'distilbert':
            encoder = DistilBertEncoder()
        if self.model == 'roberta':
            encoder = RobertaEncoder()
                
        if self.wmoe:
            classifiers = MOEClassifier(self.units) 
        else:
            classifiers = Classifier()
                
        if self.wmoe:
            exp = self.expertsnum
            moelayer = MoEModule(self.size_output, self.units, exp, load_balance=self.load_balance)
        
        if self.load:
            encoder = init_model(encoder, restore=self.ckpt+"_"+param.encoder_path)
            classifiers = init_model(classifiers, restore=self.ckpt+"_"+param.cls_path)
            if self.wmoe:
                moelayer = init_model(moelayer, restore=self.ckpt+"_"+param.moe_path)
        else:
            encoder = init_model(encoder)
            classifiers = init_model(classifiers)
            if self.wmoe:
                moelayer = init_model(moelayer)
        
        train_metrics = []
        if self.pretrain and (not self.shuffle):
            train_sets = []
            test_sets = []
            valid_sets = []
            limit = 40000
            # for key, p in dataformat.entity_alignment_data.items():
            #     if p[0] == "train":
            #         train_sets.append(get_data(p[1]+"train-large.json", num=limit))
            #         valid_sets.append(get_data(p[1]+"valid-large.json", num=limit))
            #         train_metrics.append(p[2])
            # for key, p in dataformat.string_matching_data.items():
            #     if p[0] == "train":
            #         train_sets.append(get_data(p[1]+"train-large.json", num=limit))
            #         valid_sets.append(get_data(p[1]+"valid-large.json", num=limit))
            #         train_metrics.append(p[2])        
            # for key, p in dataformat.new_deepmatcher_data.items():
            #     if p[0] == "train":
            #         train_sets.append(get_data(p[1]+"train.json", num=limit))
            #         valid_sets.append(get_data(p[1]+"valid.json", num=limit))
            #         train_metrics.append(p[2])                
            for key, p in dataformat.schema_matching_data.items():
                if p[0] == "train":
                    print(p[1])
                    train, valid = self.parse_data_dir_gdc(p[1])
                    train_sets.append(train)
                    valid_sets.append(valid)
                    train_metrics.append(p[2])
            # for key, p in dataformat.column_type_data.items():
            #     if p[0] == "train":
            #         train_sets.append(get_data(p[1]+"train.json", num=limit))
            #         valid_sets.append(get_data(p[1]+"valid.json", num=limit))
            #         train_metrics.append(p[2])
            # for key, p in dataformat.entity_linking_data.items():
            #     if p[0] == "train":
            #         train_sets.append(get_data(p[1]+"train.json", num=limit))
            #         valid_sets.append(get_data(p[1]+"valid.json", num=limit))
            #         train_metrics.append(p[2])

            train_data_loaders = []
            valid_data_loaders = []
            
            if self.model in ['bert', 'deberta_base', 'deberta_large', 'distilbert', 'mpnet']:
                for i in range(len(train_sets)):
                    fea = predata.convert_examples_to_features([ [x[0]+" [SEP] " +x[1]] for x in train_sets[i] ], [int(x[2]) for x in train_sets[i]], self.max_seq_length, tokenizer)
                    train_data_loaders.append(predata.convert_fea_to_tensor(fea, self.batch_size, do_train=1))
                for i in range(len(valid_sets)):
                    fea = predata.convert_examples_to_features([ [x[0]+" [SEP] " +x[1]] for x in valid_sets[i] ], [int(x[2]) for x in valid_sets[i]], self.max_seq_length, tokenizer)
                    valid_data_loaders.append(predata.convert_fea_to_tensor(fea, self.batch_size, do_train=0))
            if self.model in ['roberta', 'distilroberta']:
                for i in range(len(train_sets)):
                    fea = predata.convert_examples_to_features_roberta([ [x[0]+" [SEP] "+x[1]] for x in train_sets[i] ], [int(x[2]) for x in train_sets[i]], self.max_seq_length, tokenizer)
                    train_data_loaders.append(predata.convert_fea_to_tensor(fea, self.batch_size, do_train=1))
                for i in range(len(valid_sets)):
                    fea = predata.convert_examples_to_features_roberta([ [x[0]+" [SEP] "+x[1]] for x in valid_sets[i] ], [int(x[2]) for x in valid_sets[i]], self.max_seq_length, tokenizer)
                    valid_data_loaders.append(predata.convert_fea_to_tensor(fea, self.batch_size, do_train=0))
            if self.model == 'xlnet':
                for i in range(len(train_sets)):
                    fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in train_sets[i] ], [int(x[2]) for x in train_sets[i]], self.max_seq_length, tokenizer, cls_token='<cls>', sep_token='<sep>')
                    train_data_loaders.append(predata.convert_fea_to_tensor(fea, self.batch_size, do_train=1))
                for i in range(len(valid_sets)):
                    fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in valid_sets[i] ], [int(x[2]) for x in valid_sets[i]], self.max_seq_length, tokenizer, cls_token='<cls>', sep_token='<sep>')
                    valid_data_loaders.append(predata.convert_fea_to_tensor(fea, self.batch_size, do_train=0))
            print("train datasets num: ", len(train_data_loaders))
            print("valid datasets num: ", len(valid_data_loaders))
            if self.wmoe:
                encoder, moelayer, classifiers = pretrain.train_moe(self.get_args(), encoder, moelayer, classifiers, train_data_loaders, valid_data_loaders, train_metrics)
            else:
                encoder, classifiers = pretrain.train_wo_moe(self.get_args(), encoder, classifiers, train_data_loaders, valid_data_loaders, train_metrics)

            
        test_sets = []
        test_metrics = []

        if self.usecase_path:
            test = self.parse_data_dir_gdc(self.usecase_path, is_test=True)
            test_sets.append(test)
            test_metrics.append("recall")
        else:
            for key, p in dataformat.schema_matching_data.items():
                if p[0] == "test":
                    test = self.parse_data_dir_gdc(p[1], is_test=True)
                    test_sets.append(test)
                    test_metrics.append(p[2])
        # for key,p in dataformat.entity_alignment_data.items():
        #     if p[0] == "test":
        #         test_sets.append(get_data(p[1]+"test.json"))
        #         test_metrics.append(p[2])
        # for key,p in dataformat.string_matching_data.items():
        #     if p[0] == "test":
        #         test_sets.append(get_data(p[1]+"test.json"))
        #         test_metrics.append(p[2])
        # for key,p in dataformat.new_deepmatcher_data.items():
        #     if p[0] == "test":
        #         test_sets.append(get_data(p[1]+"test.json"))
        #         test_metrics.append(p[2])
        # for key,p in dataformat.new_schema_matching_data.items():
        #     if p[0] == "test":
        #         test_sets.append(get_data(p[1]+"test.json"))
        #         test_metrics.append(p[2])
        # for key,p in dataformat.column_type_data.items():
        #     if p[0] == "test":
        #         test_sets.append(get_data(p[1]+"test.json"))
        #         test_metrics.append(p[2])
        # for key,p in dataformat.entity_linking_data.items():
        #     if p[0] == "test":
        #         test_sets.append(get_data(p[1]+"test.json"))
        #         test_metrics.append(p[2])

        test_data_loaders = []
        if self.model in ['bert','deberta_base','deberta_large','distilbert','mpnet']:
            for i in range(len(test_sets)):
                print("======================== ", i)
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in test_sets[i] ], [int(x[2]) for x in test_sets[i]], self.max_seq_length, tokenizer)
                test_data_loaders.append(predata.convert_fea_to_tensor(fea, self.batch_size, do_train=0))
        if self.model in ['roberta','distilroberta']:
            for i in range(len(test_sets)):
                print("======================== ", i)
                fea = predata.convert_examples_to_features_roberta([ [x[0]+" [SEP] "+x[1]] for x in test_sets[i] ], [int(x[2]) for x in test_sets[i]], self.max_seq_length, tokenizer)
                test_data_loaders.append(predata.convert_fea_to_tensor(fea, self.batch_size, do_train=0))
        if self.model=='xlnet':
            for i in range(len(test_sets)):
                fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in test_sets[i] ], [int(x[2]) for x in test_sets[i]], self.max_seq_length, tokenizer, cls_token='<cls>', sep_token='<sep>')
                test_data_loaders.append(predata.convert_fea_to_tensor(fea, self.batch_size, do_train=0))

        print("test datasets num: ",len(test_data_loaders))
        f1s = []
        recalls = []
        accs = []
        for k in range(len(test_data_loaders)):
            print("test datasets : ",k+1)
            if test_metrics[k]=='hit': # for EA
                if self.wmoe:
                    prob = evaluate.evaluate_moe(encoder, moelayer, classifiers, test_data_loaders[k], args=self.get_args(), flag="get_prob", prob_name="prob.json")
                else:
                    prob = evaluate.evaluate_wo_moe(encoder, classifiers, test_data_loaders[k], args=self.get_args(), flag="get_prob", prob_name="prob.json")
                evaluate.calculate_hits_k(test_sets[k], prob)
                continue
            if self.wmoe:
                f1, recall, acc = evaluate.evaluate_moe(encoder, moelayer, classifiers, test_data_loaders[k], args=self.get_args(), all=1)
            else:
                f1, recall, acc = evaluate.evaluate_wo_moe(encoder, classifiers, test_data_loaders[k], args=self.get_args(), all=1)
            f1s.append(f1)
            recalls.append(recall)
            accs.append(acc)
        print("F1: ", f1s)
        print("Recall: ", recalls)
        print("ACC.", accs)

    def parse_data_dir_gdc(self, data_dir, is_test=False):
        data_name = data_dir
        source = pd.read_csv(os.path.join(data_dir, "source.csv"))
        target = pd.read_csv(os.path.join(data_dir, "target.csv"))
        # gt = pd.read_csv("../../datasets/dou/groundtruth.csv")
        ground_truth=pd.read_csv(os.path.join(data_dir, "groundtruth.csv"))
        dataset_json = self.convert_dataset_to_unicorn_tokens(source, target, ground_truth, is_test)

        if not is_test:
            train, valid = train_test_split(dataset_json, test_size=0.3)
            return train, valid
        else:
            return dataset_json


    def convert_dataset_to_unicorn_tokens(self, source, target, ground_truth, is_test=False):
        dataset_json = []
        for row in ground_truth.itertuples():
            source_colname = row.source
            source_values = source[source_colname].unique()[:20]
            source_str = f"[ATT] {source_colname} [VAL] {' [VAL] '.join(str(x) for x in list(source_values))}"
    
            matching_targets = [target.strip() for target in row.target.split(";")]
            if True:
                for target_colname in matching_targets:
                    target_values = target[target_colname].unique()[:20]
                    target_str = f"[ATT] {target_colname} [VAL] {' [VAL] '.join(str(x) for x in list(target_values))}"
                    dataset_json.append([source_str, target_str, 1])
                
            else:
                for target_colname in target.columns:
                    is_matching = 0
                    if target_colname in matching_targets:
                        is_matching = 1
                    target_values = target[target_colname].unique()[:20]
                    target_str = f"[ATT] {target_colname} [VAL] {' [VAL] '.join(str(x) for x in list(target_values))}"
                    dataset_json.append([source_str, target_str, is_matching])
        return dataset_json

        



