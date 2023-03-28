# -*- coding: utf-8 -*-


"""#Data"""

import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback

def random_idx(start,end):
  return random.randint(start,end)

def gen_target_df(df, target_domain):
  df.rename(columns = {'encoded_label': 'labels'}, inplace=True)
  target_df = df[df["troll_domain"] == target_domain]
  print('target dataframe size ', target_df.shape)
  #target_df = shuffle(target_df)
  target_df_troll = target_df[target_df['label']=='Troll']
  target_df_nontroll = target_df[target_df['label']=='Non-Troll']
  target_df_troll = shuffle(target_df_troll)
  target_df_nontroll = shuffle(target_df_nontroll)
  #support and query split
  return target_df_troll, target_df_nontroll


def gen_dataset(idx,rnd_state,target_df_troll, target_df_nontroll,num_shots):
  #print('target_df troll', target_df_troll.head())
  shuffled_troll = target_df_troll.sample(frac=1, random_state=rnd_state).reset_index()
  shuffle_nontroll = target_df_nontroll.sample(frac=1, random_state=rnd_state).reset_index()

  #print('shuffled head ', shuffled_troll.head())

  train_troll_df = shuffled_troll.iloc[idx:num_shots+idx]
  train_nontroll_df = shuffle_nontroll.iloc[idx:num_shots+idx]
  vali_troll_df = shuffled_troll.iloc[num_shots+idx:num_shots*2+idx]
  vali_nontroll_df = shuffle_nontroll.iloc[num_shots+idx:num_shots*2+idx]
  test_troll_df = shuffled_troll.iloc[num_shots*2+idx:num_shots*3+idx]
  test_nontroll_df = shuffle_nontroll.iloc[num_shots*2+idx:num_shots*3+idx]

  #target_df_train_vali = pd.concat([train_vali_troll_df,train_vali_nontroll_df])
  #train_df, vali_df = train_test_split(target_df_train_vali, test_size=0.5)
  train_df = pd.concat([train_troll_df, train_nontroll_df])
  vali_df = pd.concat([vali_troll_df, vali_nontroll_df])
  test_df = pd.concat([test_troll_df, test_nontroll_df])
  train_df = shuffle(train_df)
  vali_df = shuffle(vali_df)

  train_dataset = Dataset.from_pandas(train_df)
  vali_dataset = Dataset.from_pandas(vali_df)
  test_dataset = Dataset.from_pandas(test_df)


  ds = DatasetDict()

  ds['train'] = train_dataset
  ds['validation'] = vali_dataset
  ds['test'] = test_dataset

  print('*********df trian',ds['train'][0])
  return ds



# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def init_model():
  # Define pretrained tokenizer and model
  #model_name = "bert-base-uncased"
  #tokenizer_name = "bert-base_uncased"
  model_name = "/content/drive/MyDrive/TrollBERT_pretrain/v1/model/"
  tokenizer_name = "/content/drive/MyDrive/TrollBERT_pretrain/v1/vocab/"
  tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
  model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
  return model



def start_training(model, dataset,is_train=True):
  eval_vali_acc = 0
  eval_vali_res = None
  if is_train:
    args = TrainingArguments(
        output_dir="./train_output",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        seed=0,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train pre-trained model
    print('####### Start Training #######')
    trainer.train()
    print('####### Start Evaluation ######')
    eval_vali_res = trainer.evaluate()
    print('eval_vali_res', eval_vali_res)
    eval_vali_acc = eval_vali_res['eval_accuracy']
  eval_trainer = Trainer(
      model=model,
      args=TrainingArguments(output_dir="./test_output", remove_unused_columns=False,),
      eval_dataset=dataset["test"],
      compute_metrics=compute_metrics,
    )
  print('###Start Testing ###')
  eval_res = eval_trainer.evaluate()
  eval_acc = eval_res['eval_accuracy']
  
  return model, eval_vali_res, eval_res, eval_acc, eval_vali_acc

from transformers import BertTokenizerFast
tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/TrollBERT_pretrain/v1/vocab/")
#tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["clean_tweets_string"], max_length=240, truncation=True, padding="max_length")

import statistics

def meta_train(is_model,loop_size, target_domain, num_shots,is_train=True):
  test_lst = []
  vali_lst = []

  #print('model,',model)
  for i in range(loop_size):
    if is_model is None:
      model = init_model()
    else:
      model = is_model
    #model = init_model_with_adapter()
    #tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/AdapertBERT_pretrain_v2/vocab/")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


    idx = random_idx(0,num_shots)
    rand_state = i
    
    #dataset = set_dataset(idx, df, num_shots, target_domain)
    target_df_troll, target_df_nontroll = gen_target_df(df, target_domain)
    ds = gen_dataset(idx,rand_state,target_df_troll, target_df_nontroll,num_shots)
    #ds = gen_ds(idx,df, num_shots, target_domain)
    # Encode the input data
    dataset = ds.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    #dataset.rename_column("encoded_label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    model, vali_res, test_res, test_acc, vali_acc = start_training(model, dataset,is_train)
    print('******i*******',i)
    print('test_res',test_res)
    test_lst.append(test_acc)
    vali_lst.append(vali_acc)
    print('END I *******',i)
  return test_lst, vali_lst,model

class Result(object):
  def __init__(self, domain, test_lst, vali_lst, num_shots):
    self.domain = domain
    self.test_lst = test_lst
    self.vali_lst = vali_lst
    self.num_shots = num_shots

    self.test_mean = statistics.mean(self.test_lst)
    self.vali_mean = statistics.mean(self.vali_lst)
    self.test_min = min(self.test_lst)
    self.test_median = statistics.median(self.test_lst)
  
  def to_dict(self):
    return {
        'domain': self.domain,
        'test_lst': self.test_lst,
        'vali_lst': self.vali_lst,
        'num_shots': self.num_shots,
        'test_mean': self.test_mean,
        'vali_mean': self.vali_mean,
        'test_min': self.test_min,
        'test_median': self.test_median

    }
