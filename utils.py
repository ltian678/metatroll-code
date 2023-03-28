import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
import random
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import pandas as pd
import statistics

import numpy as np

import math
import torch
import torch.nn.functional as F

import logging
logger = logging.getLogger()
logging.basicConfig(filename='meta_utils.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))


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


def gen_meta_dataset(idx,rnd_state,target_df_troll, target_df_nontroll,num_shots):
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

  return ds


def gen_stage1_dataset(data_file):
	df = pd.read_pickle(data_fille)
	train_df = df[df['split']=='train']
	vali_df = df[df['split']=='vali']
	test_df = df[df['split']=='test']

	train_dataset = Dataset.from_pandas(train_df)
	vali_dataset = Dataset.from_pandas(vali_df)
	test_dataset = Dataset.from_pandas(test_df)

	ds = DatasetDict()
	ds['train'] = train_dataset
	ds['validation'] = vali_dataset
	ds['test'] = test_dataset

	return ds




# Define Trainer parameters
def compute_metrics(p):
	pred, labels = p
	pred = np.argmax(pred, axis=1)

	accuracy = accuracy_score(y_true=labels, y_pred=pred)
	recall = recall_score(y_true=labels, y_pred=pred)
	precision = precision_score(y_true=labels, y_pred=pred)
	f1 = f1_score(y_true=labels, y_pred=pred)

	return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def random_seed(value):
	torch.backends.cudnn.deterministic=True
	torch.manual_seed(value)
	torch.cuda.manual_seed(value)
	np.random.seed(value)
	random.seed(value)

def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
	idxs = list(range(0,len(taskset)))
	if is_shuffle:
		random.shuffle(idxs)
	for i in range(0,len(idxs), batch_size):
		yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]


# Result object is created to store all the results
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


def convert_results_to_pd(result_lst, output_dir):
	"""
	Convert meta-test results to data frame
	"""
	results = []
	for res in result_lst:
		results_single = Result(res['domain'], res['test_lst'],res['vali_lst'], res['num_shots'], res['test_mean'], res['vali_mean'], res['test_min'], res['test_median'])
		results.append(results_single)
	res_df = pd.DataFrame(results)
	res_df.to_pickle(otuput_dir)
	return pd.DataFrame(results)


def loss(test_logits_sample, test_labels, device=None):
	"""
	Compute the cross entropy loss.
	"""
	return F.cross_entropy(test_logits_sample, test_labels)

def aggregate_accuracy(test_logits_sample, test_labels):
	"""
	Compute classification accuracy.
	"""
	# averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
	return torch.mean(torch.eq(test_labels, torch.argmax(test_logits_sample, dim=-1)).float())

def linear_classifier(x, param_dict):
	"""
	Classifier.
	"""
	return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])


def mean_pooling(x):
	return torch.mean(x, dim=0, keepdim=True)

def extract_indices(seq, target):
	mask = torch.eq(seq, target) 
	mask_indices = torch.nonzero(mask, as_tuple=False) 
	return torch.reshape(mask_indices, (-1,))  # reshape to be a 1D vector

