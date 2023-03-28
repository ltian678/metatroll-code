# coding=utf-8
""" This the 2nd stage of MetaTroll """
import argparse
import os
import pathlib
import numpy as np
import random
import time
import datetime
import pandas as pd
import logging

import torch
from sklearn.metrics import roc_auc_score
from torch import  nn
import torch.nn.functional as F

from trollData import load_data_file


from model import BERTTrollClassifierSQ, PTBERTTrollClassifierSQ, BertWithAdapterSQ, RobertaWithAdapterSQ
from transformers import get_linear_schedule_with_warmup

from utils import flat_accuracy, format_time

from torch.autograd import grad
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,Subset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from traitlets.traitlets import default

# init a logger for stage II
logging.basicConfig(filename='metaTroll_stage2.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('This is the Stage II of MetaTroll')



# Process of this stage is only meta-train the one general adapter 


"""#Start Training"""


class Step2TrainingArgs:
    def __init__(self):
        self.num_labels = 2
        self.meta_epoch=10
        self.k_support=10
        self.k_query=10
        self.outer_batch_size = 2
        self.inner_batch_size = 12
        self.outer_update_lr = 1e-5
        self.inner_update_lr = 5e-5
        self.inner_update_step = 10
        self.inner_update_step_eval = 40
        self.bert_model = 'bert-base-uncased' #'xlmr'
        self.num_task_train = 5000
        self.num_task_test = 100

args = Step2TrainingArgs()



class MetaTask(Dataset):
    
    def __init__(self, args, examples, num_task, k_support, k_query, tokenizer):
        """
        :param samples: list of samples
        :param num_task: number of training tasks.
        :param k_support: number of support sample per task
        :param k_query: number of query sample per task
        """
        self.examples = examples
        random.shuffle(self.examples)

        self.args = args
        
        self.num_task = num_task
        self.k_support = k_support
        self.k_query = k_query
        self.tokenizer = tokenizer
        self.create_batch(self.num_task)
    
    def create_batch(self, num_task):
        self.supports = []  # support set
        self.queries = []  # query set
        
        for b in range(num_task):  # for each task
            # 1.select domain randomly
            domain_lst = list(set([e.Domain for e in self.examples]))
            domain = random.choice(domain_lst)
            domainExamples = [e for e in self.examples if e.Domain == domain]
            
            # 1.select k_support + k_query examples from domain randomly
            selected_examples = random.sample(domainExamples,self.k_support + self.k_query)
            random.shuffle(selected_examples)
            exam_train = selected_examples[:self.k_support]
            exam_test  = selected_examples[self.k_support:]
            
            self.supports.append(exam_train)
            self.queries.append(exam_test)

    def create_feature_set(self,examples):
        all_input_ids      = torch.empty(len(examples), self.args.max_seq_length, dtype = torch.long)
        all_attention_mask = torch.empty(len(examples), self.args.max_seq_length, dtype = torch.long)
        all_segment_ids    = torch.empty(len(examples), self.args.max_seq_length, dtype = torch.long)
        all_label_ids      = torch.empty(len(examples), dtype = torch.long)

        for id_,example in enumerate(examples):
            input_ids = tokenizer.encode(example.Tweets, max_length=self.args.max_seq_length, truncation=True, padding="max_length")
            attention_mask = [1] * len(input_ids)
            segment_ids    = [0] * len(input_ids)

            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                attention_mask.append(0)
                segment_ids.append(0)

            label_id = example.Label
            all_input_ids[id_] = torch.Tensor(input_ids).to(torch.long)
            all_attention_mask[id_] = torch.Tensor(attention_mask).to(torch.long)
            all_segment_ids[id_] = torch.Tensor(segment_ids).to(torch.long)
            all_label_ids[id_] = torch.Tensor([label_id]).to(torch.long)

        tensor_set = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)  
        return tensor_set
    
    def __getitem__(self, index):
        support_set = self.create_feature_set(self.supports[index])
        query_set   = self.create_feature_set(self.queries[index])
        return support_set, query_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.num_task






class MetaAdapter(nn.Module):
	""" 
        Meta-learning adapter in this step.
    """
	def __init__(self, args, tokenizer, model_classifier):
		super().__init__()
		self.args = args
		self.tokenizer = tokenizer
		self.base_classifier = model_classifier

	def init_model(self):
		""" Initilize model, load pretrained model and tokenizer. """
		base_config = self.base_classifier.from_pretrained(self.bert_model_name,
			cache_dir=self.args.config_cache_dir if self.args.config_cache_dir else None)

		# Load tokenizer
		tokenizer = self.tokenizer.from_pretrained(self.bert_model_name,
			do_lower_case=True,
			cache_dir=self.args.tokenizer_cache_dir if self.args.tokenizer_cache_dir else None)

		# Load model
		if self.args.load_pt:
			logger.info('loading pretrained base model from ' + self.args.base_pretrained)
			self.model = BertWithAdapterSQ(base_config,
									 pt_encoder_state_dict=torch.load(self.args.base_pretrained)['model_state_dict'])
		else:
			self.model = BertWithAdapterSQ(base_config)
		self.model.to(self.args.device)
		
		return tokenizer, model

	def setup_grad(self):
		if self.args.freeze_base_model:
			# Freenze the BERT model with linear. bert/roberta
			for p in self.bert.parameters():
				p.requires_grad = False
		else:
			# Only adapters and last linear layer are trainable.
			for n, p in self.bert.named_parameters():
				if not 'adapter' in n:
					p.requires_grad = False

		if self.args.freeze_linear_layer:
			# Whether to freeze the last linear layer.
			for p in self.linear.parameters():
				p.requires_grad = False

	def setup_data(self):
		filepath = self.args.source_data_filepath
		meta_trainset, meta_vali_set, meta_testset = load_data_file(self.args,filepath)
		self.meta_trainset = meta_trainset
		self.meta_vali_set = meta_vali_set
		self.meta_testset = meta_testset
		return meta_trainset, meta_vali_set, meta_testset

	def save_checkpoint(self, iteration, name='checkpoint'):
		checkpoint_output_dir = os.path.join(self.args.output_dir, name)
		if not os.path.exists(checkpoint_output_dir):
			os.makedirs(checkpoint_output_dir)

		temp_path = os.path.join(checkpoint_output_dir, 'temp.pt')
		torch.save({
			'iteration': iteration,
			'optimizer_state_dict': self.optimizer.state_dict(),
			'best_accuracy': self.validation_accuracies.get_current_best_accuracy_dict(),
			'model_state_dict': self.model.state_dict(),
			'args': self.args
		}, temp_path)
		os.replace(temp_path, os.path.join(checkpoint_output_dir, self.args.checkpoint_name))
		logger.info(f"Saved iteration {iteration} to {checkpoint_output_dir}")


	def train(self):
		self.model.train()
		self.model.zero_grad()

		train_accuracies = []
		losses = []
		task_name_list = []
		sum_pi_grads = None # Accumulation of gradients of Psi networks

		num_train_epochs = self.args.num_training_epochs
		num_episode_per_epoch = self.args.train_num_episode_per_epoch
		num_episode_per_iteration = self.args.num_episodes_per_device * max(1, len(self.device_list)) * self.args.num_iterations_per_optimize_step
		total_iterations = num_episode_per_epoch * num_train_epochs // num_episode_per_iteration
		logger.info(f'Total training iterations: {total_iterations}, num epochs: {num_train_epochs}, num episode per iteration: {num_episode_per_iteration}')

		for iteration in range(self.start_iteration, total_iterations):
			# Each iteration is training on one task

			# Sample a task
			meta_batch = self.get_train_episode(self.args.num_episodes_per_device, self.device_list)
			task_name_list += meta_batch.task_name_list

			# Train on the task
			outputs = self.model(meta_batch)
			task_loss, task_accuracy = outputs[0], outputs[1]
			if len(task_loss.shape) > 0:
				task_loss = sum(task_loss)
			# Whether to use the output gradients
			use_output_grads = True if len(outputs) > 2 else False
			task_pi_grads = outputs[2] if use_output_grads else None

			task_loss = task_loss / self.args.num_episodes_per_optimize_step
			if not use_output_grads:
				task_loss.backward()
			else:
				# Store the output gradients and apply them in the next optimization step
				if sum_psi_grads is None:
					sum_psi_grads = task_pi_grads
				else:  # Accumulate all gradients from different episode learner
					sum_psi_grads = [torch.add(i, j) for i, j in zip(sum_psi_grads, task_pi_grads)]

			train_accuracies.append(task_accuracy)
			losses.append(task_loss.item())

			num_episodes_so_far = (iteration + 1) * self.args.num_episodes_per_device * max(1, len(self.device_list))
			if ( (iteration + 1) % self.args.num_iterations_per_optimize_step == 0) or (iteration == (total_iterations - 1)):
				if not use_output_grads:
					self.optimizer.step()
					self.optimizer.zero_grad()
				else:
					sum_psi_grads = [g.mean(dim=0) for g in sum_psi_grads] 
					dummy_loss = self.model.dummy_forward(meta_batch)
					self.optimize_with_psi_gradients(dummy_loss, sum_psi_grads)

				t_loss = sum(losses)
				t_acc = torch.cat(train_accuracies).sum().item() / self.args.num_episodes_per_optimize_step
				
				results = { 'train_loss': t_loss, 'train_acc': t_acc, 'iteration': iteration, 'num_episodes': num_episodes_so_far }
				logger.info(results)

				if not len(task_name_list) == self.args.num_episodes_per_optimize_step == torch.cat(train_accuracies).shape[0]:
					logger.warning("Number of episode in this optimization step does map the predefined number."
								   "This may due to break in the middle of an optimization step in preview running.")
					
				train_accuracies = []
				losses = []
				task_name_list = []

			if (iteration+1)%self.args.val_freq==0 or iteration==(total_iterations-1):
				self.model.eval()
				accuracy_dict = self.validate()
				logger.info('Validation results:')
				logger.info('\n'.join([f'{k}: {v}' for k, v in accuracy_dict.items()]))
				res_dict= {'eval_'+k : v for k, v in accuracy_dict.items()}
				res_dict['eval_iteration'] = iteration
				res_dict['eval_num_episodes'] = num_episodes_so_far
				# save the model if validation is the best so far
				if self.validation_accuracies.is_better(accuracy_dict):
					self.validation_accuracies.replace(accuracy_dict)
					torch.save(self.model.state_dict(), self.checkpoint_path_validation)
					logger.info('Best validation model was updated.')
					self.save_checkpoint(iteration + 1, self.CHECKPOINT_DIR_NAME['current-best'])
				self.model.train()

			if (iteration + 1) % self.args.checkpoint_freq == 0:
				self.save_checkpoint(iteration + 1, self.CHECKPOINT_DIR_NAME['latest'])
			
			if self.validation_accuracies.early_stop():
				logger.warn(f"Haven't improved for {self.validation_accuracies.earyly_stop_steps} steps. \
					Stop training and save the current iteration to {self.CHECKPOINT_DIR_NAME['final']}")
				break
		
			if (iteration + 1) % self.args.res_print_freq == 0:
				logger.info(f'Finished {iteration+1} training iterations.')

		self.save_checkpoint(iteration + 1, self.args.checkpint_output_dir['final'])

	def clip_grad_norm_(self, grads, max_norm, norm_type=2):
		r"""Clips gradient norm of an iterable of parameters.

		The norm is computed over all gradients together, as if they were
		concatenated into a single vector. Gradients are modified in-place.

		Arguments:
			parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
				single Tensor that will have gradients normalized
			max_norm (float or int): max norm of the gradients
			norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
				infinity norm.

		Returns:
			Total norm of the parameters (viewed as a single vector).
		"""
		max_norm = float(max_norm)
		norm_type = float(norm_type)
		if norm_type == torch._six.inf:
			total_norm = max(g.data.abs().max() for g in grads)
		else:
			total_norm = 0
			for g in grads:
				param_norm = g.data.norm(norm_type)
				total_norm += param_norm.item() ** norm_type
			total_norm = total_norm ** (1. / norm_type)
		clip_coef = max_norm / (total_norm + 1e-6)
		if clip_coef < 1:
			for g in grads:
				g.data.mul_(clip_coef)
		return total_norm

	def optimize_with_psi_gradients(self, dummy_loss, sum_grads_pi):
		self.optimizer.zero_grad()
		dummy_loss.backward()
		self.clip_grad_norm_(sum_grads_pi, self.args.max_grad_norm)
		with torch.no_grad():
			for p, g in zip(self.model.meta_parameters(), sum_grads_pi):
				assert p.shape == g.shape
				p.grad.copy_(g.data)
		self.optimizer.step()


	def validate(self):
		logger.info("***** Running evaluation *****")
		accuracy_dict ={}
		for batch in self.meta_vali_set.val_episode_loop(num_episodes_per_device=self.args.num_episodes_per_device,
													   device_list=self.device_list,
													   max_num_episode=self.args.max_num_val_episodes):
			loss, acc = self.model(batch, eval=True)
			task_name_list = batch.task_name_list
			assert (acc != -1).sum().item() == len(task_name_list)
			for task_name, acc in zip(task_name_list, acc):
				if task_name in accuracy_dict:
					accuracy_dict[task_name].append(acc.item())
				else:
					accuracy_dict[task_name] = [acc.item()]

		for task in accuracy_dict:
			accuracies = accuracy_dict[task]
			accuracy = np.array(accuracies).mean() * 100.0
			confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

			accuracy_dict[task] = {"accuracy": accuracy, "confidence": confidence}
			logger.info(accuracy_dict)

		return accuracy_dict

	def test(self):
		logger.info('Testing on the test data')
		self.model.eval()
		device_list = self.device_list if self.args.parallel_episode_per_device else [self.device_list[0]]

		if '-' in self.args.mode:
			test_shot_list = [int(self.args.mode.split('-')[-1])]
		else:
			test_shot_list = self.meta_testset.k_support

		for num_shots in test_shot_list:
			for task in self.meta_testset.task_list:
				r_file = os.path.join(test_dir, f'K{num_shots}_{task}_results.bin')

				if os.path.exists(r_file):
					logger.info(f'Test results for {num_shots} found in {r_file}, skip testing on this.')
				else:
					specified_task_list = [task]

				acc_list = []
				for meta_batch in self.meta_testset.episode_loop(self.args.num_episodes_per_device,
														device_list, num_shots, validation=False,
														specified_task_list=specified_task_list):
					_, acc = self.model(meta_batch, eval=True)
					task_name_list = meta_batch.task_name_list
					assert (acc != -1).sum().item() == len(task_name_list)
					for task_name, acc in zip(task_name_list, acc):
						assert acc != -1
						acc_list.append(acc.item())
						logger.info(f'Test task: {task_name}, {num_shots} shots, acc: {acc.item()}')

				accuracy = np.array(acc_list).mean() * 100.0
				std = np.array(acc_list).std() * 100.0
				confidence = (196.0 * std) / np.sqrt(len(acc_list))

				r_dict = {"accuracy": accuracy, "confidence": confidence, "std": std}
				logger.info(f'Test task: {task}, mean acc: {accuracy}, std: {std}, confidence: {confidence}')

				logger.info(f'Saving testing results to {r_file}')
				torch.save(r_dict, r_file)

class TrainingArgs_Stage2:
	def __init__(self):
		self.num_labels = 2
		self.meta_epoch=10
		self.k_support=50
		self.k_query=10
		self.outer_batch_size = 2
		self.inner_batch_size = 12
		self.outer_update_lr = 5e-5
		self.inner_update_lr = 5e-5
		self.inner_update_step = 10
		self.inner_update_step_eval = 40
		self.bert_model = 'bert-base-uncased'
		self.num_task_train = 500
		self.num_task_test = 5
		self.train_domains = [ ]
		self.validation_domains = []
		self.test_domains = []
		self.train_num_task = 
		self.train_k_support = 
		self.train_k_query = 
		self.vali_num_task = 
		self.vali_k_support = 
		self.vali_k_query = 
		self.test_num_task = 
		self.test_k_support = 
		self.test_k_query = 

##args.train_domains, validation_domains, test_domains, train_num_task, train_k_support, train_k_query, vali_num_task, vali_k_support, vali_k_query, test_num_task, test_k_support, test_k_query

def main():
	# Set the seed value all over the place to make this reproducible.
	seed_val = 42

	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)
	# Arg parser initialization and parsing TODO: need to filling in the int_arg_parser here!!!!
	args = TrainingArgs_Stage2()

	data = gen_stage2_dataset(args.raw_data_file)

	meta_adapter = MetaAdapter(args,tokenizer, model_classifier)
	meta_adapter.train()

if __name__ == "__main__":
	main()


>>>>>>> 989f76edf30952a39e503c6b7a034cf030b92c2f

