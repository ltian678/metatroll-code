"""Run all the extraction for a model across many templates.
"""
import argparse
import os
import random
import torch
from utils import convert_results_to_pd
from initArgs import init_arg_parser
from trollData import TrollDataENGProcessor,TrollDataMultiProcessor

from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer,XLMRobertaTokenizer

import logging
# init a logger
logging.basicConfig(filename='metaTroll_run.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('This is MetaTroll Run')



class MetaTroll(object):
	def __init__(self, args):
		super().__init__()
		self.args = args


	def run(self):
		if self.args.mode == 'train':
			self.run_train()
		if self.args.mode == 'test':
			self.run_test()

	def run_train(self):
		# Load and prepare model
		if not self.args.start_from_scratch:
			if self.args.checkpoint_path is not None:
				self.resume_from_checkpoint(self.args.checkpoint_path)
			else:
				self.resume_from_latest_or_best(load_from_latest=True, exit_if_failed=False)
		self.setup_dataparallel()

		# Init training dataset
		self.dataset = self._load_train_data(self.args.lang)

		self.train()
	
	def _load_train_data(self):
		if self.args.lang == 'ENG':
			data_dir = self.args.english_data_dir
			return TrollDataENGProcessor(self.args,self.tokenizer, data_dir)
		if self.args.lang == 'Multi':
			data_dir = self.args.multi_data_dir
			return TrollDataMultiProcessor(self.args, self.x_tokenizer, data_dir)
		
		
	def train(self):
		self.model.train()
		self.model.zero_grad()
		train_accuracies = []
		losses = []
		task_name_list = []
		sum_psi_grads = None # Accumulation of gradients of psi networks

		num_train_epochs = self.args.num_training_epochs
		num_episode_per_epoch = self.dataset.train_num_episode_per_epoch
		num_episode_per_iteration = self.args.num_episodes_per_device * max(1, self.args.n_gpu) * self.args.num_iterations_per_optimize_step
		total_iterations = num_episode_per_epoch * num_train_epochs // num_episode_per_iteration
		logger.info(f'Total training iterations: {total_iterations}, num epochs: {num_train_epochs}, num episode per iteration: {num_episode_per_iteration}')

		for iteration in range(self.start_iteration, total_iterations):
			# Sample a batch of tasks.
			meta_batch = self.dataset.get_train_episode(self.args.num_episodes_per_device, self.device_list)
			task_name_list += meta_batch.task_name_list

			# Forward pass.
			outputs = self.model(meta_batch)
			task_loss, task_accuracy = outputs[0], outputs[1]
			if len(self.device_list) > 1:
				task_loss = sum(task_loss)
			# Whether to use the output gradients
			use_output_grads = True if len(outputs) > 2 else False
			task_psi_grads = outputs[2] if use_output_grads else None

			task_loss = task_loss / self.args.num_episodes_per_optimize_step
			if not use_output_grads:
				# If gradients are not output, do backward()
				task_loss.backward()
			else:
				# Store the output gradients and apply them in the next optimization step
				if sum_psi_grads is None:
					sum_psi_grads = task_psi_grads
				else:  # Accumulate all gradients from different episode learner
					sum_psi_grads = [torch.add(i, j) for i, j in zip(sum_psi_grads, task_psi_grads)]

			# Store results of the current batch
			train_accuracies.append(task_accuracy)
			losses.append(task_loss.item())

			# Optimize & Log
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
				
				results = { 'train_loss': t_loss, 'train_acc': t_acc,
							'iteration': iteration, 'num_episodes': num_episodes_so_far }

				train_accuracies = []
				losses = []
				task_name_list = []

			# Validate every val_freq optimization steps
			# if (num_episodes_so_far % (self.args.val_freq * self.args.num_episodes_per_optimize_step) == 0) and (iteration + 1) != total_iterations:
			# if ( (iteration + 1) % (self.args.num_iterations_per_optimize_step * self.args.val_freq) == 0) and (iteration != (total_iterations - 1)):
			if (iteration+1)%self.args.val_freq==0 or iteration==(total_iterations-1):
				self.model.eval()

				accuracy_dict = self.validate()
				logger.info('Validation results:')
				logger.info('\n'.join([f'{k}: {v}' for k, v in accuracy_dict.items()]))

				# save the model if validation is the best so far
				if self.validation_accuracies.is_better(accuracy_dict):
					self.validation_accuracies.replace(accuracy_dict)
					# torch.save(self.model.state_dict(), self.checkpoint_path_validation)
					logger.info('Best validation model was updated.')
					self.save_checkpoint(iteration + 1, self.CHECKPOINT_DIR_NAME['current-best'])

				self.model.train()

			if (iteration + 1) % self.args.checkpoint_freq == 0:
				self.save_checkpoint(iteration + 1, self.CHECKPOINT_DIR_NAME['latest'])
			
			if self.validation_accuracies.early_stop():
				logger.warn(f"Haven't improved for {self.validation_accuracies.earyly_stop_steps} steps. \
					Stop training and save the current iteration to {self.CHECKPOINT_DIR_NAME['final']}")
				break
		
			if (iteration + 1) % self.PRINT_FREQUENCY == 0:
				logger.info(f'Finished {iteration+1} training iterations.')

		# Save the final model
		self.save_checkpoint(iteration + 1, self.CHECKPOINT_DIR_NAME['final'])

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

	def optimize_with_psi_gradients(self, dummy_loss, sum_grads_psi):
		self.optimizer.zero_grad()
		dummy_loss.backward()
		# for g in sum_grads_psi:
		self.clip_grad_norm_(sum_grads_psi, self.args.max_grad_norm)
		with torch.no_grad():
			for p, g in zip(self.model.meta_parameters(), sum_grads_psi):
				assert p.shape == g.shape
				p.grad.copy_(g.data)
		self.optimizer.step()

	def validate(self):
		accuracy_dict ={}
		for batch in self.dataset.val_episode_loop(
				num_episodes_per_device=self.args.num_episodes_per_device,
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

		return accuracy_dict

	def run_test(self):
		# test_lasest, test_best: resume previous experiments and test the latest/best model.
		# test_checkpoing: load the model from another experiment given by the checkpoint file path.
		if 'test_latest' in self.args.mode:
			self.resume_from_latest_or_best(load_from_latest=True, exit_if_failed=True)
		elif 'test_best' in self.args.mode:
			self.resume_from_latest_or_best(load_from_latest=False, exit_if_failed=True)
		elif 'test_checkpoint' in self.args.model and self.args.checkpoint_path is not None:
				self.resume_from_checkpoint(self.args.checkpoint_path, model_only=True)
		else:
			raise ValueError('Exp mode not support: ' + self.args.mode)

		self.setup_dataparallel()
		self.test()

	def test(self):
		logger.info('Testing on the original metaTroll test data')
		if self.args.lang == 'ENG':
			test_set = TrollDataENGProcessor(self.args, self.tokenizer, self.args.test_data_dir)
		if self.args.lang == 'Multi':
			test_set = TrollDataMultiProcessor(self.args, self.x_tokenizer, self.args.multi_data_dir)
		self.model.eval()
		device_list = self.device_list if self.parallel_episode_per_device else [self.device_list[0]]

		if '-' in self.args.mode:
			test_shot_list = [int(self.args.mode.split('-')[-1])]
		else:
			test_shot_list = test_set.NUM_SHOTS_LIST

		for num_shots in test_shot_list:
			test_dir = os.path.join(self.args.output_dir, f"metatroll-test-{self.start_iteration}-iteration")
			pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True) 

			for task in test_set.TEST_DATASET_LIST:
				r_file = os.path.join(test_dir, f'K{num_shots}_{task}_results.bin')

				if os.path.exists(r_file):
					logger.info(f'Test results for {num_shots} found in {r_file}, skip testing on this.')
					continue
				else:
					specified_task_list = [task]

				acc_list = []
				for meta_batch in test_set.episode_loop(self.args.num_episodes_per_device,
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


	def save_checkpoint(self, iteration, name='default'):
		checkpoint_output_dir = os.path.join(self.args.output_dir, name)
		if not os.path.exists(checkpoint_output_dir):
			os.makedirs(checkpoint_output_dir)

		temp_path = os.path.join(checkpoint_output_dir, 'temp.pt')
		torch.save({
			'iteration': iteration,
			'optimizer_state_dict': self.optimizer.state_dict(),
			'best_accuracy': self.validation_accuracies.get_current_best_accuracy_dict(),
			'model_state_dict': self.model.module.state_dict() if type(self.model) is MyDataParallel else self.model.state_dict(),
			'training_args': self.args,
			'rng' : torch.random.get_rng_state(),
			'np_rand_state': np.random.get_state()
		}, temp_path)
		os.replace(temp_path, os.path.join(checkpoint_output_dir, self.CHECKPOINT_FILE_NAME))
		logger.info(f"Saved iteration {iteration} to {checkpoint_output_dir}")

	def resume_from_checkpoint(self, checkpoint_path=None, model_only=False):
		if os.path.exists(os.path.join(checkpoint_path, self.CHECKPOINT_FILE_NAME)):
			logger.info(f'Loading from given checkpoint path: {checkpoint_path}')
			self.load_checkpoint(checkpoint_path, model_only)
		else:
			raise Error(f'Was asked to load from {checkpoint_path} but cound not find checkpoint file in it.')

	def resume_from_latest_or_best(self, load_from_latest=True, exit_if_failed=False):
		ckp_path = os.path.join(self.args.output_dir,
								self.CHECKPOINT_DIR_NAME['current-best'] if not load_from_latest else self.CHECKPOINT_DIR_NAME['latest'])
		if os.path.exists(os.path.join(ckp_path, self.CHECKPOINT_FILE_NAME)):
			logger.info(f'Loading from checkpoint path: {ckp_path}')
			self.load_checkpoint(ckp_path)
		else:
			if not exit_if_failed:
				logger.warning(f'No checkpoint path is given nor found under {ckp_path}. Keep using the initial model.')
			else:
				raise Error(f'No checkpoint path is given nor found under {ckp_path}.')

	def load_checkpoint(self, checkpoint_path, model_only=False):
		checkpoint = torch.load(os.path.join(checkpoint_path, self.CHECKPOINT_FILE_NAME))
		if not model_only:
			self.start_iteration = checkpoint['iteration']
			try:
				self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			except:
				logger.warning('Failed to load optimizer from the given checkpoint, skip loading...')
			self.validation_accuracies.replace(checkpoint['best_accuracy'])
			self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
			# args = checkpoint['training_args']
			self.tokenizer = BertTokenizer.from_pretrained(self.args.checkpoint_path,do_lower_case=args.do_lower_case)
			self.x_tokenizer = XLMRobertaTokenizer.from_pretrained(self.args.x_checkpoint_path)
			torch.random.set_rng_state(checkpoint['rng'])
			np.random.set_state(checkpoint['np_rand_state'])
		else:
			self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
		logger.info(f"Loaded iteration {self.start_iteration} from {checkpoint_path}")



if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

	args = init_arg_parser()

	metaTroll = MetaTroll(args)
	metaTroll.run()
