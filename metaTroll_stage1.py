# coding=utf-8
""" This the first stage of MetaTroll """

import argparse
import os
import pathlib
import numpy as np
import random
import time
import datetime
import pandas as pd

import torch
from sklearn.metrics import roc_auc_score
from torch import  nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

from sklearn.metrics import classification_report


from model import BERTTrollClassifierSQ, PTBERTTrollClassifierSQ
from transformers import get_linear_schedule_with_warmup

from utils import flat_accuracy, format_time


import logging
# init a logger for stage I 
logging.basicConfig(filename='metaTroll_stage1.log', level=logging.INFO)

logger = logging.getLogger(__name__)

logger.info('This is the Stage I')


class FineTuningBaseModel(nn.Module):
	""" Fine tuning the base model
		No meta-learning happens in this step.
	"""
	def __init__(self, args, tokenizer, model_classifier):
		super().__init__()
		self.args = args
		self.tokenizer = tokenizer
		self.base_classifier = model_classifier

	def forward(self, data):
		device = self.args.device
		train_dataloader, validation_dataloader, test_dataloader = self.data_handler(data)
		base_classifier, optimizer, scheduler = self.init_model(data)
		training_stats = []

		# Measure the total training time for the whole run.
		total_t0 = time.time()

		# For each epoch...
		for epoch_i in range(0, self.args.num_epoch_s1):
			
			# ========================================
			#               Training
			# ========================================
			
			# Perform one full pass over the training set.

			logger.info("")
			logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
			logger.info('Training...')

			# Measure how long the training epoch takes.
			t0 = time.time()

			# Reset the total loss for this epoch.
			total_train_loss = 0

			base_classifier.train()

			# For each batch of training data...
			for step, batch in enumerate(train_dataloader):

				# Progress update every 40 batches.
				if step % 40 == 0 and not step == 0:
					# Calculate elapsed time in minutes.
					elapsed = format_time(time.time() - t0)
					
					# Report progress.
					logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


				b_input_ids = batch[0].to(device)
				b_input_mask = batch[1].to(device)
				b_labels = batch[2].to(device)

				base_classifier.zero_grad()        

				loss, logits = base_classifier(b_input_ids, 
									 token_type_ids=None, 
									 attention_mask=b_input_mask, 
									 labels=b_labels)

				total_train_loss += loss.item()

				loss.backward()

				torch.nn.utils.clip_grad_norm_(base_classifier.parameters(), 1.0)

				optimizer.step()

				# Update the learning rate.
				scheduler.step()

			avg_train_loss = total_train_loss / len(train_dataloader)            
			
			training_time = format_time(time.time() - t0)

			logger.info("")
			logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
			logger.info("  Training epcoh took: {:}".format(training_time))
				
			# ========================================
			#               Validation
			# ========================================

			logger.info("")
			logger.info("Running Validation...")

			t0 = time.time()

			# Put the model in evaluation mode--the dropout layers behave differently
			# during evaluation.
			base_classifier.eval()

			# Tracking variables 
			total_eval_accuracy = 0
			total_eval_loss = 0
			nb_eval_steps = 0

			# Evaluate data for one epoch
			for batch in validation_dataloader:

				b_input_ids = batch[0].to(device)
				b_input_mask = batch[1].to(device)
				b_labels = batch[2].to(device)

				with torch.no_grad():        
					(loss, logits) = base_classifier(b_input_ids, 
										   token_type_ids=None, 
										   attention_mask=b_input_mask,
										   labels=b_labels)
					
				# Accumulate the validation loss.
				total_eval_loss += loss.item()

				# Move logits and labels to CPU
				logits = logits.detach().cpu().numpy()
				label_ids = b_labels.to('cpu').numpy()

				total_eval_accuracy += flat_accuracy(logits, label_ids)
				

			# Report the final accuracy for this validation run.
			avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
			logger.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))

			# Calculate the average loss over all of the batches.
			avg_val_loss = total_eval_loss / len(validation_dataloader)
			
			# Measure how long the validation run took.
			validation_time = format_time(time.time() - t0)
			
			logger.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
			logger.info("  Validation took: {:}".format(validation_time))

			# Record all statistics from this epoch.
			training_stats.append(
				{
					'epoch': epoch_i + 1,
					'Training Loss': avg_train_loss,
					'Valid. Loss': avg_val_loss,
					'Valid. Accur.': avg_val_accuracy,
					'Training Time': training_time,
					'Validation Time': validation_time
				}
			)

		logger.info("Training complete!")

		logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

		self.training_stats = training_stats
		return training_stats

	def encode_data(self, a, labels):
		logger.info('Encoding the input sequences and labels ')
		# Tokenize all of the sentences and map the tokens to thier word IDs.
		input_ids = []
		attention_masks = []

		# For every sentence...
		for i,sent in enumerate(a):
			encoded_dict = self.tokenizer.encode_plus(
								text=sent,
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = self.args.max_seq_length,           # Pad & truncate all sentences.
								pad_to_max_length = True,
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
								truncation = True
						   )
			
			# Add the encoded sentence to the list.    
			input_ids.append(encoded_dict['input_ids'])
			
			# And its attention mask (simply differentiates padding from non-padding).
			attention_masks.append(encoded_dict['attention_mask'])

		# Convert the lists into tensors.
		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)
		labels = torch.tensor(labels)
		return input_ids, attention_masks, labels



	def train_data_loading(self,data):
		train_data = data['train']
		validation_data = data['validation']

		train_input_ids, train_attention_masks, train_labels = self.encode_data(train_data['full_text'], train_data['labels'])
		validataion_input_ids, validation_attention_masks, validation_labels = self.encode_data(validation_data['full_text'], validation_data['labels'])

		# Combine the training inputs into a TensorDataset.
		train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
		val_dataset = TensorDataset(validataion_input_ids, validation_attention_masks, validation_labels)

		train_dataloader = DataLoader(
			train_dataset,  # The training samples.
			sampler = RandomSampler(train_dataset), # Select batches randomly
			batch_size = self.args.batch_size_s1_train # Trains with this batch size.
		)

		# For validation the order doesn't matter, so we'll just read them sequentially.
		validation_dataloader = DataLoader(
			val_dataset, # The validation samples.
			sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
			batch_size = self.args.batch_size_s1_train # Evaluate with this batch size.
		)
		return train_dataloader, validation_dataloader

	def test_data_loading(self, data):
		test_data = data['test']
		a = test_data['full_text']
		labels = test_data['labels']

		logger.info('number of labels in the test set',len(labels))
		input_ids, attention_mask, labels = self.encode_data(a, labels)

		# Create the DataLoader.
		prediction_data = TensorDataset(input_ids, attention_masks, labels)
		prediction_sampler = SequentialSampler(prediction_data)
		prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.args.batch_size_s1_test)
		return prediction_dataloader

	def data_handler(self,data):
		train_dataloader, validation_dataloader = self.train_data_loading(data)
		return train_dataloader, validation_dataloader

	def test_data_handler(self,data):
		test_dataloader = self.test_data_loading(data)
		return test_dataloader

	def init_model(self,data):
		base_classifier.to(self.args.device)
		optimizer = AdamW(self.base_classifier.parameters(), lr=self.args.learning_rate_s1)
		epochs = self.args.num_epoch_s1
		train_dataloader, validation_dataloader. test_dataloader = self.data_handler(data)
		total_steps = len(train_dataloader) * epochs
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.num_warmup_steps, num_training_steps=total_steps)


		if self.args.checkpoint_path.lower() != 'none':
			checkpoint = torch.load(os.path.join(self.args.checkpoint_path, "exp_checkpoint.pt"))
			self.base_classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)

		return base_classifier, optimizer, scheduler

	def test(self,data):
		test_dataloader = self.test_data_handler(data)
		# Prediction on test set

		logger.info('Predicting labels for {:,} test sentences...'.format(len(test_dataloader)))

		# Put model in evaluation mode
		self.base_classifier.eval()

		# Tracking variables 
		predictions , true_labels = [], []

		# Predict 
		for batch in test_dataloader:
		  # Add batch to GPU
		  batch = tuple(t.to(device) for t in batch)
		  
		  # Unpack the inputs from our dataloader
		  b_input_ids, b_input_mask, b_labels = batch
		  
		  # Telling the model not to compute or store gradients, saving memory and 
		  # speeding up prediction
		  with torch.no_grad():
			  # Forward pass, calculate logit predictions
			  outputs = self.base_classifier(b_input_ids, token_type_ids=None, 
							  attention_mask=b_input_mask)

		  logits = outputs[0]

		  # Move logits and labels to CPU
		  logits = logits.detach().cpu().numpy()
		  label_ids = b_labels.to('cpu').numpy()
		  
		  # Store predictions and true labels
		  predictions.append(logits)
		  true_labels.append(label_ids)

		logger.info('DONE.')

		interpret_results(predictions, true_labels)



	def export_train_stats(self):
		# Display floats with two decimal places.
		pd.set_option('precision', 2)

		# Create a DataFrame from our training statistics.
		df_stats = pd.DataFrame(data=self.training_stats)

		# Use the 'epoch' as the row index.
		df_stats = df_stats.set_index('epoch')
		self.df_stats = df_stats

		return df_stats

	def draw_training_stats(self):
		# Use plot styling from seaborn.
		sns.set(style='darkgrid')

		# Increase the plot size and font size.
		sns.set(font_scale=1.5)
		plt.rcParams["figure.figsize"] = (12,6)

		# Plot the learning curve.
		plt.plot(self.df_stats['Training Loss'], 'b-o', label="Training")
		plt.plot(self.df_stats['Valid. Loss'], 'g-o', label="Validation")

		# Label the plot.
		plt.title("Training & Validation Loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		plt.xticks([1, 2, 3, 4])

		plt.show()

	def interpret_results(predictions, true_labels):
		# Combine the results across all batches. 
		flat_predictions = np.concatenate(predictions, axis=0)

		#Get the exactly softmax score for each record
		flat_pre = flat_predictions

		# For each sample, pick the label (0 or 1) with the higher score.
		flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

		# Combine the correct labels for each batch into a single list.
		flat_true_labels = np.concatenate(true_labels, axis=0)

		print(classification_report(flat_true_labels, flat_predictions, digits=4))
		logger.info(classification_report(flat_true_labels, flat_predictions, digits=4))


class TrainingArgs_Stage1:
    def __init__(self):
        self.num_labels = 2
        self.bert_model = 'bert-base-uncased'
        self.raw_data_file = 'data/all_troll.pkl'
        self.num_epoch_s1 = 5
        self.batch_size_s1_train = 32
        self.max_seq_length = 384
        self.batch_size_s1_test = 32
        self.learning_rate_s1 = 2e-5
        self.num_warmup_steps = 0
        self.checkpoint_path = 'model/stage1'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def main():
	# Set the seed value all over the place to make this reproducible.
	seed_val = 42

	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)
	# Arg parser initialization and parsing TODO: need to filling in the int_arg_parser here!!!!
	args = TrainingArgs_Stage1()

	data = gen_stage1_dataset(args.raw_data_file)

	ft_base = FineTuningBaseModel(args,tokenizer, model_classifier)
	train_results = ft_base(data)


if __name__ == "__main__":
	main()
