
import torch
from torch import nn
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from transformers import (
	BertTokenizer, DistilBertTokenizer, RobertaTokenizer,XLMRobertaTokenizer
)
from transformers import (
	BertModel, RobertaModel, XLMRobertaModel
)

from transformers import (BertForSequenceClassification, RobertaForSequenceClassification,XLMRobertaForSequenceClassification)

from transformers import BertSelfOutput,BertOutput


model_dict = {
  'bert-base-uncased':BertModel,
  'roberta-base': RobertaModel,
  'xlm-roberta-base':XLMRobertaModel
}

tokenizer_dict = {
  'bert-base-uncased': BertTokenizer,
  'roberta-base': RobertaTokenizer,
  'xlm-roberta-base':XLMRobertaTokenizer
}

model_sq_dict = {
  'bert-base-uncased': BertForSequenceClassification,
  'roberta-base':RobertaForSequenceClassification,
  'xlm-roberta-base': XLMRobertaForSequenceClassification
}



def init_model(base_model):
  model = model_dict[base_model].from_pretrained(base_model)
  tokenizer = tokenizer_dict[base_model].from_pretrained(base_model)
  return model, tokenizer


def init_model_sq(base_model, num_labels):
  model = model_sq_dict[base_model].from_pretrained(base_model, num_laebls=num_labels)
  tokenizer = tokenizer_dict[base_model].from_pretrained(base_model)
  return model, tokenizer


def init_pretrain_model(base_model,base_model_path, tokenizer_path):
  model = model_dict[base_model].from_pretrained(base_model_path)
  tokenizer = tokenizer_dict[base_model].from_pretrained(tokenizer_path)
  return model, tokenizer


def init_pretrain_model_seq(base_model, base_model_path, tokenizer_path, num_labels):
  model_sq = model_sq_dict[base_model].from_pretrained(base_model_path, num_laebls=num_labels)
  tokenzier = tokenizer_dict[base_model].from_pretrained(tokenizer_path)
  return model_sq, tokenizer


class BERTTrollClassifier(nn.Module):
  def __init__(self, args, base_model):
	super().__init__()
	self.bert, self.tokenizer = init_model(base_model)
	self.dropout = nn.Dropout(args.dropout)
	self.linear = nn.Linear(self.bert.config.hidden_size, args.num_labels)
	self.args_lst = ['input_ids', 'attention_mask', 'token_type_ids']

  def forward(self, batch, start=None, end=None):
	start = 0 if start is None else start
	end = batch['input_ids'].shape[0] if end is None else end
	assert start >=0 and start < end

	embs = self.bert(**{k:v[start:end] for k,v in batch.items() if k in self.args_lst})[1]
	embs = self.dropout(embs)
	embs = self.linear(embs)
	return embs
	
class BERTTrollClassifierSQ(nn.Module):
  def __init__(self, args, base_model):
	super().__init__()
	self.bert, self.tokenizer = init_model_sq(base_model)

  def forward(self, input_ids, attention_mask):
	logits = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
	return logits

class PTBERTTrollClassifierSQ(nn.Module):
  def __init__(self, args, base_model, base_model_path, tokenizer_path):
	super().__init__()
	self.bert, self.tokenizer = init_pretrain_model_seq(base_model, base_model_path, tokenizer_path)

  def forward(self, input_ids, attention_mask):
	logits = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
	return logits


#Adapter
#based on https://arxiv.org/pdf/1902.00751.pdf

class BottleneckAdapter(nn.Module):
	""" Bottleneck layer for model adaptation, with adaptation on the middle layer.
	"""
	def __init__(self, config):
		super().__init__()
		self.down_proj_layer = nn.Linear(config.hidden_size,
										config.bn_adapter_hidden_size)
		self.up_proj_layer = nn.Linear(config.bn_adapter_hidden_size,
									config.hidden_size)
		if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
			self.adapter_act_fn = ACT2FN[config.hidden_act]
		else:
			self.adapter_act_fn = config.hidden_act
		self.layer_norm_n_shape = (config.hidden_size, ) 
		self.layer_norm_eps=config.layer_norm_eps
	
	def forward_given_parameters(self, input_tensor, params):
		hidden_states = F.linear(input_tensor, params['down_proj_weight'], params['down_proj_bias'])
		hidden_states = self.adapter_act_fn(hidden_states)
		output_states = F.linear(hidden_states, params['up_proj_weight'], params['up_proj_bias'])
		output_states = F.layer_norm(output_states, self.layer_norm_n_shape,
										params['layer_norm_weight'],
										params['layer_norm_bias'],
										eps=self.layer_norm_eps)
		return output_states+input_tensor, None

	def forward(self, input_tensor, params=None, return_hidden=False):
		if params is None or 'use_hyper_net' not in params or not params['use_hyper_net']:
			# forward pass using own parameters
			hidden_states = self.down_proj_layer(input_tensor)
			hidden_states = self.adapter_act_fn(hidden_states)
			if params is not None and 'scale' in params:
				hidden_states = hidden_states * params['scale']
			if params is not None and 'shift' in params:
				hidden_states = hidden_states + params['shift']
			output_states = self.up_proj_layer(hidden_states)

			return output_states+input_tensor, None
		else:
			# forward pass using parameters generated by hypernetwork
			return self.forward_given_parameters(input_tensor, params)

	def linear_layer_init_with_zeros(self, input_size, output_size):
		l = nn.Linear(input_size, output_size)
		l.weight.data.fill_(0.)
		l.bias.data.fill_(0.)
		return l

	def mlp_init_last_with_zeros(self, input_size, output_size):
		last_linear = nn.Linear(input_size // 2, output_size)
		last_linear.weight.data.fill_(0.)
		last_linear.bias.data.fill_(0.)
		m_list = nn.Sequential(
			nn.Linear(input_size, input_size // 2),
			nn.ELU(),
			last_linear)
		return m_list


class BertSelfWithAdapter(BertSelfOutput):
	""" Fully connected layer after self-attention layer,
		with the bottleneck adapter layer.
	"""
	def __init__(self, config):
		super().__init__(config)
		self.adapter = BottleneckAdapter(config)
	
	def forward(self, hidden_states, input_tensor, params=None, return_bn_hidden=False):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		# Apply the adaptation layer.
		hidden_states, hidden_rep_tuple = self.adapter(hidden_states, params=params, return_hidden=return_bn_hidden)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)

		return hidden_states, hidden_rep_tuple


class BertOutputWithAdapter(BertOutput):
	def __init__(self, config):
		super().__init__(config)
		self.adapter = BottleneckAdapter(config)

	def forward(self, hidden_states, input_tensor, params=None, return_bn_hidden=False):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		# Apply the adaptation layer.
		hidden_states, hidden_rep_tuple = self.adapter(hidden_states, params, return_hidden=return_bn_hidden)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states, hidden_rep_tuple


class BertWithAdapter(BertModel):
	def __init__(self, config):
		super().__init__(config)
		self.encoder = BertWithAdapter(config)
		if hasattr(config, 'base_pretrained_state_dict'):
			state_dict = self.state_dict()
			for pt_n, pt_p in self.pretrained_state_dict.items():
				if pt_n in state_dict: 
					state_dict[pt_n].copy_(pt_p.data)
		else:
			raise ValueError('base_pretrained_state_dict need to be specified.')
		self.init_weights()

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
				head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
				params_dict_list=None, return_bn_hidden=False):

		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			input_shape = input_ids.size()
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")

		device = input_ids.device if input_ids is not None else inputs_embeds.device

		if attention_mask is None:
			attention_mask = torch.ones(input_shape, device=device)
		if encoder_attention_mask is None:
			encoder_attention_mask = torch.ones(input_shape, device=device)
		if token_type_ids is None:
			token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

		if attention_mask.dim() == 3:
			extended_attention_mask = attention_mask[:, None, :, :]

		# Provided a padding mask of dimensions [batch_size, seq_length]
		# - if the model is a decoder, apply a causal mask in addition to the padding mask
		# - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
		if attention_mask.dim() == 2:
			if self.config.is_decoder:
				batch_size, seq_length = input_shape
				seq_ids = torch.arange(seq_length, device=device)
				causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
				extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
			else:
				extended_attention_mask = attention_mask[:, None, None, :]

		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


		if encoder_attention_mask.dim() == 3:
			encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
		if encoder_attention_mask.dim() == 2:
			encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

		encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
		encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

		if head_mask is not None:
			if head_mask.dim() == 1:
				head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
				head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
			elif head_mask.dim() == 2:
				head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
			head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
		else:
			head_mask = [None] * self.config.num_hidden_layers

		embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
		encoder_outputs, bn_hidden_tuple = self.encoder(embedding_output,
									   attention_mask=extended_attention_mask,
									   head_mask=head_mask,
									   encoder_hidden_states=encoder_hidden_states,
									   encoder_attention_mask=encoder_extended_attention_mask,
									   params_dict_list=params_dict_list,
									   return_bn_hidden=return_bn_hidden)
		sequence_output = encoder_outputs[0]
		pooled_output = self.pooler(sequence_output)

		outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
		outputs += (bn_hidden_tuple,)
		return outputs   # sequence_output, pooled_output, (hidden_states), (attentions) : outputs


class RobertaWithAdapter(RobertaModel):
	def __init__(self, config):
		super().__init__(config)
		self.encoder = RobertaWithAdapter(config)
		self.init_weights()

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
				head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
				params_dict_list=None, return_bn_hidden=False):

		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			input_shape = input_ids.size()
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")

		device = input_ids.device if input_ids is not None else inputs_embeds.device

		if attention_mask is None:
			attention_mask = torch.ones(input_shape, device=device)
		if encoder_attention_mask is None:
			encoder_attention_mask = torch.ones(input_shape, device=device)
		if token_type_ids is None:
			token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

		if attention_mask.dim() == 3:
			extended_attention_mask = attention_mask[:, None, :, :]

		if attention_mask.dim() == 2:
			if self.config.is_decoder:
				batch_size, seq_length = input_shape
				seq_ids = torch.arange(seq_length, device=device)
				causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
				extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
			else:
				extended_attention_mask = attention_mask[:, None, None, :]

		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


		if encoder_attention_mask.dim() == 3:
			encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
		if encoder_attention_mask.dim() == 2:
			encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

		encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
		encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

		if head_mask is not None:
			if head_mask.dim() == 1:
				head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
				head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
			elif head_mask.dim() == 2:
				head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
			head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
		else:
			head_mask = [None] * self.config.num_hidden_layers

		embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
		encoder_outputs, bn_hidden_tuple = self.encoder(embedding_output,
									   attention_mask=extended_attention_mask,
									   head_mask=head_mask,
									   encoder_hidden_states=encoder_hidden_states,
									   encoder_attention_mask=encoder_extended_attention_mask,
									   params_dict_list=params_dict_list,
									   return_bn_hidden=return_bn_hidden)
		sequence_output = encoder_outputs[0]
		pooled_output = self.pooler(sequence_output)

		outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
		outputs += (bn_hidden_tuple,)
		return outputs   # sequence_output, pooled_output, (hidden_states), (attentions) : outputs



class BertWithAdapterSQ(BertForSequenceClassification):
	def __init__(self, config):
		super().__init__(config)
		self.bert = BertWithAdapter(config)

		for n, p in self.bert.named_parameters():
			if not 'adapter' in n:
				p.requires_grad = False
		self.init_weights()

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
				position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
				params_dict_list=None):

		outputs = self.bert(input_ids,
							attention_mask=attention_mask,
							token_type_ids=token_type_ids,
							position_ids=position_ids,
							head_mask=head_mask,
							inputs_embeds=inputs_embeds,
							params_dict_list=params_dict_list)

		pooled_output = outputs[1]

		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

		if labels is not None:
			  loss_fct = CrossEntropyLoss()
			  loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs

		return outputs  # output : (loss), logits, (hidden_states), (attentions)


class RobertaWithAdapterSQ(RobertaForSequenceClassification):
	def __init__(self, config):
		super().__init__(config)
		self.bert = RobertaWithAdapter(config)

		for n, p in self.bert.named_parameters():
			if not 'adapter' in n:
				p.requires_grad = False
		self.init_weights()

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
				position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
				params_dict_list=None):

		outputs = self.bert(input_ids,
							attention_mask=attention_mask,
							token_type_ids=token_type_ids,
							position_ids=position_ids,
							head_mask=head_mask,
							inputs_embeds=inputs_embeds,
							params_dict_list=params_dict_list)

		pooled_output = outputs[1]

		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		outputs = (logits,) + outputs[2:]  

		if labels is not None:
			  loss_fct = CrossEntropyLoss()
			  loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs

		return outputs 


class TrollDomainBuildingNetwork(nn.Module):
	"""
	A simple classifer adaptation network built up for each troll domain.
	The weights for classifier are initialized as the weights/bias of all troll domains.
	"""
	def __init__(self):
		super().__init__()
		pass

	def forward(self, features, domains):
		""" 
		Args:
			representation_dict (dict<torch.tensors>): Dictionary containing domain-level representations for each troll domain

		Returns:
			dict<torch.tensors>: Dictionary containing the weights and biases for the classification of each class
				 in the task. Model can extract parameters and build the classifier accordingly. Supports sampling if
				 ML-PIP objective is desired.
		"""
		class_params = []

		for d in torch.unique(domains, sorted=True):
			# filter out feature vectors which have domain d
			class_features = torch.index_select(features,
												0,
												extract_indices(domains, d))
			class_params.append(mean_pooling(class_features))

		prototypes = torch.cat(class_params, dim=0)

		return prototypes
