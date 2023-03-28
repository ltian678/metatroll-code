import torch
from torch import nn
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
from utils import extract_indices, mean_pooling
from model import BERTTrollClassifierSQ, PTBERTTrollClassifierSQ, BertWithAdapterSQ, RobertaWithAdapterSQ, BertWithAdapter, RobertaWithAdapter, init_pretrain_model, init_model,TrollDomainBuildingNetwork


class AdaptiveLinearClassifier(nn.Module):
	"""
	Adaptive linear classifier in MetaTroll
	"""
	def __init__(self, input_size, output_size):
		super().__init__()
		self.processor = self._dense_block(input_size, output_size+1)

	def _dense_block(in_size, out_size):
		return nn.Sequential(nn.Linear(in_size, out_size),nn.Tanh(),nn.Linear(out_size, out_size))

	def _extract_domain_indices(self, labels, domain):
		domain_mask = torch.eq(labels, domain)
		domain_mask_indices = torch.nonzero(domain_mask, as_tuple=False)  
		return torch.reshape(domain_mask_indices, (-1,))

	def forward(self, features, labels):        class_params = []
		for c in torch.unique(labels, sorted=True):
			# filter out feature vectors for each domain
			domain_features = torch.index_select(features,0,self._extract_class_indices(labels, c))
			domain_params.append(mean_pooling(domain_features))

		domain_params = torch.cat(domain_params, dim=0)

		classifier_param_dict = {}
		classifier_param_dict['weight_mean'] = class_params[:, :-1]
		classifier_param_dict['bias_mean'] = class_params[:, -1]

		return classifier_param_dict



class MetaDomainSpecificAdapterALC(nn.Module):
	bert_arg_list = ['input_ids', 'attention_mask', 'token_type_ids']
	batch_arg_list = bert_arg_list + ['labels']

	def __init__(self, args, bert_name, bert_config):
		super().__init__()
		self.args = args
		self.pretrained_bert_name = bert_name
		self.bert_config = bert_config
		self.bert = self.init_bert()
		self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
		# linear layer after BERT
		self.linear = nn.Linear(bert_config.hidden_size, self.args.bert_linear_size)

		self.setup_grad()

		self.domain_build = TrollDomainBuildingNetwork()
		# self.cos = nn.CosineSimilarity(dim=-1)
		self.dist_metric = self.init_dist_metric()
		# Task embedding model
		self.task_emb_model = self.init_task_emb_model()
		# Adaptation network 
		self.adapt_module_dict = self.init_adapt_module_dict()


	def init_task_emb_model(self):
		if self.args.load_pt:
			self.emb_model = init_pretrain_model(self.args.base_model,self.args.pt_base_model_path, self.args.pt_tokenizer_path)
		else:
			self.emb_model = init_model(self.args.base_model)
		raise NotImplementedError()

	def init_adapt_module_dict(self):
		if self.args.load_pt:
			self.model = BertWithAdapterSQ(self.args.model_base_config,
									 pt_encoder_state_dict=torch.load(self.args.base_pretrained)['model_state_dict'])
		else:
			self.model = BertWithAdapterSQ(self.args.model_base_config)

	def init_dist_metric(self):
		return EuclideanDist(dim=-1)

	def init_bert(self):
		return self.args.bert_class.from_pretrained(
						self.args.pretrained_bert_name,
						from_tf=False,
						config=self.args.model_base_config,
						cache_dir=self.args.cache_dir if self.args.cache_dir else None)

	def setup_grad(self):
		if self.args.freeze_base_model:
			# Freenze the BERT model with linear.
			for p in self.bert.parameters():
				p.requires_grad = False
		else:
			# Only bottleneck adapters and last linear layer are trainable.
			for n, p in self.bert.named_parameters():
				if not 'adapter' in n:
					p.requires_grad = False

		if self.args.freeze_linear_layer:
			# Whether to freeze the last linear layer.
			for p in self.linear.parameters():
				p.requires_grad = False

	def _text_encoding_by_batch(self, task, add_arg_names, add_arg_values, batch_size, cpu=False):
		features_list = []
		total_num = task['input_ids'].shape[0]
		for i in range(0, total_num, batch_size):
			features = self._text_encoding(task, start=i, end=i+batch_size,
										   add_arg_names = add_arg_names,
										   add_arg_values = add_arg_values)
			features = features.cpu() if cpu else features
			features_list.append(features)
		return torch.cat(features_list)

	def _text_encoding(self, batch, start=None, end=None, add_arg_names=None, add_arg_values=None):
		start = 0 if start is None else start
		end = batch['input_ids'].shape[0] if end is None else end
		if not (start >=0 and start < end):
			raise ValueError(f'Invalid start and end value: start {start}, end {end}')
		
		# Text encoding using BERT
		args = {k:v[start:end] for k,v in batch.items() if k in self.bert_arg_list}
		if add_arg_names is not None and add_arg_values is not None:
			assert len(add_arg_values) == len(add_arg_names)
			for n, v in zip(add_arg_names, add_arg_values):
				args[n] = v
		embs = self.bert(**args)[1]
		embs = self.dropout(embs)
		embs = self.linear(embs)
		return embs

	def domain_loss_acc(self, spt_features, spt_labels, qry_features, qry_labels):
		# num_classes * hidden_size
		prototypes = self.domain_build(spt_features, spt_labels)
		num_classes = spt_labels.unique().shape[0]
		assert num_classes == prototypes.shape[0]
		num_query = qry_features.shape[0]
		# num_query * num_classes
		qry_logits = self.dist_metric(
			qry_features.unsqueeze(1).expand(-1, num_classes, -1),  # num_query * num_classes * hidden_size
			prototypes.expand(num_query, -1, -1))                   # num_query * num_classes * hidden_size

		loss = self.loss(qry_logits, qry_labels)
		acc = self.accuracy_fn(qry_logits, qry_labels)
		return loss, acc

	def _task_encoding(self, task, num_steps=5):
		for n, p in self.bert.named_parameters():
			if 'adapter' in n:
				p.requires_grad = True
				p.grad_accumu = torch.zeros_like(p.data)
				p.grad_accumu_count = 0
		
		num_classes = task['labels'].unique().shape[0]
		num_examples = task['input_ids'].shape[0]
		num_support = num_examples // 2
	   
		for i in range(num_steps):
			embs = self.bert(**task)[1]
			# embs = self.dropout(embs)
			embs = self.linear(embs)

			prototypes = self.domain_build(embs[:num_support],task['labels'][:num_support])
			assert num_classes == prototypes.shape[0]
			query_embs = embs[num_support:]
			num_query = query_embs.shape[0]
			query_logits = self.dist_metric(query_embs.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
									prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size

			target = torch.multinomial(F.softmax(query_logits, dim=-1), 1).detach().view(-1)
			loss = self.loss(query_logits, target)
			# loss = self.loss(query_logits, task['labels'][num_support:])
			# acc = self.accuracy_fn(query_logits, task['labels'][num_support:])
			self.bert.zero_grad()
			loss.backward()
			for n, p in self.bert.named_parameters():
				if 'adapter' in n:
					assert p.grad is not None
					p.grad_accumu += p.grad.data ** 2 
					p.grad_accumu_count += 1
		
		for n,p in self.bert.named_parameters():
			if 'adapter' in n:
				p.grad_accumu /= p.grad_accumu_count
				p.requires_grad = False
		
		all_grads = [] 
		for n, m in self.bert.named_modules():
			if hasattr(m, 'weight') and hasattr(m.weight, 'grad_accumu'):
				grad = m.weight.grad_accumu
				all_grads.append(grad.reshape(-1))
		all_grads = torch.stack(all_grads) # 48 * 12288
		all_grads = all_grads.reshape(self.bert.config.num_hidden_layers, -1) # 12 * (4 * 12288)
		all_grads = all_grads.unsqueeze(0)
		self.task_emb_model.flatten_parameters()
		task_emb = self.task_emb_model(all_grads)[0][0]
		return task_emb

	def forward(self, batch, eval=False):
		device = self.get_device()
		device_name = self.get_device_name()
		if hasattr(batch, device_name):
			local_batch = getattr(batch, device_name)
			loss = 0
			query_acc = []

			for task in local_batch:
				if not eval:
					t_loss, t_acc = self.forward_task(task)
				else:
					t_loss, t_acc = self.eval_task(task)
				loss += t_loss
				query_acc.append(t_acc)
		   
			query_acc = torch.stack(query_acc)
			return loss, query_acc.detach()
		else:
			assert eval # this could happen only during evaluation with multiple gpu
			loss, query_acc = torch.tensor(0.).to(device), torch.tensor([-1.]).to(device)
			return loss, query_acc

	def forward_task(self, task):
		task_labels = task['labels']
		num_classes = task_labels.unique().shape[0]
		num_shots = self.args.num_shots_support
		num_support = num_classes * num_shots

		adapt_arg_names, adapt_arg_values = self.gen_adapt_param(task, num_support,
																 batch_size=num_support,
																 detach=False)
		task_features = self._text_encoding(task, 
											add_arg_names = adapt_arg_names,
											add_arg_values = adapt_arg_values)

		# Prototypical loss.
		if not self.args.use_improve_loss:
			loss, acc = self.domain_loss_acc(task_features[:num_support],
											task_labels[:num_support],
											task_features[num_support:],
											task_labels[num_support:])
		else:
			loss_after_adp, acc_after_adp = self.domain_loss_acc(task_features[:num_support],
											task_labels[:num_support],
											task_features[num_support:],
											task_labels[num_support:])

			task_features = self._text_encoding(task)
			loss_before_adp, acc_before_adp = self.domain_loss_acc(task_features[:num_support],
																	task_labels[:num_support],
																	task_features[num_support:],
																	task_labels[num_support:])
			loss = F.relu(loss_after_adp - loss_before_adp + 1.0)
			import ipdb; ipdb.set_trace()
			acc = acc_after_adp
			
		return loss, acc

	def gen_adapt_param(self, task, num_support, batch_size, detach=False):
		if self.args.cnap_adapt:
			task_emb_sum = None
			task_emb_num = 0
			for ind in range(0, num_support, batch_size):
				task_emb = self._task_encoding({
					k:v[ind:ind+batch_size] if k in self.batch_arg_list else v for k,v in task.items()})
				task_emb_num += 1
				task_emb_sum = task_emb if task_emb_sum is None else task_emb_sum + task_emb
			task_emb = task_emb_sum / task_emb_num 
			if detach:
				task_emb = task_emb.detach()
			adapt_param_list = self._shift_scale_params(task_emb)
			return ['params_dict_list'], [adapt_param_list]
		else:
			return None, None

	def get_adapt_param(self, detach=False):
		params = []
		if self.args.adapt_mode:
			for n, p in self.bert.named_parameters():
				if not 'adapter' in n:
					params.append(p)
			return params
		else:
			return None

	def eval_task(self, task):
		device = self.get_device()
		task_labels = task['labels']
		if not 'num_classes' in task:
			num_classes = task_labels.unique().shape[0]
			num_shots = self.args.num_shots_support
		else:
			num_classes = task['num_classes'].item()
			num_shots = task['num_shots'].item()
		num_support = num_classes * num_shots
		num_query = task_labels.shape[0] - num_support

		adapt_arg_names, adapt_arg_values = self.gen_adapt_param(task, num_support,
																 num_support if num_shots <= 6 else 4*num_classes,
																 detach=True)
		with torch.no_grad():
			support_features = self._text_encoding_by_batch(
				{k:v[:num_support] if k in self.bert_arg_list else v for k,v in task.items()},
				adapt_arg_names,
				adapt_arg_values,
				num_support if num_shots <= 6 else 100,
				cpu=True
			)
			query_features = self._text_encoding_by_batch(
				{k:v[num_support:] if k in self.bert_arg_list else v for k,v in task.items()},
				adapt_arg_names,
				adapt_arg_values,
				min(100, num_query),
				cpu=True
			)

			loss, acc = self.domain_loss_acc(support_features,
											   task_labels[:num_support].cpu(),
											   query_features,
											   task_labels[num_support:].cpu())
			return loss.to(device), acc.to(device)


class TrainingArgs_Stage3:
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



def main():
	# Set the seed value all over the place to make this reproducible.
	seed_val = 42

	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)
	# Arg parser initialization and parsing TODO: need to filling in the int_arg_parser here!!!!
	args = TrainingArgs_Stage3()

	data = gen_stage1_dataset(args.raw_data_file)

	meta_adapter = MetaDomainSpecificAdapterALC(args,tokenizer, model_classifier)
	meta_adapter.train()

if __name__ == "__main__":
	main()

