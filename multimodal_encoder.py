import easyocr


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

lang_dict= {
	'chn':'ch_sim',
	'fr': 'fr',
	'thai': 'th',
	'eng': 'en',
	'es': 'es',
}


def convert_lang_lst(langs):
	res = []
	for lang in langs:
		res.append(lang_dict[lang])
	return res

def read_img(langs, domain, img_file):
	reader = easyocr.Reader(convert_lang_lst(langs))
	result = reader.readtext(img_file)
	return result

#for simpler output
def read_img_simple(langs, domain, img_file):
	reader = easyocr.Reader(convert_lang_lst(langs))
	simple_res = reader.readtext(img_file, detail=0)
	return simple_res

def emb_img(img_file):
	imgemb = ImgEmb()
	img = Image.open(img_file).convert('RGB')
	i_emb = imgemb.get_vec(img)
	return i_emb


def get_combo_img_rep(langs, domain, img_file):
	ocr_emb = read_img(langs, domain, img_file)
	i_emb = emb_img(img_file)
	return ocr_emb, i_emb

class ImgEmb():
	def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512, gpu=0):
		self.device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
		self.layer_output_size = layer_output_size
		self.model_name = model

		self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

		self.model = self.model.to(self.device)

		self.model.eval()

		self.scaler = transforms.Resize((224, 224))
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		self.to_tensor = transforms.ToTensor()

	def get_emb(self, img, tensor=False):
		if type(img) == list:
			a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
			images = torch.stack(a).to(self.device)
			my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

			def copy_data(m, i, o):
				my_embedding.copy_(o.data)

			h = self.extraction_layer.register_forward_hook(copy_data)
			with torch.no_grad():
				h_x = self.model(images)
			h.remove()

			if tensor:
				return my_embedding
			else:
				return my_embedding.numpy()[:, :, 0, 0]
		else:
			image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)
			my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

			def copy_data(m, i, o):
				my_embedding.copy_(o.data)

			h = self.extraction_layer.register_forward_hook(copy_data)
			with torch.no_grad():
				h_x = self.model(image)
			h.remove()

			if tensor:
				return my_embedding
			else:
				return my_embedding.numpy()[0, :, 0, 0]

	def _get_model_and_layer(self, model_name, layer):
		model = models.resnet18(pretrained=True)
		if layer == 'default':
			layer = model._modules.get('avgpool')
			self.layer_output_size = 512
		else:
			layer = model._modules.get(layer)

		return model, layer