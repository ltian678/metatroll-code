# -*- coding: utf-8 -*-
"""

We use the adapter-transformers to simply train a language model + an adapter 

"""

"""## Dataset Preprocessing

Before we start to train our adapter, we first prepare the training data. Our training dataset can be loaded via HuggingFace `datasets` using one line of code:
"""

import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertConfig, BertModelWithHeads
import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

def data_preprocess(input_file, target_domain):
    df = pd.read_pickle(input_file)
    df.rename(columns = {'encoded_label': 'labels'}, inplace=True)
    target_df = df[df["troll_domain"] == target_domain]
    target_df = target_df.sample(frac=1)
    target_df_troll = target_df[target_df['label']=='Troll']
    target_df_nontroll = target_df[target_df['label']=='Non-Troll']

    train_vali_troll_df = target_df_troll.iloc[:10]
    train_vali_nontroll_df = target_df_nontroll.iloc[:10]
    test_troll_df = target_df_troll.iloc[10:]
    test_nontroll_df = target_df_nontroll.iloc[10:]

    target_df_train_vali = pd.concat([train_vali_troll_df,train_vali_nontroll_df])
    train_df, vali_df = train_test_split(target_df_train_vali, test_size=0.5)
    test_df = pd.concat([test_troll_df, test_nontroll_df])
    test_df = test_df.sample(frac=1)

    #train_df, test = train_test_split(target_df, test_size=0.2)
    #vali_df, test_df = train_test_split(test, test_size=0.8)
    target_df.shape, train_df.shape, vali_df.shape, test_df.shape

    train_dataset = Dataset.from_pandas(train_df)
    vali_dataset = Dataset.from_pandas(vali_df)
    test_dataset = Dataset.from_pandas(test_df)


    ds = DatasetDict()

    ds['train'] = train_dataset
    ds['validation'] = vali_dataset
    ds['test'] = test_dataset

    return ds

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["clean_tweets_string"], max_length=240, truncation=True, padding="max_length")

def adapter_run(input_file, target_domain):
    ds = data_preprocess(input_file, target_domain)
    tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/AdapertBERT_pretrain_v2/vocab/")
    # Encode the input data
    dataset = ds.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    #dataset.rename_column("encoded_label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    config = BertConfig.from_pretrained(
    "/content/drive/MyDrive/AdapertBERT_pretrain_v2/model/",
        num_labels=2,
    )
    model = BertModelWithHeads.from_pretrained(
        "/content/drive/MyDrive/AdapertBERT_pretrain_v2/model/",
        config=config,
    )
    # Add a new adapter
    model.add_adapter("troll_adapter")
    # Add a matching classification head
    model.add_classification_head(
        "troll_adapter",
        num_labels=2,
        id2label={ 0: "non-troll", 1: "troll"}
      )
    # Activate the adapter
    model.train_adapter("troll_adapter")

    training_args = TrainingArguments(
        learning_rate=2e-5,
        num_train_epochs=6,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=200,
        output_dir="./training_output",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )



    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_accuracy,
    )

    """Start the training 🚀"""

    trainer.train()

    """Looks good! Let's evaluate our adapter on the validation split of the dataset to see how well it learned:"""

    trainer.evaluate()

    eval_trainer = AdapterTrainer(
        model=model,
        args=TrainingArguments(output_dir="./test_output", remove_unused_columns=False,),
        eval_dataset=dataset["test"],
        compute_metrics=compute_accuracy,
    )
    eval_trainer.evaluate()

    print('target_domain',target_domain)


class AdapterTrainingArgs:
    def __init__(self):
        self.input_data_file = './data/troll_source.pkl'
        self.target_domain = 'CHNU'




if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = AdapterTrainingArgs()

    adapter_run(args.input_data_file, args.target_domain)





