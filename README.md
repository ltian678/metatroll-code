# MetaTroll: Few-shot Detection of State-Sponsored Trolls with Transformer Adapters

## Dataset
The troll data are download through Twitter (https://transparency.twitter.com/en/reports/moderation-research.html).


## Running Environment
This repo is built upon a local copy of `transformers==3.0.0`.
This repo has been tested on `torch==1.4.0` with `python 3.7` and `CUDA 10.1`.

To start, create a new environment and install: 
```bash
conda create -n metaTroll python=3.7
conda activate metaTroll
cd metaTroll
pip install -e .
```


## Baselines:


| `MODEL` | GITHUB |
| :---- | :---------- |
| Induct | modified based on (https://github.com/zhongyuchen/few-shot-text-classification) |
| HATT | source code from authors (https://github.com/thunlp/HATT-Proto) |
| DS | source code from authors (https://github.com/YujiaBao/Distributional-Signatures) |

# Further Pre-train & Evaluation
To further pre-train base BERT/XLM-R models:

Clone the transformer to local directory
```
git clone https://github.com/huggingface/transformers.git
```


For further pre-training the language models:
```
python transformers/examples/language-modeling/run_language_modeling.py ,
        --output_dir='BERT_DAPT',
        --model_type=bert ,
        --model_name_or_path=bert-base-cased-freeze-we,
        --do_train,
        --overwrite_output_dir,
        --train_data_file='train.txt',
        --do_eval,
        --block_size=512,
        --eval_data_file='vali.txt',
        --mlm"
```


## Publicaton
This is the source code for 
[MetaTroll: Few-shot Detection of State-Sponsored Trolls with Transformer Adapters]


If you find this code useful, please let us know and cite our paper.  
If you have any question, please contact Lin at: s3795533 at student dot rmit dot edu dot au.
