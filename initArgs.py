import argparse

#parser = init_arg_parser(MetaLearner)



def init_arg_parser(desc):
    parser = argparse.ArgumentParser(description=desc)

    ## Required parameters #WIP NEED TO UPDATE THIS LIST
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", required=True, nargs="?",
                        choices=list(learner.MODEL_CLASSES.keys()),
                        help="Model type.")
    parser.add_argument("--pretrained_model_dir", default=None, type=str, required=True, help="The pretrained model dir.")

    ## Experiment parameters -- common and shared parameters
    parser.add_argument("--num_labels", default=2, type=int, required=True, help="Number of final predicted labels.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=False, help="checkpoint data path.")
    parser.add_argument("--max_seq_length", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")


    ## Stage I Hypeparameters
    parser.add_argument("--learning_rate_s1", default=5e-5, type=float, help="The initial learning rate for Adam for MetaTroll Stage I.")
    parser.add_argument("--num_epoch_s1", default=5, type=int, help="The number of epochs for MetaTroll Stage I.")
    parser.add_argument("--num_warmup_steps_s1", default=10, type=int, help="The number of warmup steps for MetaTroll Stage I.")
    parser.add_argument("--batch_size_s1_train", default=32, type=int, help="The training batch size for MetaTroll Stage I.")
    parser.add_argument("--batch_size_s1_test", default=16, type=int, help="The test batch size for MetaTroll Stage I.")

    
    ## Meta learning parameters


    parser.add_argument("--num_shots_support", default=5, type=int, help="Number of support")
    parser.add_argument("--num_shots_query", default=5, type=int, help="Number of query")


    ## State II Hyperparameters



    parser.add_argument("--slurm_job_id", default=None, type=str, required=False,
                        help="Slurm job id.")
    parser.add_argument("--exp_id", default=None, type=str, required=False,
                        help="Exp id.")
    parser.add_argument("--lm_type", nargs="?", default='bert-base-uncased',
                        choices=ALL_MODELS,
                        help="Pretrained language model type.")
    parser.add_argument("--checkpoint_freq", type=int, default=1000,
                        help="Number of iterations between validations.")
    parser.add_argument("--start_from_scratch", action='store_true',
                        help="Training from scratch.")
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="Path to checkpoint to start from.")
    parser.add_argument("--load_from_current_best", action='store_true',
                        help="Loading from previous best checkpoint.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models")
    parser.add_argument("--mode", default="train", type=str,
                        help="Whether do training / testing.")
                        # choices=['train', 'test_latest', 'test_best'],
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--val_freq", type=int, default=1000,
                        help="Number of iterations between validations.")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help="Local rank for distributed training.")

    # Modeling parameters
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # Hyperparameters
    parser.add_argument("--num_shots_support", default=10, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_shots_query", default=10, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_episodes_per_device", default=1, type=int,
                        help="Number of parallel tasks per GPU/CPU during training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_training_iterations", type=int, default=110000,
                        help="Number of meta-training iterations.")
    parser.add_argument("--num_training_epochs", type=int, default=5,
                        help="Number of meta-training epochs.")
    parser.add_argument("--num_support_batches", default=8, type=int,
                        help="Number of support batches in each episode.")
    parser.add_argument("--num_query_batches", default=1, type=int,
                        help="Number of query batches in each episode.")
    parser.add_argument("--num_iterations_per_optimize_step", default=5, type=int,
                        help="Number of tasks between parameter optimizations.")
    parser.add_argument("--max_num_val_episodes", default=100, type=int,
                        help="Number of episode for evaluation of each dataset.")
    parser.add_argument("--warmup_steps_ratio", default=0, type=float,
                        help="Linear warmup over warmup_steps_ratio*total_steps.")
    parser.add_argument("--bert_linear_size", default=768, type=int,
                        help="Size of the linear layer after BERT encoder.")
    parser.add_argument("--adapter_hidden_size", default=16, type=int,
                        help="Hidden size of the bottleneck adapter network.")
    parser.add_argument("--task_emb_size", default=100, type=int,
                        help="Size of per-layer task embedding.")
    parser.add_argument("--bn_context_size", default=100, type=int, help="Size of context vector.")
    parser.add_argument("--classifier_hidden_size", default=200, type=int, help="Size of hidden unit in classifier.")
    parser.add_argument("--freeze_base_model", default=False, type=lambda x: (str(x).lower() == 'true'), help="Freeze the base BERT model of MetaTroll, only train the adaptation network.")
    parser.add_argument("--freeze_linear_layer", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to freeze the last linear layer in second stage training.")
    parser.add_argument('--model_adapt', default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether adapting the base model using an adaptation network.")
    parser.add_argument('--model_pretrained', default=None, type=str, help="Pretrained BERT with bottleneck adapters.")

    parser.add_argument("--early_stop_by", default="avg", nargs="?", const="avg", choices=['vote', 'avg'], help="How to eatly stop.")
    parser.add_argument("--early_stop_patience", default=5, type=int, help="Early stop patience.")
                    
    parser.add_argument("--fine_tune_epochs", default=10, type=int, required=False, help="Fine tuning epochs for simple baseline.")

    parser.add_argument("--use_improve_loss", default=False, type=lambda x: (str(x).lower() == 'true'),help="Whether to use the loss before and after adaptation as the final loss.")

    return parser