import argparse
import copy
import os

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from transformers import AutoTokenizer
from evaluator import m6a_evaluator
from dataset import m6a_dataset
from model.BERT_like import RNATokenizer
from model.RNAErnie.tokenization_rnaernie import RNAErnieTokenizer
import torch

MAX_SEQ_LEN = {"RNABERT": 440,
               "RNAMSM": 512,
               "RNAFM": 512,
               "DNABERT": 512,
               "DNABERT2": 512,
               "SpliceBERT": 512,
               "RNAErnie": 512,
               "NucleotideTransformer": 100,
               "GENA-LM-base": 512//4,
               "GENA-LM-large": 512//4,
               "UTRLM": 512,
               "DeepM6A": 101,
               "bCNNMethylpred": 101,
               }
available_methods = MAX_SEQ_LEN.keys()


def str2list(v):
    if isinstance(v, list):
        return v
    elif isinstance(v, str):
        vs = v.split(",")
        return [v.strip() for v in vs]
    else:
        raise argparse.ArgumentTypeError(
            "Str value seperated by ', ' expected.")


def build_evaluator(args):
    if args.method == 'RNAFM':
        args.replace_T = True
        args.replace_U = False
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        tokenizer = RNATokenizer(args.vocab_path)
        return m6a_evaluator.RNAFMEvaluator(args, tokenizer=tokenizer)

    if args.method == 'RNAMSM':
        args.max_seq_len = MAX_SEQ_LEN["RNAMSM"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)
        return m6a_evaluator.RNAMsmEvaluator(args, tokenizer=tokenizer)

    if args.method == 'RNABERT' or args.method == 'RNABERT_RAW':
        args.max_seq_len = MAX_SEQ_LEN["RNABERT"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)
        return m6a_evaluator.RNABertEvaluator(args, tokenizer=tokenizer)

    if (args.method == 'SpliceBERT') or (args.method == 'DNABERT'):
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        return m6a_evaluator.DNABERTEvaluator(args, tokenizer=tokenizer)

    if args.method == 'DNABERT2':
        args.max_seq_len = MAX_SEQ_LEN["DNABERT2"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        return m6a_evaluator.DNABERT2Evaluator(args, tokenizer=tokenizer)

    if args.method == 'RNAErnie':
        args.max_seq_len = MAX_SEQ_LEN["RNAErnie"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNAErnieTokenizer.from_pretrained(args.model_path)
        return m6a_evaluator.RNAErnieEvaluator(args, tokenizer=tokenizer)

    if args.method == 'NucleotideTransformer':
        args.max_seq_len = MAX_SEQ_LEN["NucleotideTransformer"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        return m6a_evaluator.NTEvaluator(args, tokenizer=tokenizer)

    if args.method == 'GENA-LM-base' or args.method == 'GENA-LM-large':
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        return m6a_evaluator.GENAEvaluator(args, tokenizer=tokenizer)

    if args.method == 'DeepM6A':
        args.replace_T = False
        args.replace_U = True
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        return m6a_evaluator.DeepM6ASeqEvaluator(args, tokenizer=None)

    if args.method == 'bCNNMethylpred':
        args.replace_T = False
        args.replace_U = True
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        return m6a_evaluator.bCNNEvaluator(args, tokenizer=None)

    if args.method == 'UTRLM':
        args.max_seq_len = MAX_SEQ_LEN["UTRLM"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = RNATokenizer(args.vocab_path)
        return m6a_evaluator.UTRLMEvaluator(args, tokenizer=tokenizer)

    raise ValueError(f"Unsupported method: {args.method}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--method", choices=available_methods, default='RNABERT')
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--class_num", default=3, type=int)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--output_dir", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=1, type=int)
    parser.add_argument("--pad_token_id", default=0, type=int)
    parser.add_argument("--dataset_train", type=str)
    parser.add_argument("--dataset_test", type=str)
    parser.add_argument("--extra_eval", type=str, default=None)
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc,pr_auc,auc",)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--num_folds", default=5, type=int)

    args = parser.parse_args()
    assert args.output_dir, "output_dir is required."

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if ('101bp' in args.dataset_train) and (MAX_SEQ_LEN[args.method] > 103):
        MAX_SEQ_LEN[args.method] = 103
    if ('101bp' in args.dataset_train) and (args.method == 'NucleotideTransformer'):
        MAX_SEQ_LEN[args.method] = 25

    dataset_train = m6a_dataset.M6ADataset(fasta_dir=args.dataset_train)
    dataset_test = m6a_dataset.M6ADataset(fasta_dir=args.dataset_test)

    labels = [item[1] for item in dataset_train.data]
    splitter = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(dataset_train.data, labels), start=1):
        print(f"\n========== Fold {fold_id}/{args.num_folds} ==========")
        fold_train = Subset(dataset_train, train_idx.tolist())
        fold_val = Subset(dataset_train, val_idx.tolist())

        fold_args = copy.deepcopy(args)
        fold_args.output_dir = os.path.join(args.output_dir, f"fold_{fold_id}")

        evaluator = build_evaluator(fold_args)
        evaluator.run(
            fold_args,
            train_data=fold_train,
            eval_data=fold_val,
            extra_eval_data=dataset_test,
            extra_eval_name="Independent_test_set",
        )
