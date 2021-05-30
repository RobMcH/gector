import argparse
import pandas as pd

from utils.helpers import read_lines
from gector.gec_model import GecBERTModel


def predict_for_file(input_file, output_file, model, batch_size=32):
    test_data = read_lines(input_file)
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    with open(output_file, 'w') as f:
        f.write("\n".join([" ".join(x) for x in predictions]) + '\n')
    return cnt_corrections


def predict_for_file_mask(input_file, output_file, model, batch_size=32):
    # IMPORTANT: ensure number of iterations is set to 1
    # And min_len = 0, max_len = 1000
    test_data = read_lines(input_file)
    masked_sentences = []
    predictions = []
    error_probs_full = []
    masked_words = []
    sent_map = []
    cnt_corrections = 0
    batch = []
    for s_idx, sent in enumerate(test_data):
        sent_split = sent.split()
        for w_idx, masked_word in enumerate(sent_split):
            masked_words.append(masked_word)
            mask_sent = sent_split[:w_idx] + ['[MASK]'] + sent_split[w_idx+1:]
            masked_sentences.append(mask_sent)
            batch.append(mask_sent)
            sent_map.append(s_idx)
            if len(batch) == batch_size:
                preds, cnt, error_probs = model.handle_batch(batch, return_error_probs=True)
                error_probs_full.extend(error_probs)
                predictions.extend(preds)
                # print(len(masked_words), len(sent_map), len(error_probs_full) )
                # if len(error_probs_full) != len(sent_map):
                #     print(len(masked_words), len(sent_map), len(error_probs_full) + len(batch))
                cnt_corrections += cnt
                batch = []
        if  len(error_probs_full) + len(batch) != len(sent_map):
            print(len(masked_words), len(sent_map), len(error_probs_full) + len(batch))

    if batch:
        preds, cnt, error_probs = model.handle_batch(batch, return_error_probs=True)
        error_probs_full.extend(error_probs)
        predictions.extend(preds)
        cnt_corrections += cnt

    error_prob_df = pd.DataFrame({'words': masked_words, 'id': sent_map, 'error_probs': error_probs_full})
    min_prob_words = error_prob_df.groupby('id').min()
    min_prob_words.to_csv('test_cases/masked_results.csv')

    with open('test_cases/masked_inputs.txt', 'w') as f:
        f.write("\n".join([" ".join(x) for x in masked_sentences]) + '\n')

    with open(output_file, 'w') as f:
        f.write("\n".join([" ".join(x) for x in predictions]) + '\n')
    return cnt_corrections


def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights,
                         remove_first_layer=args.remove_first_layer)

    cnt_corrections = predict_for_file_mask(args.input_file, args.output_file, model,
                                       batch_size=args.batch_size)
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--remove_first_layer',
                        help='Remove first layer from transformer',
                        dest='remove_first_layer',
                        action='store_true')
    args = parser.parse_args()
    main(args)
