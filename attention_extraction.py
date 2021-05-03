import argparse
import numpy as np
from tqdm import tqdm
from utils.helpers import read_lines
from gector.gec_model import GecBERTModel
import utils.adversarial as adv


def perturbations_for_file(input_file: str, label_file: str, model: GecBERTModel, batch_size: int = 32):
    test_data = read_lines(input_file)
    lengths = []
    batch_extracted_words = []
    batch = []
    # Extract attention scores for inputs.
    for sent in tqdm(test_data):
        split = sent.split()
        lengths.append(len(split))
        batch.append(split)
        if len(batch) == batch_size:
            extracted_words = model.extract_candidate_words(batch)
            batch_extracted_words.extend(extracted_words)
            batch = []
    if batch:
        extracted_words = model.extract_candidate_words(batch)
        batch_extracted_words.extend(extracted_words)
    lengths = np.array(lengths)
    labels = read_lines(label_file)
    # Generate perturbed outputs.
    perturbations, perturbation_labels = [], []
    indices = np.where(lengths >= model.min_len)[0]
    for j, i in tqdm(enumerate(indices), total=indices.size):
        perturbation, label = adv.find_word_perturbation(test_data[i], labels[i], batch_extracted_words[j])
        # If perturbation equals input, ignore it.
        if perturbation == test_data[i]:
            continue
        perturbations.append(perturbation)
        perturbation_labels.append(label)
    # Each item in perturbed_batch is of form (perturbed_input, label).
    return perturbations, perturbation_labels


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
                         weigths=args.weights)
    perturbations, perturbation_labels = perturbations_for_file(args.input_file, args.output_file, model,
                                                                batch_size=args.batch_size)
    # Write to file.
    with open("perturbed_inputs.txt", "w") as f:
        for perturbation in perturbations:
            f.write(perturbation + "\n")
    with open("perturbation_labels.text", "w") as f:
        for label in perturbation_labels:
            f.write(label + "\n")


if __name__ == '__main__':

    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.',
                        default='models/bert_0_gector.th',
                        nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        default='test_cases/input_test.txt',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        default='test_cases/output_test.txt',
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
    args = parser.parse_args()
    main(args)
