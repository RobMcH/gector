import argparse
import numpy as np
from tqdm import tqdm
from utils.helpers import read_lines, analyse_rules
from gector.gec_model import GecBERTModel
import utils.adversarial as adv
from collections import Counter


def attention_for_file(input_file: str, model: GecBERTModel, acc: str, agg: str, batch_size: int = 32,
                       indices_per_sent: int = 1, attention_sorting: str = "max"):
    data = read_lines(input_file)
    lengths = []
    batch_extracted_words = []
    batch = []
    min_scores = True if attention_sorting == "min" else False
    # Extract attention scores for inputs.
    for sent in tqdm(data):
        split = sent.split()
        lengths.append(len(split))
        batch.append(split)
        if len(batch) == batch_size:
            extracted_words = model.extract_candidate_words(batch, aggregation=acc, head_aggregation=agg,
                                                            n=indices_per_sent, min_scores=min_scores)
            batch_extracted_words.extend(extracted_words)
            batch = []
    if batch:
        extracted_words = model.extract_candidate_words(batch, aggregation=acc, head_aggregation=agg,
                                                        n=indices_per_sent, min_scores=min_scores)
        batch_extracted_words.extend(extracted_words)
    return np.array(lengths), batch_extracted_words


def attention_perturbations(input_file: str, label_file: str, model: GecBERTModel, acc: str, agg: str,
                            batch_size: int = 32, perturbations_per_sent: int = 1, attention_sorting: str = "max"):
    # Perturb sentences according to rules by choosing a vulnerable token identified by attention scores.
    data = read_lines(input_file, skip_strip=True)
    labels = read_lines(label_file, skip_strip=True)
    # Get vulnerable tokens by attention scores.
    lengths, batch_extracted_words = attention_for_file(input_file, model, acc, agg, batch_size, perturbations_per_sent,
                                                        attention_sorting)
    # Generate perturbed outputs.
    perturbations, perturbation_labels = [], []
    indices = np.where(lengths >= model.min_len)[0]
    pos_counter = Counter()
    for j, i in tqdm(enumerate(indices), total=indices.size):
        perturbations_temp, labels_temp, pos, _ = adv.find_word_perturbation(data[i], labels[i], batch_extracted_words[j],
                                                                          perturbations_per_sent)
        for k in range(len(perturbations_temp)):
            perturbation, label = perturbations_temp[k], labels_temp[k]
            if len(perturbation.strip()) == 0 or perturbation == data[i]:
                # Ignore empty or duplicate perturbations.
                continue
            pos_counter[pos[k]] += 1
            perturbations.append(perturbation)
            perturbation_labels.append(label)
    # Each item in perturbed_batch is of form (perturbed_input, label).
    print(analyse_rules(pos_counter))
    return perturbations, perturbation_labels


def random_perturbations(input_file: str, label_file: str, perturbations_per_sent: int = 1):
    # Perturb sentences according to rules by choosing a random token.
    data = read_lines(input_file, skip_strip=True)
    labels = read_lines(label_file, skip_strip=True)
    perturbations, perturbation_labels = [], []
    pos_counter = Counter()
    for i, sent in tqdm(enumerate(data), total=len(data)):
        perturbations_temp, labels_temp, pos, _ = adv.random_perturbation(sent, labels[i], perturbations_per_sent)
        for k in range(len(perturbations_temp)):
            perturbation, label = perturbations_temp[k], labels_temp[k]
            if len(perturbation.strip()) == 0 or perturbation == data[i] or perturbation == label:
                # Ignore empty or duplicate perturbations.
                continue
            pos_counter[pos[k]] += 1
            perturbations.append(perturbation)
            perturbation_labels.append(label)
    print(analyse_rules(pos_counter))
    return perturbations, perturbation_labels


def main(args):
    if args.attack == 'attention':
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
        perturbations, perturbation_labels = attention_perturbations(args.input_file, args.output_file, model,
                                                                     args.accumulator, args.head_aggregator,
                                                                     args.batch_size, args.num_perturbations,
                                                                     args.attention_sorting)
    elif args.attack == 'random':
        perturbations, perturbation_labels = random_perturbations(args.input_file, args.output_file,
                                                                  args.num_perturbations)
    # Write to file.
    with open(f"{args.attack}_{args.accumulator}_{args.head_aggregator}_perturbed_inputs.txt", "w") as f:
        for perturbation in perturbations:
            f.write(perturbation + "\n")
    with open(f"{args.attack}_{args.accumulator}_{args.head_aggregator}_perturbation_labels.txt", "w") as f:
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
    parser.add_argument('--label_file',
                        help='Path to the label file',
                        default='test_cases/label_test.txt',
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
    parser.add_argument('--attack',
                        help='Define the attack mode.',
                        choices=['attention', 'random'],
                        default='attention')
    parser.add_argument('--attention_sorting',
                        help='Whether to use the min or max attention scores. Only relevant for attack=attention.',
                        choices=['min', 'max'],
                        default='max')
    parser.add_argument('--accumulator',
                        help='Define how to accumulate attention scores.',
                        choices=['sum', 'max', 'mean'],
                        default='sum')
    parser.add_argument('--head_aggregator',
                        help='How to aggregate the attention heads.',
                        choices=['sum', 'prod'],
                        default='sum')
    parser.add_argument('--num_perturbations',
                        help='The maximum number of perturbations per sentence.',
                        type=int,
                        default=1)
    args = parser.parse_args()
    main(args)
