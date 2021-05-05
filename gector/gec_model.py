"""Wrapper of AllenNLP model. Fixes errors based on model predictions"""
import logging
import os
import sys
from time import time

import numpy as np

import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util

from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.seq2labels_model import Seq2Labels
from gector.wordpiece_indexer import PretrainedBertIndexer
from utils.helpers import PAD, UNK, get_target_sent_by_edits, START_TOKEN

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


def get_weights_name(transformer_name, lowercase):
    if transformer_name == 'bert' and lowercase:
        return 'bert-base-uncased'
    if transformer_name == 'bert' and not lowercase:
        return 'bert-base-cased'
    if transformer_name == 'distilbert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'distilbert-base-uncased'
    if transformer_name == 'albert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'albert-base-v1'
    if lowercase:
        print('Warning! This model was trained only on cased sentences.')
    if transformer_name == 'roberta':
        return 'roberta-base'
    if transformer_name == 'gpt2':
        return 'gpt2'
    if transformer_name == 'transformerxl':
        return 'transfo-xl-wt103'
    if transformer_name == 'xlnet':
        return 'xlnet-base-cased'


class GecBERTModel(object):
    def __init__(self, vocab_path=None, model_paths=None,
                 weigths=None,
                 max_len=50,
                 min_len=3,
                 lowercase_tokens=False,
                 log=False,
                 iterations=3,
                 model_name='roberta',
                 special_tokens_fix=1,
                 is_ensemble=True,
                 min_error_probability=0.0,
                 confidence=0,
                 resolve_cycles=False,
                 heads_to_prune=None
                 ):
        self.model_weights = list(map(float, weigths)) if weigths else [1] * len(model_paths)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.min_len = min_len
        self.lowercase_tokens = lowercase_tokens
        self.min_error_probability = min_error_probability
        self.vocab = Vocabulary.from_files(vocab_path)
        self.log = log
        self.iterations = iterations
        self.confidence = confidence
        self.resolve_cycles = resolve_cycles
        # assuming we will only be doing it for a single model for now....
        self.model_name = model_name
        self.special_tokens_fix = special_tokens_fix
        # set training parameters and operations
        self.indexers = []
        self.models = []
        #heads we would like to prune
        self.heads_to_prune = heads_to_prune
        for model_path in model_paths:
            if is_ensemble:
                model_name, special_tokens_fix = self._get_model_data(model_path)
            weights_name = get_weights_name(model_name, lowercase_tokens)
            self.indexers.append(self._get_indexer(weights_name, special_tokens_fix))
            model = Seq2Labels(vocab=self.vocab,
                               text_field_embedder=self._get_embbeder(weights_name, special_tokens_fix),
                               confidence=self.confidence
                               ).to(self.device)
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load(model_path,
                                                 map_location=torch.device('cpu')))

            if self.heads_to_prune is not None:
                model.text_field_embedder.token_embedder_bert.bert_model.prune_heads(self.heads_to_prune)
            model.eval()
            self.models.append(model)

    @staticmethod
    def _get_model_data(model_path):
        model_name = model_path.split('/')[-1]
        tr_model, stf = model_name.split('_')[:2]
        return tr_model, int(stf)

    def _restore_model(self, input_path):
        if os.path.isdir(input_path):
            print("Model could not be restored from directory", file=sys.stderr)
            filenames = []
        else:
            filenames = [input_path]
        for model_path in filenames:
            try:
                if torch.cuda.is_available():
                    loaded_model = torch.load(model_path)
                else:
                    loaded_model = torch.load(model_path,
                                              map_location=lambda storage,
                                                                  loc: storage)
            except:
                print(f"{model_path} is not valid model", file=sys.stderr)
            own_state = self.model.state_dict()
            for name, weights in loaded_model.items():
                if name not in own_state:
                    continue
                try:
                    if len(filenames) == 1:
                        own_state[name].copy_(weights)
                    else:
                        own_state[name] += weights
                except RuntimeError:
                    continue
        print("Model is restored", file=sys.stderr)

    def predict(self, batches):
        t11 = time()
        predictions = []
        for batch, model in zip(batches, self.models):
            batch = util.move_to_device(batch.as_tensor_dict(), 0 if torch.cuda.is_available() else -1)
            with torch.no_grad():
                prediction = model.forward(**batch)
            predictions.append(prediction)

        preds, idx, error_probs = self._convert(predictions)
        t55 = time()
        if self.log:
            print(f"Inference time {t55 - t11}")
        return preds, idx, error_probs

    def get_token_action(self, token, index, prob, sugg_token):
        """Get lost of suggested actions for token."""
        # cases when we don't need to do anything
        if prob < self.min_error_probability or sugg_token in [UNK, PAD, '$KEEP']:
            return None

        if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token == '$DELETE':
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
            start_pos = index + 1
            end_pos = index + 1

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob

    def _get_embbeder(self, weigths_name, special_tokens_fix):
        embedders = {'bert': PretrainedBertEmbedder(
            pretrained_model=weigths_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=special_tokens_fix)
        }

        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders=embedders,
            embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]},
            allow_unmatched_keys=True)
        return text_field_embedder

    def _get_indexer(self, weights_name, special_tokens_fix):
        bert_token_indexer = PretrainedBertIndexer(
            pretrained_model=weights_name,
            do_lowercase=self.lowercase_tokens,
            max_pieces_per_token=5,
            use_starting_offsets=True,
            truncate_long_sequences=True,
            special_tokens_fix=special_tokens_fix,
            is_test=True
        )
        return {'bert': bert_token_indexer}

    def preprocess(self, token_batch):
        seq_lens = [len(sequence) for sequence in token_batch if sequence]
        if not seq_lens:
            return []
        max_len = min(max(seq_lens), self.max_len)
        batches = []
        for indexer in self.indexers:
            batch = []
            for sequence in token_batch:
                tokens = sequence[:max_len]
                tokens = [Token(token) for token in ['$START'] + tokens]
                batch.append(Instance({'tokens': TextField(tokens, indexer)}))
            batch = Batch(batch)
            batch.index_instances(self.vocab)
            batches.append(batch)

        return batches

    def _convert(self, data):
        all_class_probs = torch.zeros_like(data[0]['class_probabilities_labels'])
        error_probs = torch.zeros_like(data[0]['max_error_probability'])
        for output, weight in zip(data, self.model_weights):
            all_class_probs += weight * output['class_probabilities_labels'] / sum(self.model_weights)
            error_probs += weight * output['max_error_probability'] / sum(self.model_weights)

        max_vals = torch.max(all_class_probs, dim=-1)
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
        return probs, idx, error_probs.tolist()

    def update_final_batch(self, final_batch, pred_ids, pred_batch,
                           prev_preds_dict):
        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def postprocess_batch(self, batch, all_probabilities, all_idxs,
                          error_probs,
                          max_len=50):
        all_results = []
        noop_index = self.vocab.get_token_index("$KEEP", "labels")
        for tokens, probabilities, idxs, error_prob in zip(batch,
                                                           all_probabilities,
                                                           all_idxs,
                                                           error_probs):
            length = min(len(tokens), max_len)
            edits = []

            # skip whole sentences if there no errors
            if max(idxs) == 0:
                all_results.append(tokens)
                continue

            # skip whole sentence if probability of correctness is not high
            if error_prob < self.min_error_probability:
                all_results.append(tokens)
                continue

            for i in range(length + 1):
                # because of START token
                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]
                # skip if there is no error
                if idxs[i] == noop_index:
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i],
                                                             namespace='labels')
                action = self.get_token_action(token, i, probabilities[i],
                                               sugg_token)
                if not action:
                    continue

                edits.append(action)
            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results

    def handle_batch(self, full_batch):
        """
        Handle batch of requests.
        """
        final_batch = full_batch[:]
        batch_size = len(full_batch)
        prev_preds_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        short_ids = [i for i in range(len(full_batch))
                     if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        total_updates = 0

        for n_iter in range(self.iterations):
            orig_batch = [final_batch[i] for i in pred_ids]

            sequences = self.preprocess(orig_batch)

            if not sequences:
                break
            probabilities, idxs, error_probs = self.predict(sequences)

            pred_batch = self.postprocess_batch(orig_batch, probabilities,
                                                idxs, error_probs)
            if self.log:
                print(f"Iteration {n_iter + 1}. Predicted {round(100*len(pred_ids)/batch_size, 1)}% of sentences.")

            final_batch, pred_ids, cnt = \
                self.update_final_batch(final_batch, pred_ids, pred_batch,
                                        prev_preds_dict)
            total_updates += cnt

            if not pred_ids:
                break

        return final_batch, total_updates

    def extract_candidate_words(self, full_batch, layer = 0):
        #adapting the handle batch and predict methods in order to extract attention weights
        final_batch = full_batch[:]
        short_ids = [i for i in range(len(full_batch))
                     if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]

        #assuming one iteration for now... TBD what we will do here
        # for n_iter in range(self.iterations):
        orig_batch = [final_batch[i] for i in pred_ids]
        sequences = self.preprocess(orig_batch)
        batch_imp_tokens = []
        #we will only be using a single model...BERT
        for batch, model in zip(sequences, self.models):
            batch = util.move_to_device(batch.as_tensor_dict(), 0 if torch.cuda.is_available() else -1)
            with torch.no_grad():
                input_ids, offsets =batch['tokens']['bert'], batch['tokens']['bert-offsets']
                _, attention_outputs =  self.models[0].text_field_embedder.token_embedder_bert(input_ids, offsets,
                                                                                        extract_attention = True)
            #take out attention matrices for layer of interest
            layer_attn = attention_outputs[layer]  # shape = (2,12,10,10)
            #sum over multi heads
            layer_aggr_attn = torch.sum(layer_attn, axis=1)  # aggregate heads
            # layer_aggr_attn =layer_attn[:,4,:,:]  # or pick a particular head
            # now shape is (2,10,10)
            #sum attention weights for each input token
            token_list = list(self.indexers[0]['bert'].vocab.keys())
            attention_vals = torch.sum(layer_aggr_attn, axis=1)
            max_idxs = []
            for s_idx, sentence in enumerate(attention_vals):
                # for debugging
                bert_sen = [token_list[idx] for idx in input_ids[s_idx]]
                print('bert sentence:', bert_sen)
                print('sentence attn vals:', sentence)

                sent_scores = []
                word_score = 0
                for a_idx, attention_val in enumerate(sentence):
                    word = token_list[input_ids[s_idx, a_idx]]
                    # print(f'word: {word} with attn val {attention_val}')
                    if word not in ['[CLS]', '$', '##ST', '##AR', '##T', '[PAD]']:
                        if word == '[SEP]':
                            # sentence over
                            sent_scores.append(word_score)
                            # print('End of sentence')
                        else:
                            if word.startswith('##'):
                                # add to previous word
                                word_score += attention_val
                                # print(f'starts with ##, append to previous, current word score {word_score}')
                            elif word_score > 0:
                                sent_scores.append(word_score)
                                # print(f'end previous word with score {word_score}\n')
                                word_score = attention_val
                            else:
                                # first word in sentence
                                word_score += attention_val
                                # print('first word in sentence')
                print('original sentence: ', orig_batch[s_idx])
                b = np.argmax(sent_scores)
                print('idx of best:', b, '=', orig_batch[s_idx][b])
                max_idxs.append(np.argmax(sent_scores))

        return max_idxs

        #     # now make a mask to remove unwanted tokens
        #     # first count number of non-padding tokens
        #     nonzeros = attention_vals.count_nonzero(dim=1) - 1  # count non-zeros
        #     n_ones = nonzeros[0]
        #     ncols = attention_vals.shape[1]
        #     # now create a mask to eventually zero out unwanted tokens
        #     # start by zeroing out padding and last token in sentence
        #     mask = torch.repeat_interleave(torch.tensor([1, 0]), torch.tensor([n_ones, ncols - n_ones]), dim=None).unsqueeze(0)
        #     # loop through rows appending appropriate mask for each row
        #     for i, n_ones in enumerate(nonzeros[1:]):
        #         new_row = torch.repeat_interleave(torch.tensor([1, 0]), torch.tensor([n_ones, ncols - n_ones]), dim=None).unsqueeze(0)
        #         mask = torch.cat([mask, new_row], axis=0)
        #     # finally remove first 5 tokens from each sentence
        #     mask[:, :5] -= 1
        #     attention_vals *= mask  # apply the mask
        #
        #     # find token with max attention
        #     imp_token_index = attention_vals.argmax(axis=1)
        #     #extract id of token deemed important
        #     imp_tokens_id = input_ids.gather(1, imp_token_index.view(-1, 1))
        #     # given these id's we wish to find the original sentence id and return that
        #     # issue: have to deal with duplicates
        #     # bool if imp token is repeated in each sentence
        #     duplicate_seqs = torch.sum(input_ids == imp_tokens_id, axis=1) > 1
        #     # loop through duplicates and recover
        #     for idx, dup in enumerate(duplicate_seqs):
        #         if dup == True:
        #             # do something to deal with dups
        #             pass
        #     # we have the index in the bert embedded sentence
        #
        #     #get corresponding token using id
        #     token_list = list(self.indexers[0]['bert'].vocab.keys()) #ordered dict keys->list, okay to index in to
        #     imp_tokens = [token_list[idx] for idx in imp_tokens_id]
        #     #input id and token ,pairs
        #     #probably need both the index in input sentence and the word
        #     batch_imp_tokens.append(imp_tokens)
        # return batch_imp_tokens
