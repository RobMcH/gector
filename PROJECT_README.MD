# PROJECT README
In this readme we outline how to use our code and how to reproduce the results reported in the report. All code is also
available on Github on our [project repository](https://github.com/RobMcH/gector).

## Setup
To install the dependencies of the repository execute the command `pip install -r requirements.txt`. Additionally, a
spacy model must be installed via `python -m spacy download en_core_web_sm`. Further, the [M2 scorer](https://www.comp.nus.edu.sg/~nlp/conll14st.html) as well as [ERRANT 2.0](https://github.com/chrisjbryant/errant)
are required for evaluation. Note that they require specific Python and spacy versions which clash with our dependencies,
therefore separate Python installations might be required to run them.

## Obtaining the Models
To get the baseline GECToR model please refer to the official [Grammarly repository](https://github.com/grammarly/gector).
As our models are about 450 MBytes each, we provide download links to them on Amazon S3:

| Model     | Link                                                          |
|-----------|---------------------------------------------------------------|
| Random<sup>1</sup>  | [Download](https://gector.s3.eu-west-2.amazonaws.com/model_random.th)     |
| Att<sup>1</sup><sub>min</sub> | [Download](https://gector.s3.eu-west-2.amazonaws.com/model_att_min.th)    |
| Att<sup>1</sup><sub>max</sub> | [Download](https://gector.s3.eu-west-2.amazonaws.com/model_att_max.th)    |
| Random<sup>2</sup>  | [Download](https://gector.s3.eu-west-2.amazonaws.com/model_random_2p.th)  |
| Att<sup>2</sup><sub>min</sub> | [Download](https://gector.s3.eu-west-2.amazonaws.com/model_att_min_2p.th) |
| Att<sup>2</sup><sub>max</sub> | [Download](https://gector.s3.eu-west-2.amazonaws.com/model_att_max_2p.th) |

## Inference
To use models for inference the following command can be used  
```python predict.py --model_path MODEL --input_file INPUT --output_file OUTPUT --transformer_model bert --special_tokens_fix 0 --min_error_probability MEP --additional_confidence AC```  
where capitalised variables are placeholders.

## Extract Adversarial Examples
To extract adversarial examples the following command can be used  
```python attention_extraction.py --model_path MODEL --input_file INPUT --label_file LABELS --transformer_model bert --special_tokens_fix 0 --attack AT_TYPE --num_perturbations NP --attention_sorting AS_TYPE```  
where `AT_TYPE` can be one of `random, attention` and `AS_TYPE` can be one of `min, max`. Note that `label_file` here refers
to the file containing the ground truth for the grammatically incorrect sentences. This is needed to generate an adjusted
label file in case any tokens are swapped for synonyms. After running the above command two files will be created in the
local directory. These contain perturbations for all input sentences. To obtain only the ones that the model can't predict
correctly, the following command can be used
```python predict.py --model_path MODEL --input_file INPUT --output_file OUTPUT --transformer_model bert --special_tokens_fix 0 --min_error_probability MEP --additional_confidence AC --only_wrong 1 --label_file LABELS```
which will create two files `OUTPUT` and `OUTPUT_labels` containing the perturbed sentences that the model could not correct
and their corresponding labels.

## Fine-tuning the Models
In our report we always extend the original training [W&I+LOCNESS training corpus](https://www.cl.cam.ac.uk/research/nl/bea2019st/)
with adversarial examples generated on the same corpus. For this, commands such as `cat CORPUS ADV > NEW_CORPUS` can be used,
with `CORPUS` being the path to the original data and `ADV` being the path to the adversarial examples (generated as described
above). To use this data the GECToR pre-processing script needs to be run on it:  
```python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE```
with `SOURCE` being the input and `TARGET` being the corresponding labels. The fine-tuning can then be run with the following
command:  
```python train.py --train_set INPUT --dev_set INPUT --model_dir MODELS --n_epoch 1 --transformer_model 'bert' --special_tokens_fix 0 --pretrain_folder PATH_TO_MODEL_FOLDER --pretrain MODEL_NAME --vocab_path data/output_vocabulary/ --cold_steps_count 0 --accumulation_size 2 --tn_prob 1 --tp_prob 1 --updates_per_epoch 0 --skip_correct 0```  
where `INPUT` denotes the pre-processed data, `MODELS` the folder where the fine-tuned models will be output to,
`PATH_TO_MODEL_FOLDER` the path to the folder where the pre-trained model is stored, `MODEL_NAME` the filename of the model (without file extension),
and the remaining variables being hyperparameters that we adopted from the Grammarly training procedure (c.f. [here](https://github.com/grammarly/gector/blob/master/docs/training_parameters.md)).

## Evaluating the Models
To evaluate a model first predictions for the corresponding input file need to be made as described above. Depending on the data
further processing is required afterwards. For the BEA-2019 data sets the files need to be converted to the M2 format using ERRANT (c.f. [here](https://github.com/chrisjbryant/errant)).
The resulting M2 file can then be evaluated with ERRANT by comparing it to the ground truth M2 file (provided as part of the data set on the official website).
For CoNLL-2014 the M^2 scorer needs to be used, but here the output of predict.py can be directly used as an input to the scorer. The ground
truth needs to be in M2 format, though (also provided as part of the data set from the official website).

## Hyperparameters
For the inference hyperparameters used we refer to the appendix of the report.
