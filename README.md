# Probing the Dynamics of Cross-lingual Alignment throughout Training in Multilingual Language Models

This repository contains code to replicate the experiments in the paper: Probing the Dynamics of Cross-lingual Alignment throughout Training in Multilingual Language Models.

<!-- TODO: add paper link -->

The experiments mainly fall into three categories, each in a seperate directory: 

[**Neuron Probing**](##Neuron-Probing) | [**Cross-lingual Transfer**](##Cross-lingual-Transfer-Ability-Evaluation) | [**Parallel Sentence Similarity**](##Paralllel-Sentence-Similarity)

## Neuron Probing
This batch of code is under `multilingual-typology-probing/`, which is used to probe the neurons that encode the most morphosyntactic information. This batch of code is inherented from the paper: [Same Neurons, Different Languages: Probing Morphosyntax in Multilingual Pre-trained Models (Stańczak et al., NAACL 2022)](https://arxiv.org/abs/2205.02023) and their [repo](https://github.com/copenlu/multilingual-typology-probing). 

### Setup
Firstly, run the following commands to install and activate the conda environment:
```
git clone https://github.com/ErikaaWang/probing-multilingual-dynamics.git
cd probing-multilingual-dynamics/multilingual-typology-probing
conda env create -f environment.yml
conda activate multilingual-typology-probing
```

Then, follow the instructions to prepare the data for probes:

1. Run `mkdir unimorph && cd unimorph && wget https://raw.githubusercontent.com/unimorph/um-canonicalize/master/um_canonicalize/tags.yaml`
2. Download [UD 2.1 treebanks](https://universaldependencies.org/) and put them in `data/ud/ud-treebanks-v2.1`
3. Clone the modified [UD converter](https://github.com/ltorroba/ud-compatibility) under `probing-multilingual-dynamics/` and then convert the treebank annotations to the UniMorph schema using`./scripts/ud_to_um.sh`.
4. Run `./scripts/preprocess_bloom.sh $BLOOM $LAYER $CHECKPOINT` to prepare the dataset $\mathcal{D} = \{(\pi^{(n)}, \bm{h}^{(n)})\}^{N}_{n=1}$ mentioned in our paper. To set variables, replace `$BLOOM`, `$LAYER` and `$CHECKPOINT` with the proper model sizes and their available layers and checkpoints you desire. e.g. use `./scripts/preprocess_bloom.sh bloom-560m 17 1000` to prepare the data for probing the inter layer 17 of [bloom-560m](https://huggingface.co/bigscience/bloom-560m) with the [checkpoints](https://huggingface.co/bigscience/bloom-560m-intermediate) at global step 1000. 

### Run Probing
Use the following command to run the neuron probing experiments:
```
python run.py --language $LANG --experiment-name inter-layer-$LAYER --attribute $ATTR --trainer poisson --gpu --embedding $BLOOM-intermediate-global_step$CHECKPOINT greedy --selection-size 50 --selection-criterion mi
```

for a specific langugage-attribute pair. All available `$LANG` is listed in `scripts/languages_bloom.lst` and all available `$ATTR` is listed in `scripts/properties.lst`, and `$BLOOM`, `$LAYER`, `$CHECKPOINT` should be already used in the aforementioned data preparation.


## Cross-lingual Transfer Ability Evaluation
This batch of code is under `xtreme/`, which is an edited version of the original [xtreme benchmark](https://github.com/google-research/xtreme) that allows to test cross-lingual transfer ability of BLOOM on XNLI and POS tagging tasks. Each checkpoint model is trained on English data only, and test directly on other languages. [QLoRA](https://arxiv.org/abs/2305.14314) is integrated to allow fine-tuning on single GPU. 

Please first follow the instructions in the [Download the data](https://github.com/ErikaaWang/xtreme?tab=readme-ov-file#download-the-data) section of XTREME to prepare the environment and data ready. We recommend to install a seperate conda environment instead of using the previous one. 

Then, use the following command: 

```
bash scripts/train.sh $BLOOM-intermediate-global_step$CHECKPOINTS $TASK
```
where `$BLOOM` and `$CHECKPOINTS` keeps the same as above. For variable `$TASK`, use `xnli` for sequence classification on XNLI and `udpos` for POS tagging. 

e.g. 
```
bash scripts/train.sh bloom-560m-intermediate-global_step1000 udpos
```

## Parallel Sentence Similarity
Please first download the OPUS data [here](), and upzip it under `parallel-sentence-similarity/data`.

Then, use the following command to compute the average similarity of the parallel sentences, in the same environment you created for xtreme:

```
python similarity.py $BLOOM-intermediate-global_step$CHECKPOINTS $DATASET $SRC_LANG $TRG_LANG --use-gpu --sent-upper-bound $UPPER_BOUND
```
where `$BLOOM` and `$CHECKPOINTS` keeps the same as above. For `$DATASET`, use `opus` for similarity between English and other target languages, and `CodeXGLUE` for similarity between natural language and code. In our paper, we set `$UPPER_BOUND` as 30000. 

For opus, we always set source language to be English(`en`). We compute the similarity of a list of target languages in ISO 639-1 code: `'ar', 'es', 'eu', 'fr', 'hi', 'pt', 'ta', 'ur', 'vi'`.  An example for similarity between English(en) and Arabic(ar):
```
python similarity.py bloom-560m-intermediate-global_step1000 opus en ar --use-gpu --sent-upper-bound 30000
```

For CodeXGLUE, the source and target language should be set as `nl` and `code` respectively. An example for similarity between natural language and code:
```
python similarity.py bloom-560m-intermediate-global_step1000 CodeXGLUE nl code --use-gpu --sent-upper-bound 30000
```


## Extra Information

#### Citation

Please cite our paper if you found it useful:


```
```

#### Contact

To ask questions or report problems, please open an [issue](https://github.com/ErikaaWang/probing-multilingual-dynamics/issues).

