import os
from os import path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import csv

from transformers import BloomTokenizerFast, BloomModel
from datasets import load_dataset
from argparse import ArgumentParser

_DEFAULT_PARALLEL_CORPUS_ROOT = path.join(os.getcwd(), 'data')
# # for autoDL machines
# _DEFAULT_PARALLEL_CORPUS_ROOT = path.join('../autodl-tmp', 'data')
_DEFAULT_OUTPUT_FILE_ROOT = 'experiments/cos_similarity_csv/'

parser = ArgumentParser()
parser.add_argument("bloom", default=None, help='bloom model type. Options: bloom-560m, bloom1b1 ... or thier intermedaite models.')
# bloom-1b1-intermediate-global_step1000
parser.add_argument("parallel_corpus", type=str, help='corpus used. Options: opus, CodeXGLUE') 
parser.add_argument("src_lang", type=str, help='source language. ')   # en, 'nl' if use CodeXGLUE corpus
parser.add_argument("trg_lang", type=str, help='target language. ')   # vi, 'code' if use CodeXGLUE corpus
parser.add_argument("--use-gpu", action="store_true", default=False)
parser.add_argument("--sent-upper-bound", type=int, default=None, help="Set the upper bound of comparing data. ")
parser.add_argument("--inter-layer", type=lambda x: x.split(','), default=list(range(1,26)), help='default to compute all layers. \
                    Preprocess the embedding from intermediate layer. available layer: [1,25]. ')
parser.add_argument("--reset-HF-cache-dir", action="store_true", default=False, help="If enabled, the hugging face\
                    cache dir will be switched to the current work dir. ") 
args = parser.parse_args()

# deal with args  
if args.parallel_corpus == 'CodeXGLUE':
    assert args.src_lang == 'nl' and args.trg_lang == 'code'
bloom_model = 'bigscience/' + args.bloom
checkpoint = None
if '-global_step' in args.bloom:   # e.g. bloom-1b1-intermediate-global_step1000
    bloom_args = args.bloom.split('-global_step')
    bloom_model = 'bigscience/' + bloom_args[0]
    checkpoint = 'global_step' + bloom_args[1]

parallel_data_path = path.join(_DEFAULT_PARALLEL_CORPUS_ROOT, args.parallel_corpus, args.src_lang + '-' + args.trg_lang)
output_file_dir_root = path.join(_DEFAULT_OUTPUT_FILE_ROOT, args.bloom)

device = 'cpu'
if args.use_gpu:
    print("Using GPU")
    device = 0
    
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")

model_args_dic = {}
if args.reset_HF_cache_dir:
    model_args_dic['cache_dir'] = os.getcwd()

if args.inter_layer is not None:
    # keep in mind that the layer index should be (args.inter_layer - 1)
    model_args_dic['output_hidden_states'] = True
    print(f"Saving sentence representation from layer {args.inter_layer}...")

if checkpoint is not None:
    model_args_dic['revision'] = checkpoint
    print(f"Model revision: {model_args_dic['revision']}...")

model = BloomModel.from_pretrained(bloom_model,  **model_args_dic).to(device)
print(f"Using model {model.config.name_or_path}...")

similarity_list_dict = {i: [] for i in args.inter_layer}
# Dict: {layer_no: list of tuple: [(sent_id, cos_similarity), ...]}
total_sent = 0
upper_bound_flag = False

if args.parallel_corpus == 'opus': 
    print("Using Corpus: opus...")
    for split in ['train', 'dev', 'test']:
        src_file_name = split + '.' + args.src_lang
        trg_file_name = split + '.' + args.trg_lang
        print(f"Calculating for src files: {src_file_name}, trg: {trg_file_name}")

        src_lang_file = path.join(parallel_data_path, src_file_name)
        trg_lang_file = path.join(parallel_data_path, trg_file_name)
        

        with open(src_lang_file, "r") as src_f, open(trg_lang_file, "r") as trg_f: 
            # Prepare to compute BLOOM embeddings
            model.eval()

            for sent_id, (src_sent, trg_sent) in enumerate(tqdm(zip(src_f, trg_f))):

                # get the last hidden state of each sent pair respectively, and compare their similarity
                src_inputs = tokenizer(src_sent, return_tensors="pt").to(device)
                trg_inputs = tokenizer(trg_sent, return_tensors="pt").to(device)

                with torch.no_grad():
                    # last_hidden_state: output[0] 
                    # shape: (batch_size, sent_length, embedding_size)
                    src_outputs = model(**src_inputs)
                    trg_outputs = model(**trg_inputs)
                    
                    # get the last hidden state of the last token: full sentence representation
                    # shape: (batch_size, embedding_size)
                    # src_sent_hidden_state = src_outputs[0][:, -1, :].squeeze(1)
                    # trg_sent_hidden_state = trg_outputs[0][:, -1, :].squeeze(1)

                    for i in args.inter_layer:
                        # shape: (batch_size, embedding_size)
                        src_sent_hidden_state = src_outputs['hidden_states'][i-1][:, -1, :].squeeze(1)
                        trg_sent_hidden_state = trg_outputs['hidden_states'][i-1][:, -1, :].squeeze(1)

                        cos_similarity = F.cosine_similarity(src_sent_hidden_state, trg_sent_hidden_state)
                        similarity_list_dict[i].append((sent_id, float(cos_similarity)))     
                    
                total_sent += 1
                if args.sent_upper_bound is not None and total_sent > args.sent_upper_bound:
                    print("Upper Bound Reached!")
                    upper_bound_flag = True
                    break

        print(f"Current similarity amount collected: {[len(x) for x in similarity_list_dict.values()]}")           

        if upper_bound_flag:
            break

elif args.parallel_corpus == 'CodeXGLUE':
    print("Using Corpus: CodeXGLUE...")
    dataset = load_dataset('code_x_glue_tc_text_to_code')
    model.eval()

    for split, dataset_split in dataset.items():

        if split == 'test':
            # dataset_split[i]['code'] will always be none in test split, which will report an error
            continue

        for i in tqdm(range(0, len(dataset_split))):
            nl_inputs = tokenizer(dataset_split[i]['nl'], return_tensors="pt").to(device)
            code_inputs = tokenizer(dataset_split[i]['code'], return_tensors="pt").to(device)

            with torch.no_grad():
                # last_hidden_state: output[0] 
                # shape: (batch_size, sent_length, embedding_size)
                nl_outputs = model(**nl_inputs)
                code_outputs = model(**code_inputs)

                # get the last hidden state of the last token: full sentence representation
                # shape: (batch_size, embedding_size)
                nl_sent_hidden_state = nl_outputs[0][:, -1, :].squeeze(1)
                code_sent_hidden_state = code_outputs[0][:, -1, :].squeeze(1)

                for i in args.inter_layer: # save hidden_states
                    # shape: (batch_size, embedding_size)
                    nl_sent_hidden_state = nl_outputs['hidden_states'][i-1][:, -1, :].squeeze(1)
                    code_sent_hidden_state = code_outputs['hidden_states'][i-1][:, -1, :].squeeze(1)

                    cos_similarity = F.cosine_similarity(nl_sent_hidden_state, code_sent_hidden_state)
                    similarity_list_dict[i].append((dataset_split[i]['id'], float(cos_similarity)))  
                
            total_sent += 1
            if args.sent_upper_bound is not None and total_sent > args.sent_upper_bound:
                print("Upper Bound Reached!")
                upper_bound_flag = True
                break

        print(f"Current similarity amount collected: {[len(x) for x in similarity_list_dict.values()]}")       

        if upper_bound_flag:
            break

else:
    raise Exception("Please use corpus: opus or CodeXGLUE")

# save results
for i in args.inter_layer:

    # create output dir
    output_file_dir = path.join(output_file_dir_root, 'inter-layer-' + str(i))
    if i == 25:
        output_file_dir = path.join(output_file_dir_root, 'last-layer')
    if not path.exists(output_file_dir):
        os.makedirs(output_file_dir) 
    
    sent_ids, cos_similarities = zip(*similarity_list_dict[i])
    avg_similarity = sum(cos_similarities)/len(cos_similarities)
    print(f"There are {len(sent_ids)} similarities in total, average: {avg_similarity}")
    similarity_list_dict[i].append(('avg', avg_similarity))

    output_file = path.join(output_file_dir, args.src_lang + '-' + args.trg_lang + '.csv')
    with open(output_file,'w') as out:
        csv_out = csv.writer(out)
        for row in similarity_list_dict[i]:
            csv_out.writerow(row)

    print(f"Result saved in file: {output_file}")