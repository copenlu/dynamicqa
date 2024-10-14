import argparse
import os
import pickle
import random
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import wandb
from generation_utils import *
import config
import tqdm
import json

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation_model', type=str, default='opt-350m')
# parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--dataset', type=str, default='run_1')
parser.add_argument('--add_context', type=bool, default=False)
parser.add_argument('--use_original_context', type=bool, default=False)
parser.add_argument("--start_num", type=int, default=0)


def get_neg_loglikelihoods(model, out):
    selected_indices = [i for i in range(args.start_num, len(out))]
    with torch.no_grad():
        result = []
        i = 0
        for sample in tqdm.tqdm(out):
            
            if i not in selected_indices: ### Allow us to run several codes in parallel
                i += 1
                continue
                
            if i % 100 == 0:
                print("Saved at %d" %(i))
                filename = f"../{config.output_dir}/sequences/{args.dataset}/{args.dataset}_{model_name}_{args.add_context}_{args.use_original_context}_generations_likelihoods_{str(args.start_num)}.pkl"
                pickle.dump(result, open(filename, "wb"))
                
            if args.use_original_context:
                if i%2 == 0: # REP context is EVEN #
                    i+= 1
                    continue
            else:
                if i%2 == 1: # OG context is ODD #
                    i+=1
                    continue
                    
            result_dict = {}
            id_ = sample['id_csv']
            prompt = encode_as_chat(dataset.loc[id_]).to(device)
            prompt = prompt['input_ids']
            average_neg_log_likelihoods = torch.zeros((len(sample[gen_key]),))
            average_unconditioned_neg_log_likelihoods = torch.zeros((len(sample[gen_key]),))
            neg_log_likelihoods = torch.zeros((len(sample[gen_key]),))
            neg_unconditioned_log_likelihoods = torch.zeros((len(sample[gen_key]),))
            pointwise_mutual_information = torch.zeros((len(sample[gen_key]),))
            sequence_embeddings = []

            if any(prompt[0] > model.vocab_size):
                print(f" PROMPT ERROR: {id_}")
                continue
                #prompt = torch.reshape(prompt[prompt!=g], (1,-1))
            
            generations = []

            for generation_index in range(len(sample[gen_key])):

                if 'Qwen' in args.evaluation_model:
                    if sample[gen_key][generation_index] == '':
                        generation = torch.LongTensor(tokenizer.encode('<|im_end|>'))
                    else:
                        generation = torch.LongTensor(tokenizer.encode(sample[gen_key][generation_index]+' <|im_end|>'))
                        
                else:
                    if sample[gen_key][generation_index] == '':
                        generation = torch.LongTensor([tokenizer.eos_token_id])
                    else:
                        generation = torch.LongTensor(tokenizer.encode(sample[gen_key][generation_index]))
                        
                # haeun : added to prevent index error
                if any(generation > model.vocab_size):
                    print(f"GENERATION ERROR: {id_}")
                    continue
                    #generation = generation[generation!=g]

                if generation.size(0) == 0:
                    generation = torch.tensor([tokenizer.eos_token_id])

                generation = generation.to(device)
    
                generations.append(generation.to('cpu'))

                # generation = generations[generation_index] #[generations[generation_index] != tokenizer.pad_token_id]
                # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity

                full_generation = torch.cat([prompt.T, torch.reshape(generation,(1,-1)).T]).clone().T
                target_ids = full_generation.clone()
                target_ids[:,:prompt.shape[1]] = -100
                model_output = model(torch.reshape(full_generation, (1, -1)), labels=target_ids, output_hidden_states=True)
                generation_only = generation.clone()
                unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                   labels=generation_only,
                                                   output_hidden_states=True)
                hidden_states = model_output['hidden_states']
                average_neg_log_likelihood = model_output['loss']

                average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']
                average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
                average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood
                neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
                neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * (
                    len(generation) - len(prompt))
                pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                    generation_index] + neg_unconditioned_log_likelihoods[generation_index]

                average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
                sequence_embeddings.append(average_of_last_layer_token_embeddings)

            most_likely_generation = torch.LongTensor(tokenizer.encode(answers_output[i]))######
            # haeun : added to prevent index error
            
            if any(most_likely_generation > model.vocab_size):
                print(f"MOST LIKELY GENERATION ERROR: {id_}")
                continue
            #         most_likely_generation = most_likely_generation[most_likely_generation!=g]
                    
            if most_likely_generation.size(0) == 0:
                most_likely_generation = torch.tensor([tokenizer.eos_token_id])
                
            most_likely_generation = most_likely_generation.to(device)
            most_likely_generation = (torch.cat([prompt.T, torch.reshape(most_likely_generation,(1,-1)).T]).clone().T).clone()
                    
            target_ids = most_likely_generation.clone()
            
            target_ids[:,:prompt.shape[1]] = -100
            model_output = model(torch.reshape(most_likely_generation, (1, -1)),
                                 labels=torch.reshape(target_ids,(1,-1)),
                                 output_hidden_states=True)
            # 
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood_of_most_likely_gen = model_output['loss']
            most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

            neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (
                len(most_likely_generation) - len(prompt))

            # if sequence_embeddings == []:
            #     pdb.set_trace()
            sequence_embeddings = torch.stack(sequence_embeddings)
            result_dict['prompt'] = prompt.to('cpu')
            # breakpoint()
            result_dict['generations'] = generations #.to('cpu')
            result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods.to('cpu')
            result_dict['neg_log_likelihoods'] = neg_log_likelihoods.to('cpu')
            result_dict['sequence_embeddings'] = most_likely_generation_embedding.to('cpu')
            result_dict['most_likely_sequence_embedding'] = most_likely_generation.to('cpu')
            result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods.to('cpu')
            result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods.to('cpu')
            result_dict['pointwise_mutual_information'] = pointwise_mutual_information.to('cpu')
            result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen.to('cpu')
            result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen.to('cpu')
            result_dict['id'] = id_
                
            result.append(result_dict)
            i += 1

        return result
    
if __name__ == '__main__':

    args = parser.parse_args()

    ensure_reproducibility(10)
    minimum_args(args)

    print(args.evaluation_model, args.dataset)
    print(args.add_context, args.use_original_context)

    model, tokenizer, device = load_model(args.evaluation_model, quantize=False) #quantize=True)

    model_name = args.evaluation_model.split('/')[1]#.split('-') #[0]

    with open(f"../Output/sequences/{args.dataset}/{args.dataset}_{model_name}_persuasion_output.pkl", "rb") as f:
        out = pickle.load(f)
        
    ### FOR DEBUG

    if args.dataset == 'dispute':
        dataset = pd.read_csv(f"../Data/Disputable_subset.csv", index_col='id') #.iloc[:10]

    else:
        dataset = pd.read_csv(f"../Data/{args.dataset.title()}_final.csv", index_col='id') #.iloc[:10]


    gen_key = 'answers_c_and_q' if args.add_context else 'answers_q_only'
    sim_key = 'clusters_c_and_q' if args.add_context else 'clusters_q_only'

    period_token_id = tokenizer('. ')['input_ids'][1]

    ans_key = 'context' if args.add_context else 'q_only'

    with open(f"../Output/sequences/{args.dataset}/{args.dataset}_{model_name}_{ans_key}.json", "rb") as f:
        answers_output = json.load(f)
        
    # pdb.set_trace()
    likelihoods = get_neg_loglikelihoods(model, out)

    with open(f"../{config.output_dir}/sequences/{args.dataset}/{args.dataset}_{model_name}_{args.add_context}_{args.use_original_context}_generations_likelihoods_{str(args.start_num)}.pkl",
            'wb') as outfile:
        pickle.dump(likelihoods, outfile)
