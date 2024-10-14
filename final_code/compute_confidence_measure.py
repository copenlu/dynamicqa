import argparse
import os
import pickle
import random

import config
import numpy as np
import torch
import wandb
import pandas as pd
import numpy as np
import pdb
from generation_utils import *

### From Kuhn et al, Semantic Uncertainty ICLR 2023: https://github.com/lorenzkuhn/semantic_uncertainty

def get_overall_log_likelihoods(list_of_results):
    # breakpoint()
    """Compute log likelihood of all generations under their given context.
    
    list_of_results: list of dictionaries with keys:
    
    returns: dictionary with keys: 'neg_log_likelihoods', 'average_neg_log_likelihoods'
             that contains tensors of shape (num_models, num_generations, num_samples_per_generation)
    """

    result_dict = {}

    list_of_keys = ['neg_log_likelihoods', 'average_neg_log_likelihoods', 'sequence_embeddings',\
                    'pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                    'neg_log_likelihood_of_most_likely_gen']

    for key in list_of_keys:
        list_of_ids = []
        overall_results = []
        for model_size, result in list_of_results:
            results_per_model = []
            for sample in result:
                average_neg_log_likelihoods = sample[key]
                list_of_ids.append(sample['id'])
                if torch.is_tensor(average_neg_log_likelihoods):
                    results_per_model.append(average_neg_log_likelihoods)
                # else:
                #     type(average_neg_log_likelihoods[0]) == str:
                #         results_per_model.append(torch.Tensor)
                    
                    
            # breakpoint()
            if key != 'semantic_set_ids':
                results_per_model = torch.stack(results_per_model)

            overall_results.append(results_per_model)

        if key != 'sequence_embeddings' and key != 'semantic_set_ids':
            overall_results = torch.stack(overall_results)

        result_dict[key] = overall_results

    result_dict['ids'] = list_of_ids
    return result_dict


def get_mutual_information(log_likelihoods):
    """Compute confidence measure for a given set of likelihoods"""

    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    tiled_mean = mean_across_models.tile(log_likelihoods.shape[0], 1, 1)
    diff_term = torch.exp(log_likelihoods) * log_likelihoods - torch.exp(tiled_mean) * tiled_mean
    f_j = torch.div(torch.sum(diff_term, dim=0), diff_term.shape[0])
    mutual_information = torch.div(torch.sum(torch.div(f_j, mean_across_models), dim=1), f_j.shape[-1])

    return mutual_information


def get_log_likelihood_variance(neg_log_likelihoods):
    """Compute log likelihood variance of approximate posterior predictive"""
    mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
    variance_of_neg_log_likelihoods = torch.var(mean_across_models, dim=1)

    return variance_of_neg_log_likelihoods


def get_log_likelihood_mean(neg_log_likelihoods):
    """Compute softmax variance of approximate posterior predictive"""
    mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
    mean_of_neg_log_likelihoods = torch.mean(mean_across_models, dim=1)

    return mean_of_neg_log_likelihoods


def get_mean_of_poinwise_mutual_information(pointwise_mutual_information):
    """Compute mean of pointwise mutual information"""
    mean_across_models = torch.mean(pointwise_mutual_information, dim=0)
    return torch.mean(mean_across_models, dim=1)


def get_predictive_entropy(log_likelihoods):
    """Compute predictive entropy of approximate posterior predictive"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    entropy = -torch.sum(mean_across_models, dim=1) / torch.tensor(mean_across_models.shape[1])
    return entropy

def translate_semantic_ids(semantic_set_ids):

    new_set_ids = []
    for sid in semantic_set_ids:
        to_fill = np.zeros(10, dtype=int)
        for i, st in enumerate(sid):
            for j in st:
                to_fill[j] = i
        new_set_ids.append(to_fill)
    
    return torch.Tensor(new_set_ids)


def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids):
    """Compute the semantic entropy"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    # This is ok because all the models have the same semantic set ids

    entropies = []
    for row_index in range(mean_across_models.shape[0]):
        aggregated_likelihoods = []
        row = mean_across_models[row_index]
        # breakpoint()
        semantic_set_ids_row = semantic_set_ids[row_index]
        for semantic_set_id in torch.unique(semantic_set_ids_row):
            # breakpoint()
            aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row.cpu() == semantic_set_id.cpu()], dim=0))
        # breakpoint()
        aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
        entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
        entropies.append(entropy)

    return torch.tensor(entropies)


def get_margin_probability_uncertainty_measure(log_likelihoods):
    """Compute margin probability uncertainty measure"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    topk_likelihoods, indices = torch.topk(mean_across_models, 2, dim=1, sorted=True)
    margin_probabilities = np.exp(topk_likelihoods[:, 0]) - np.exp(topk_likelihoods[:, 1])

    return margin_probabilities

def get_sem_sims():
    to_release = []
    with open(f"../Output/sequences/{args.dataset}/{args.dataset}_{model_name}_persuasion_output.pkl", "rb") as f:
        out = pickle.load(f)
    for sample in out:
        interim = [k.split('-') for k in sample[sim_key]]
        to_append = [[int(x) for x in y] for y in interim]
        to_release.append(to_append)
    return to_release


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_model', type=str, default='opt-350m')
    parser.add_argument('--dataset', type=str, default='temporal')
    parser.add_argument('--add_context', type=bool, default=False)
    parser.add_argument('--use_original_context', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=True)
    args = parser.parse_args()

    device = 'cuda'

    model_name = args.generation_model.split('/')[1].split('-')[0]
    run_name = f"{args.dataset}"


    if args.dataset == 'dispute':
        dataset = pd.read_csv(f"../Data/Disputable_subset.csv", index_col='id') #.iloc[:10]

    else:
        dataset = pd.read_csv(f"../Data/{args.dataset.title()}_final.csv", index_col='id') #.iloc[:10]

        
    gen_key = 'answers_c_and_q' if args.add_context else 'answers_q_only'
    sim_key = 'clusters_c_and_q' if args.add_context else 'clusters_q_only'


    model_name = args.generation_model.split('/')[1] #.split('-')[0]
    if args.add_context:
        file_out_type = "q_and_c"
    else:
        file_out_type = "q_only"

    # Set a seed value
    seed_value = 10
    ensure_reproducibility(seed_value)
    

    llh_shift = torch.tensor(5.0)


    list_of_results = []

    with open(f"../Output/sequences/{args.dataset}/{args.dataset}_{model_name}_{args.add_context}_{args.use_original_context}_generations_likelihoods_0.pkl", "rb") as infile:
        sequences = pickle.load(infile)
        list_of_results.append((args.generation_model, sequences))
        
    overall_results = get_overall_log_likelihoods(list_of_results)

    overall_results['semantic_set_ids']  = get_sem_sims()
    overall_results['semantic_set_ids'] = translate_semantic_ids(overall_results['semantic_set_ids'])
    mutual_information = get_mutual_information(-overall_results['neg_log_likelihoods'])
    predictive_entropy = get_predictive_entropy(-overall_results['neg_log_likelihoods'])
    predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods'],
                                                                            overall_results['semantic_set_ids'])
    unnormalised_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['neg_log_likelihoods'],
                                                                            overall_results['semantic_set_ids'])

    margin_measures = get_margin_probability_uncertainty_measure(-overall_results['average_neg_log_likelihoods'])
    unnormalised_margin_measures = get_margin_probability_uncertainty_measure(-overall_results['neg_log_likelihoods'])


    def get_number_of_unique_elements_per_row(tensor):
        assert len(tensor.shape) == 2
        return torch.count_nonzero(torch.sum(torch.nn.functional.one_hot(tensor.to(dtype=int)), dim=1), dim=1)

    # breakpoint()
    number_of_semantic_sets = get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'])
    average_predictive_entropy = get_predictive_entropy(-overall_results['average_neg_log_likelihoods'])
    average_predictive_entropy_on_subsets = []
    predictive_entropy_on_subsets = []
    semantic_predictive_entropy_on_subsets = []
    num_predictions = overall_results['average_neg_log_likelihoods'].shape[-1]
    number_of_semantic_sets_on_subsets = []
    for i in range(1, num_predictions + 1):
        offset = num_predictions * (i / 100)
        average_predictive_entropy_on_subsets.append(
            get_predictive_entropy(-overall_results['average_neg_log_likelihoods'][:, :, :int(i)]))
        predictive_entropy_on_subsets.append(get_predictive_entropy(-overall_results['neg_log_likelihoods'][:, :, :int(i)]))
        semantic_predictive_entropy_on_subsets.append(
            get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods'][:, :, :int(i)],
                                                overall_results['semantic_set_ids'][:, :int(i)]))
        number_of_semantic_sets_on_subsets.append(
            get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][:, :i]))

    average_pointwise_mutual_information = get_mean_of_poinwise_mutual_information(
        overall_results['pointwise_mutual_information'])


    overall_results['mutual_information'] = mutual_information
    overall_results['predictive_entropy'] = predictive_entropy
    overall_results['predictive_entropy_over_concepts'] = predictive_entropy_over_concepts
    overall_results['unnormalised_entropy_over_concepts'] = unnormalised_entropy_over_concepts
    overall_results['number_of_semantic_sets'] = number_of_semantic_sets.detach().cpu().numpy()
    overall_results['margin_measures'] = margin_measures
    overall_results['unnormalised_margin_measures'] = unnormalised_margin_measures

    overall_results['average_predictive_entropy'] = average_predictive_entropy
    for i in range(len(average_predictive_entropy_on_subsets)):
        overall_results[f'average_predictive_entropy_on_subset_{i + 1}'] = average_predictive_entropy_on_subsets[i]
        overall_results[f'predictive_entropy_on_subset_{i + 1}'] = predictive_entropy_on_subsets[i]
        overall_results[f'semantic_predictive_entropy_on_subset_{i + 1}'] = semantic_predictive_entropy_on_subsets[i]
        overall_results[f'number_of_semantic_sets_on_subset_{i + 1}'] = number_of_semantic_sets_on_subsets[i].detach().cpu().numpy()
    overall_results['average_pointwise_mutual_information'] = average_pointwise_mutual_information

    for k in overall_results.keys():
        if type(overall_results[k]) == list:
            if type(overall_results[k][0]) == int:
                continue
            else:
                try:
                    new_list = [it.detach().cpu().numpy() for it in overall_results[k]]
                except AttributeError:
                    new_list = overall_results[k]
        elif overall_results[k].dtype == torch.float32:
            overall_results[k] = overall_results[k].detach().cpu().numpy()
        elif overall_results[k].dtype == torch.float16:
            overall_results[k] = overall_results[k].detach().cpu().numpy()
            

    # overall_results['semantic_set_ids'] = overall_results['semantic_set_ids'].detach().cpu().numpy()
    overall_results['sequence_embeddings'] = [it.detach().cpu().numpy() for it in overall_results['sequence_embeddings']]


    with open(f"../{config.output_dir}/{model_name}_{args.dataset}_{args.use_original_context}_{file_out_type}.pkl", ## Save results
            'wb') as outfile:
        pickle.dump(overall_results, outfile)

    print(f"saved to {config.output_dir}/final/{model_name}_{args.dataset}_{args.use_original_context}_{file_out_type}.pkl")

    if args.verbose:
        print("Avg normalized SE: ", predictive_entropy_over_concepts.mean())
        print("Avg unnormalized SE: ", unnormalised_entropy_over_concepts.mean())
        