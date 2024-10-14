import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize
import accelerate
import config
import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import os
import pdb
import config
import random

def load_model(model_name, quantize=True):
    global tokenizer
    global model
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    if quantize:
      if 'llama' in model_name:
          quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
      else:
          quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)

      model = AutoModelForCausalLM.from_pretrained(model_name,
                                                quantization_config=quantization_config,
                                                # attn_implementation="flash_attention_2",
                                                trust_remote_code=True,
                                                device_map="auto")
    
    else:
      model = AutoModelForCausalLM.from_pretrained(model_name,
                                            # attn_implementation="flash_attention_2",
                                            trust_remote_code=True,
                                            device_map="auto")
  

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"
    # model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, device

def send_args(args):
    global num_beams, add_context, model_name, dataset, use_original_context, decoding_method, temperature, top_p
    num_beams = args.num_beams
    add_context = args.add_context
    model_name = args.model
    dataset = args.dataset
    use_original_context = args.use_original_context
    decoding_method = args.decoding_method
    temperature = args.temperature
    top_p = args.top_p
    return

def minimum_args(args):
    global model_name, dataset, add_context, use_original_context
    add_context = args.add_context
    model_name = args.evaluation_model
    dataset = args.dataset
    use_original_context = args.use_original_context

def ensure_reproducibility(SEED=10):
    random.seed(SEED)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    #Fix torch random seed
    torch.manual_seed(SEED)
    return
    
def encode_as_chat(d):
  if dataset == 'dispute':
    replacement = d['obj']
  else:
    replacement = '[ENTITY]'
    
  if add_context:
    system_message = "You'll be given a question and a context about the article and answer it with a one word. Answer the [Question]."
    if not use_original_context:
      if 'Qwen' in model_name:
        chat = [
            {"role": "system", "content": system_message},
            {
            "role": "user",
            "content": "This article is about %s. [Context]: %s [Question]: %s [Answer]:" % (
           d["subj"], d["context"].replace(replacement, d['replace_name'].lower()), d["question"])
        }]
      else:
        chat = [{
            "role": "user",
            "content": "%s This article is about %s. [Context]: %s [Question]: %s [Answer]:" % (system_message,
           d["subj"], d["context"].replace(replacement, d['replace_name'].lower()), d["question"])
        }]
    else:
      if 'Qwen' in model_name:
        chat = [
            {"role": "system", "content": system_message},
            {
            "role": "user",
            "content": "This article is about %s. [Context]: %s [Question]: %s [Answer]:" % (
           d["subj"], d["context"].replace(replacement, d['obj'].lower()), d["question"])
        }]
      else:   
        chat = [{
            "role": "user",
            "content": "%s This article is about %s. [Context]: %s [Question]: %s [Answer]:" % (
            system_message, d["subj"], d["context"].replace(replacement, d['obj']).lower(),  d["question"])
        }]
  else:
    system_message = "You'll be given a question about the article and answer it with a one word. Answer the [Question]."
    if 'Qwen' in model_name:
      chat = [
            {"role": "system", "content": system_message},
            {
            "role": "user",
            "content": "This article is about %s. [Question]: %s [Answer]:" % (
           d["subj"], d["question"])
        }]
    else:   
      chat = [{
            "role": "user",
            "content": "%s This article is about %s. [Question]: %s [Answer]:" % (
            system_message, d["subj"],  d["question"])
        }]

#   chat = [ user ]
  prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, return_tensors="pt")
  if 'Qwen' in model_name:
    inputs = tokenizer([prompt], return_tensors="pt")
  else:
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
  return inputs

def encode_and_format_dataset(dataset):

    dataset = dataset.map(encode_as_chat, batched=False, load_from_cache_file=False)
        
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def get_generations(model, dataloader, number_of_generations):
    """For a given model, produce a number of generation """
    
    period_token_id = tokenizer('. ')['input_ids'][1]
    # eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    # question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]
    squad_metric = evaluate.load("squad")
    rouge = evaluate.load('rouge')
    exact_match_metric = evaluate.load("exact_match")

    with torch.no_grad():
        max_length_of_generated_sequence = 20
        sequences = []
        for batch in tqdm.tqdm(dataloader):

            input_ids =  batch['input_ids'].to(device)
            
            if 'Qwen' in model_name: ### For some reason the tokenization comes out weird
              input_ids = input_ids.reshape(1,-1)
              
            # pdb.set_trace()
            if decoding_method == 'beam_search':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=5,
                                                        num_return_sequences=2,
                                                        do_sample=False,
                                                        max_new_tokens=
                                                        max_length_of_generated_sequence,
                                                        eos_token_id=period_token_id,
                                                        # bad_words_ids=question_framing_ids
                                                        )
            elif decoding_method == 'greedy':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=1,
                                                        do_sample=False,
                                                        max_new_tokens=
                                                        max_length_of_generated_sequence,
                                                        eos_token_id=period_token_id,
                                                        # bad_words_ids=question_framing_ids
                                                        )

            input_length = input_ids.shape[1] if 'Qwen' in model_name else batch['input_ids'].shape[1] #input_ids.shape[1] if config.dataset == 'trivia_qa' else 
            generations = torch.ones((number_of_generations, input_length + max_length_of_generated_sequence),
                                     dtype=torch.long,
                                     device=device)
            for i in range(number_of_generations):

                generation = model.generate(input_ids,
                                            do_sample=True,
                                            num_return_sequences=1,
                                            num_beams=num_beams,
                                            max_new_tokens = max_length_of_generated_sequence,
                                            eos_token_id=period_token_id,
                                            temperature=temperature,
                                            # bad_words_ids=question_framing_ids,
                                            top_p=top_p)
                generations[i, :generation.shape[1]] = generation

            generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
            for i in range(generations.shape[0]):

                sequence_dict = {
                    'prompt': input_ids[0],
                    'generations': generations[i],
                    'id': batch['id'],
                    'question': batch['question']
                }

                generated_texts = []
                for generation in generations[i]:
                    generated_texts.append(
                        tokenizer.decode(generation[len(batch['input_ids'][i]):], skip_special_tokens=True))

                sequence_dict['generated_texts'] = generated_texts
                sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                sequence_dict['most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[0][len(batch['input_ids'][i]):], skip_special_tokens=True)

                sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                sequence_dict['second_most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[1][len(batch['input_ids'][i]):], skip_special_tokens=True)

                sequence_dict['semantic_variability_reference_answers'] = batch[
                    'semantic_variability'] if 'semantic_variability' in batch else None
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
                for rouge_type in rouge_types:
                    if rouge_type in batch:
                        sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]

                    else:
                        sequence_dict[rouge_type + '_reference_answers'] = None

                    sequence_dict[rouge_type + '_to_target'] = 0.0

                sequence_dict['answer'] = batch['obj']
                sequence_dict['additional_answers'] = None

                sequence_dict['exact_match'] = 0.0

                reference_answers = batch['obj']

                for answer in reference_answers:
                    predictions = [sequence_dict['most_likely_generation'].lstrip()]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                         references=references,
                                                         ignore_case=True,
                                                         ignore_punctuation=True)
                    sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    for rouge_type in rouge_types:
                        sequence_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type], #.fmeasure,
                                                                       sequence_dict[rouge_type + '_to_target'])

                sequences.append(sequence_dict)

    return sequences