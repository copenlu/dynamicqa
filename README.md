
# DYNAMICQA

This is a repository for the paper [DYNAMICQA: Tracing Internal Knowledge Conflicts in Language Models](https://arxiv.org/abs/2407.17023) accepted at Findings of EMNLP 2024.

<p align="center">
  <img src="main_figure.png" width="650" alt="main_figure">
</p>

Our paper investigates the Language Model's behaviour when the conflicting knowledge exist within the LM's parameters. We present a novel dataset containing inherently conflicting data, DYNAMICQA. Our dataset consists of three partitions, **Static**, **Disputable** 🤷‍♀️, and **Temporal** 🕰️.

We also evaluate several measures on their ability to reflect the presence of intra-memory conflict: **Semantic Entropy** and a novel **Coherent Persuasion Score**. You can find our findings from our paper!


## Dataset

Our dataset is available under /Data folder.
You can also load our dataset using [huggingface datasets](https://huggingface.co/datasets/copenlu/dynamicqa). 

| Partition | Number of Questions |
| --------- | ------------------- |
| Static   | 2500 |
| Temporal | 2495 |
| Disputable | 694 |


## Code
Our code base provides the implementation of Semantic Entropy and the proposed Coherent Persuasion Score.

### Inference
You can obtain the most likely generation for each model using `final_code/inference.py`.

```bash
python inference_answer.py \
	--model_name meta-llama/Llama-2-7b-chat-hf \ 
	--mode temporal \
	--bit4 True \
	--context True
```
`model_name` accepts the huggingface path to the chosen LLM. `mode` refers to the dataset subset and can be either: `static`, `temporal` or `dispute`. `bit4` accepts boolean values and determines if the model will be run in 4bit or 8 bit precision. `context` accepts boolean values and determines if the model will be queried with or without context.

A full list of options are given as follows:

```bash
usage: inference_answer.py [-h] --model_name MODELNAME --mode MODE --bit4 IS_BIT4 --context USE_CONTEXT [--seed SEED]

options:
  -h, --help            show this help message and exit
  --model_name MODELNAME     Path to the huggingface model for inference
  --mode  MODE               Dataset subset for inference. Valid values are: 'static', 'temporal' or 'dispute'. 
  --bit4  IS_BIT4            Boolean value indicating if experiment run in 4 bit or 8 bit
  --context USE_CONTEXT      Boolean value indicating if model should be queried with or without context
  --seed SEED                Random seed
```

The output is a JSON file to `Output/[MODE]/[MODE]_[MODELNAME]_persuasion_output_[START_NUM].json` containing model generations. Average model accuracy (in comparison to the golden answers) will be printed to the screen.

### Coherent Persuasion score
You can calculate the CP score using `final_code/robust_persuasion.py`. This process generates several samples with high temperature, which is used to calculate both Coherent Persuasion and Semantic Entropy.

```bash
python robust_persuasion.py \
	--model_name meta-llama/Llama-2-7b-chat-hf \ 
	--mode temporal \
	--bit4 True \
	--context True
```
`model_name` accepts the huggingface path to the chosen LLM. `mode` refers to the dataset subset and can be either: `static`, `temporal` or `dispute`.

A full list of options are given as follows:

```bash
usage: robust_persuasion.py [-h] --model_name MODELNAME --mode MODE --bit4 IS_BIT4 --context USE_CONTEXT [--num_gen_sample NUMGEN] [--temperature TEMPERATURE] [--num_beams NUMBEAMS] [--top_p TOP_P] [--seed SEED] [--start_num START_NUM]

options:
  -h, --help            show this help message and exit
  --model_name MODELNAME     Path to the huggingface model for inference
  --mode  MODE               Dataset subset for inference. Valid values are: 'static', 'temporal' or 'dispute'. 
  --bit4  IS_BIT4            Boolean value indicating if experiment run in 4 bit or 8 bit
  --context USE_CONTEXT      Boolean value indicating if model should be queried with or without context
  --num_gen_sample NUMGEN    Number of samples to generate. Defaults to 10.
  --temperature TEMPERATURE  Temperature for sample generation. Defaults to 0.5
  --num_beams NUMBEAMS       Number of beams to run. Defaults to 1.
  --top_p TOP_P              Top p to use for generation. Defaults to 1.0
  --seed SEED                Random seed
  --start_num START_NUM      Index number of dataset to start generation.
```
The average CP score for the entire subset is printed to the screen. The output is to `Output/[MODE]/[MODE]_[MODELNAME]_persuasion_output_[START_NUM].pkl` with the following information:

- id_csv: Index available in the dataset file
- answer: The intended answer
- id: Numeric index of datapoint
- answers_q_only: Generated answers given only the question
- answers_c_and_q: Generated answers given the context and the question.
- clusters_q_only: The indexes of semantically clustered answers given only the question
- clusters_c_and_q: The indexes of semantically clustered answers given the context and question
- kl_div_cm_conflict: The KL divergence between clusters_q_only and clusters_c_and_q
- persuasion_score: Our final persuasion score, averaged across clusters.

**Stubborn** and **persuaded** instances are identified using `final_code\get_stubborn_persuaded_instances.py`. When this program is run it identifies stubborn and persuaded instance for all modes (static, temporal and dispute) and models (Llama2, Mistral and Qwen2).

It outputs the CP score of Stubborn instances to `Output/[MODE]_[MODELNAME]_persuasion_stubborn.pkl` and Persuaded instances to `Output/[MODE]_[MODELNAME]_persuasion_persuaded.pkl`. All other datapoints recieve a CP score of -1.

### Semantic Entropy

You can calculate Semantic Entropy using `final_code/get_likelihoods.py` and `final_code/compute_confidence_measure.py` which have been adapted from [Kuhn et al (2023)](https://github.com/lorenzkuhn/semantic_uncertainty)

```bash
python get_likelihoods.py \
  --evaluation_model meta-llama/Llama-2-7b-chat-hf \ 
  --dataset temporal \
  --add_context True \
```

```bash
python compute_confidence_measure.py \
  --generation_model meta-llama/Llama-2-7b-chat-hf \ 
  --dataset temporal \ 
  --add_context True 
```
A full list of options are given as follows:

```bash
usage: get_likelihoods.py [-h] --evaluation_model MODELNAME --dataset MODE --context USE_CONTEXT --use_original_context USE_OG [--start_num START_NUM]

options:
  -h, --help                      show this help message and exit
  --evaluation_model MODELNAME    Path to the huggingface model for inference
  --dataset  MODE                 Dataset subset for inference. Valid values are: 'static', 'temporal' or 'dispute'. 
  --add_context USE_CONTEXT       Boolean value indicating if model should be queried with or without context
  --use_original_context USE_OG   Boolean value indicating which of the two contexts should be used for querying
  --start_num START_NUM           Index number of dataset to start generation.
```

```bash
usage: compute_confidence_measure.py [-h] --generation_model MODELNAME --dataset MODE --context USE_CONTEXT --use_original_context USE_OG [--verbose IS_VERBOSE]

options:
  -h, --help                      show this help message and exit
  --generation_model MODELNAME    Path to the huggingface model for inference
  --dataset  MODE                 Dataset subset for inference. Valid values are: 'static', 'temporal' or 'dispute'. 
  --add_context USE_CONTEXT       Boolean value indicating if model should be queried with or without context
  --use_original_context USE_OG   Boolean value indicating which of the two contexts should be used for querying
  --verbose IS_VERBOSE            Boolean value indicating if SE value should be printed out.
```

The average normalized SE will be printed to screen. The total final output is to `Output/[MODELNAME]_[MODE]_[USE_CONTEXT]_[CONTEXT_TYPE].pkl` where context_type varies between `q_only` if context is not provided and `q_and_c` if context is provided.

The utilized SE is the column `predictive_entropy_over_concepts`

## Citation
If you use our code or dataset, kindly cite it using
```
@inproceedings{marjanović2024dynamicqatracinginternalknowledge,
      title={DYNAMICQA: Tracing Internal Knowledge Conflicts in Language Models}, 
      author={Sara Vera Marjanović and Haeun Yu and Pepa Atanasova and Maria Maistro and Christina Lioma and Isabelle Augenstein},
      year={2024},
      booktitle = {Findings of EMNLP},
      publisher = {Association for Computational Linguistics}
}
```
