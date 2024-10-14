from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
import torch
import argparse
import re, scipy, pickle
import json, pdb
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

import pdb

# transformers 4.39.1
# torch '2.2.1+cu121'

class cmConflict():
    def __init__(self, args):
        self.args = args

        self.load_model()
        self.data = self.load_data()

        self.load_mnli_model()


    def load_model(self):
        if self.args.bit4:
            config = self.bit4_config()
            if "Phi" in self.args.model_name:
                model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=config,
                                                             device_map="auto", trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=config, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name).to(self.args.device)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")

        self.model = model
        self.tokenizer = tokenizer
    # end of load_model


    def load_mnli_model(self):
        self.mnli_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
        self.mnli_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")

        self.mnli_model.to(self.args.device)
    # load_mnli_model


    def bit4_config(self):
        if "mistral" in self.args.model_name:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )

        else:
            quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

        return quant_config


    def load_data(self):
        if self.args.mode == "dispute":
            data = pd.read_csv("../Data/Disputable_final.csv", index_col=0)

        elif self.args.mode == "temporal":
            data = pd.read_csv("../Data/Temporal_final.csv", index_col=0)

        elif self.args.mode == "static":
            data = pd.read_csv("../Data/Static_final.csv", index_col=0)

        return data


    def run(self):
        total_result = []
        total_pers = []

        selected_indices = [i for i in range(args.start_num, len(self.data))]

        for i in tqdm(range(len(self.data))):
            if i in selected_indices:
                line = self.data.iloc[i]
            else:
                continue
            answer_list = [line["obj"], line["replace_name"]]

            for answer in answer_list:
                encoded_q_only = self.prepare_input(line, answer, context=False)
                encoded_c_and_q = self.prepare_input(line, answer, context=True)

                gen_q_only, probs_q_only = self.generate_multiple_samples(encoded_q_only)
                gen_c_and_q, probs_c_and_q = self.generate_multiple_samples(encoded_c_and_q)

                clusters_q_only = self.evaluate_semantic_sim(gen_q_only)
                clusters_c_and_q = self.evaluate_semantic_sim(gen_c_and_q)

                # calculate kl divergence
                # persuasion_score is the list of scores between clusters (context) and (q_only)
                # number of combination
                pers_score = self.get_persuasion_score(clusters_q_only, clusters_c_and_q, probs_q_only, probs_c_and_q)

                # what to save
                # answers from each
                # clusters : correctness

                # semantic similarity between q_only and c_and_q clusters?
                # we can do this if I save cluster info and answers

                to_save = {
                    "id_csv": line["id"],
                    "answer": answer,
                    "id": i,
                    "answers_q_only": gen_q_only,
                    "answers_c_and_q": gen_c_and_q,
                    "clusters_q_only": clusters_q_only,
                    "clusters_c_and_q": clusters_c_and_q,
                    "kl_div_cm_conflict": pers_score,
                    "persuasion_score": np.mean(pers_score)
                }
                total_result.append(to_save)
                total_pers.append(np.mean(pers_score))
                if len(total_result) < 5:
                    print(to_save)

                if len(total_result) % 100 == 0:
                    print("Saved at %d" %(len(total_result)))
                    filename = "%s_%s_persuasion_output_%d.pkl" % (self.args.mode, self.args.model_name.split("/")[-1], self.args.start_num)
                    pickle.dump(total_result, open(filename, "wb"))

        print("Total persuasion score: %.4f" %(np.mean(total_pers)))

        return total_result


    def get_persuasion_score(self, clusters_q_only, clusters_c_and_q, probs_q_only, probs_c_and_q):
        probs_a_list = []
        for cluster_a in clusters_c_and_q:
            probs_a = self.obtain_rep_probs(cluster_a, probs_c_and_q)
            probs_a_list.append(probs_a)

        probs_b_list = []
        for cluster_b in clusters_q_only:
            probs_b = self.obtain_rep_probs(cluster_b, probs_q_only)
            probs_b_list.append(probs_b)

        persuasion_score = []
        for pa in probs_a_list:
            for pb in probs_b_list:
                persuasion_per = self.calculate_kl_div(pa, pb)
                persuasion_score.append(persuasion_per)

        return persuasion_score


    def obtain_rep_probs(self, cluster, prob_list):
        members = cluster.split("-")

        rep_probs = []
        for mem in members:
            rep_probs.append(prob_list[int(mem)])

        return np.mean(rep_probs, axis=0)

    def calculate_kl_div(self, prob_1, prob_2):
        # KL(P || Q) : P's divergence from Q
        # it's not symmetric / KL(P || Q) != KL(Q || P)

        # here, context distribution's divergence from question only divergence
        # how different when the context is given

        # https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python
        vec = scipy.special.rel_entr(prob_1, prob_2)
        vec = np.ma.masked_invalid(vec).compressed()
        kl_div_score = np.sum(vec)

        return kl_div_score

    def evaluate_semantic_sim(self, generated_samples):
        entail_info = {idx:[] for idx in range(len(generated_samples))}

        for i, answer_1 in enumerate(generated_samples):
            for j in range(i+1, len(generated_samples)):
                pred = self.get_mnli_prediction(generated_samples[i], generated_samples[j])
                reverse_pred = self.get_mnli_prediction(generated_samples[j], generated_samples[i])

                # entailment is 2
                if pred == 2 and reverse_pred == 2:
                    entail_info[i].append(j)
                    entail_info[j].append(i)

        cluster_list = []
        for idx, idx_entails in entail_info.items():
            cluster = [idx]
            for ent in idx_entails:
                if idx in entail_info[ent]:
                    flag = False
                    for other_member in cluster:
                        if other_member in entail_info[ent] and ent in entail_info[other_member]:
                            flag = True
                        else:
                            flag = False
                    if flag:
                        cluster.append(ent)
            cluster_str = [str(mem) for mem in sorted(cluster)]
            cluster_list.append("-".join(cluster_str))

        return list(set(cluster_list))


    def get_mnli_prediction(self, answer1, answer2):
        input_str = answer1 + ' [SEP] ' + answer2
        encoded_input = self.mnli_tokenizer(input_str, return_tensors="pt").to(self.args.device)
        prediction = self.mnli_model(input_ids=encoded_input["input_ids"],
                                     token_type_ids=encoded_input["token_type_ids"]).logits
        predicted_label = torch.argmax(prediction, dim=1).item()

        return predicted_label


    def generate_multiple_samples(self, encoded_input):

        output_list = []
        vocab_prob_list = []
        for i in range(args.num_gen_sample):
            result_dict = self.model.generate(
                encoded_input,
                do_sample=True,
                num_return_sequences=1,
                num_beams=self.args.num_beams,
                max_new_tokens=20,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                return_dict_in_generate=True,
                output_logits=True,
                output_hidden_states=False
            )

            decoded = self.tokenizer.batch_decode(result_dict["sequences"])

            if "Mistral" in self.args.model_name:
                result = re.search(".*\[\/INST\](.*)</s>", decoded[0])

            elif "gemma" in self.args.model_name:
                result = decoded[0].split("<end_of_turn>")[-1].replace("<eos>", "")

            elif "falcon" in self.args.model_name:
                result = decoded[0].split(self.tokenizer.eos_token)[1]
                result = result.replace("\n", "")

            elif "Qwen" in self.args.model_name:
                result = decoded[0].split("<|im_start|>")[-1]
                result = result.replace("<|im_end|>", "")
                result = result.replace("Assistant:", "")
                result = re.search("\s?(.*)$", result)

            elif "llama" in self.args.model_name:
                result = re.search(".*\[\/INST\](.*)</s>", decoded[0])


            if not result:
                answer = ""
            elif isinstance(result, str):
                answer = result.strip()
            else:
                answer = result.groups()[0].strip()

            gen_word_logits = torch.stack([prob.squeeze(0) for prob in result_dict["logits"]])
            gen_word_probs = torch.nn.functional.softmax(gen_word_logits, dim=1)

            # here we do avg over vocab distributions
            avg_probs = torch.mean(gen_word_probs, dim=0).detach().cpu().numpy()

            output_list.append(answer)
            vocab_prob_list.append(avg_probs)

        return output_list, vocab_prob_list



    def prepare_input(self, line, answer, context=False):
        ctx = line["context"].replace("[ENTITY]", answer)

        if self.args.mode == "dispute":
            ctx = line["context"].replace(line["obj"], answer)
            article_info = "This article is about %s" % line["subj"]
            ctx = article_info + " " + ctx
        else:
            ctx = line["context"].replace("[ENTITY]", answer)
        if context:
            if "Qwen" in self.args.model_name:
                system = {
                    "role": "system",
                    "content": "You'll be given a question and a context about the article and answer it with a one word. Answer the [Question]."
                }
                user = {
                    "role": "user",
                    "content": "[Context]: %s [Question]: %s [Answer]:" % (
                        ctx, line["question"])
                }
            else:
                user = {
                    "role": "user",
                    "content": "You'll be given a question and a context about the article and answer it with a one word. Answer the [Question]. [Context]: %s [Question]: %s [Answer]:" % (
                        ctx, line["question"])
                }
        else:
            if "Qwen" in self.args.model_name:
                system = {
                    "role": "system",
                    "content": "You'll be given a question about the article and answer it with a one word. Answer the [Question]. "
                }
                user = {
                    "role": "user",
                    "content": "This article is about %s. [Question]: %s [Answer]:" % (
                        line["subj"], line["question"])
                }
            else:
                user = {
                    "role": "user",
                    "content": "You'll be given a question about the article and answer it with a one word. Answer the [Question]. This article is about %s. [Question]: %s [Answer]:" % (
                    line["subj"], line["question"])
                }

        if "Qwen" in self.args.model_name:
            input_msgs = [system, user]
        else:
            input_msgs = [user]

        if "falcon" in self.args.model_name:
            input_str = user["content"] + self.tokenizer.eos_token
            encoded = self.tokenizer(input_str, return_tensors="pt")
            encoded = encoded["input_ids"].to(self.args.device)
        else:
            encoded = self.tokenizer.apply_chat_template(input_msgs, return_tensors="pt").to(self.args.device)
        return encoded

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    else:
        print('No GPU available, using the CPU instead.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model : "mistralai/Mistral-7B-Instruct-v0.1" , meta-llama/Meta-Llama-Guard-2-8B , meta-llama/Meta-Llama-3-8B
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--bit4", action="store_true")
    parser.add_argument("--mode", type=str, default="temporal")
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--num_gen_sample", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--start_num", type=int, default=0)
    args = parser.parse_args()
    print(args)

    set_seed(args)

    module = cmConflict(args)

    results_list = module.run()

    filename = "../Output/%s/%s_%s_persuasion_output_%d.pkl" %(args.mode,args.mode, args.model_name.split("/")[-1], args.start_num)
    pickle.dump(results_list, open(filename, "wb"))
