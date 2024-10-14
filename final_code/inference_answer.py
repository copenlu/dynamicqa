from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import argparse
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
import pdb
import random

class inference():
    def __init__(self, args):
        self.args = args
        self.load_model()
        self.data = self.load_data()

        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


    def load_model(self):
        if self.args.bit4:
            config = self.bit4_config()
            model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=config, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name).to(self.args.device)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")

        self.model = model
        self.tokenizer = tokenizer
    # end of load_model

    def bit4_config(self):
        if "mistral" in self.args.model_name:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
        # elif "falcon" in self.args.model_name:
        else:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
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
        # if self.args.mode == "dispute":
        #     output_list = self.run_with_json()
        #
        # elif self.args.mode in ["static", "temporal"]:
        output_list = self.run_with_csv()

        # to save output
        if self.args.context:
            filename = "../Output/%s/%s_%s_context.json" % (self.args.mode,self.args.mode, self.args.model_name.split("/")[-1])
        else:
            filename = "../Output/%s/%s_%s_q_only.json" % (self.args.mode,self.args.mode, self.args.model_name.split("/")[-1])

        json.dump(output_list, open(filename, "w"), indent=4)



    def run_with_json(self):
        output_list = []
        for d in tqdm(self.data):
            for idx, ctx in enumerate(d["contexts"]):
                # only use first question
                q = d["questions"][0]
                if q == "":
                    q = d["questions"][1]

                if self.args.context:
                    user = {
                        "role": "user",
                        "content": "You'll be given a question and a context about the article and answer it with a one word. Answer the [Question]. This article is about %s. [Context]: %s [Question]: %s [Answer]:" % (
                        d["subject"], ctx, q)
                    }
                else:
                    user = {
                        "role": "user",
                        "content": "You'll be given a question about the article and answer it with a one word. Answer the [Question]. This article is about %s. [Question]: %s [Answer]:" %(d["subject"], q)
                    }

                input_msgs = [user]

                encoded = self.tokenizer.apply_chat_template(input_msgs, return_tensors="pt").to(self.args.device)
                generated_ids = self.model.generate(encoded, max_new_tokens=64, do_sample=True)
                decoded = self.tokenizer.batch_decode(generated_ids)
                result = re.search(".*\[\/INST\](.*)</s>", decoded[0])
                if not result:
                    answer = ""
                else:
                    answer = result.groups()[0].strip()
                output_list.append(answer)

        return output_list


    def run_with_csv(self):
        output_list = []
        score_list = []
        golden_list = []
        for i in tqdm(range(len(self.data))):
            line = self.data.iloc[i]

            answer_list = [line["obj"], line["replace_name"]]

            for gold_answer in answer_list:
                # context = line["context"].replace("[ENTITY]", gold_answer)

                if self.args.mode == "dispute":
                    context = line["context"].replace(line["obj"], gold_answer)
                    article_info = "This article is about %s" %line["subj"]
                    context = article_info + " " + context
                else:
                    context = line["context"].replace("[ENTITY]", gold_answer)

                if self.args.context:
                    if "Qwen" in self.args.model_name or "llama" in self.args.model_name:
                        system = {
                            "role": "system",
                            "content": "You'll be given a question and a context about the article and answer it with a one word. Answer the [Question]."
                        }
                        user = {
                            "role": "user",
                            "content": "[Context]: %s [Question]: %s [Answer]:" % (
                                context, line["question"])
                        }
                    else:
                        user = {
                            "role": "user",
                            "content": "You'll be given a question and a context about the article and answer it with a one word. Answer the [Question]. [Context]: %s [Question]: %s [Answer]:" % (
                            context, line["question"])
                        }
                else:
                    if "Qwen" in self.args.model_name or "llama" in self.args.model_name:
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

                if "Qwen" in self.args.model_name or "llama" in self.args.model_name:
                    input_msgs = [system, user]
                else:
                    input_msgs = [user]

                if "gpt" in self.args.model_name:
                    input_str = user["content"] + self.tokenizer.eos_token
                    encoded = self.tokenizer(input_str, return_tensors="pt")
                    encoded = encoded["input_ids"].to(self.args.device)
                else:
                    encoded = self.tokenizer.apply_chat_template(input_msgs, return_tensors="pt").to(self.args.device)
                generated_ids = self.model.generate(encoded, max_new_tokens=20, do_sample=True)
                decoded = self.tokenizer.batch_decode(generated_ids)

                if "mistral" in self.args.model_name:
                    result = re.search(".*\[\/INST\](.*)</s>", decoded[0])

                elif "falcon" in self.args.model_name:
                    result = decoded[0].split(self.tokenizer.eos_token)[1]
                    result = result.replace("\n", "")

                # elif "gpt" in self.args.model_name:
                #     pdb.set_trace()
                #     result = decoded[0]

                elif "Qwen" in self.args.model_name:
                    result = decoded[0].split("<|im_start|>")[-1]
                    result = result.replace("<|im_end|>", "")
                    result = re.search("\s?(.*)$", result)

                elif "llama" in self.args.model_name:
                    result = re.search(".*\[\/INST\](.*)</s>", decoded[0])


                if not result:
                    answer = ""
                elif isinstance(result, str):
                    answer = result.strip()
                else:
                    answer = result.groups()[0].strip()

                output_list.append(answer)
                golden_list.append(gold_answer)
                if i < 10:
                    print(answer)
                    print(gold_answer)

                score = self.evaluate(answer, gold_answer)
                score_list.append(score)

        print(self.args)
        print("Accuracy:", np.mean(score_list))

        return output_list


    def evaluate(self, pred_str, gold_str):
        score_dict = self.scorer.score(pred_str, gold_str)
        tmp_f1 = score_dict["rougeL"].fmeasure

        if tmp_f1 > 0.3:
            return 1.0
        else:
            return 0.0


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
    parser.add_argument("--mode", type=str, default="dispute")
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--seed", type=int, default=10)

    args = parser.parse_args()

    set_seed(args)
    # print(args)


    module = inference(args)

    module.run()


