
import pickle
import json
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
import pdb

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def load_reference(mode):
    if mode == "static":
        data = pd.read_csv("Data/Static_final.csv", index_col=0)
    elif mode == "temporal":
        data = pd.read_csv("Data/Temporal_final.csv", index_col=0)
    elif mode == "dispute":
        data = pd.read_csv("Data/Disputable_subset.csv", index_col=0)

    return data


def find_stubborn_example(model_name, mode="static"):

    # wrong with context but the same answer with the parametric knowledge
    context_file = "Output/sequences/%s/%s_%s_context.json" %(mode,mode, model_name)
    q_only_file = "Output/sequences/%s/%s_%s_q_only.json" %(mode,mode, model_name)

    context_answer_list = json.load(open(context_file))
    q_only_answer_list = json.load(open(q_only_file))

    data = load_reference(mode)

    stubborn_num = 0
    acc = []
    stubborn_list = []
    persuaded_list = []
    contradict_correct = []
    for idx, context_answer in enumerate(context_answer_list):
        gold_idx = int(idx//2)

        if int(idx%2) == 0:
            # original answer : obj
            gold_answer = data.iloc[gold_idx]["obj"]

        elif int(idx%2) == 1:
            # replace_name : replace_name
            gold_answer = data.iloc[gold_idx]["replace_name"]

        context_answer = clean_str(context_answer)

        context_score = evaluate(scorer, context_answer, gold_answer)

        acc.append(context_score)

        q_only_answer = clean_str(q_only_answer_list[idx])

        # incorrect
        if not context_score:

            c_and_q = evaluate(scorer, context_answer, q_only_answer)

            # if c_and_q != 0.0:
                # print(c_and_q, context_answer, q_only_answer)

            if c_and_q == 1.0:
                # print(c_and_q, context_answer, q_only_answer)
                stubborn_num += 1
                stubborn_list.append(idx)

        # correct
        else:
            # persuaded
            persuaded_list.append(idx)

            # contradict correct
            # check if q_only answer is correct
            q_only_score = evaluate(scorer, context_answer, q_only_answer)
            #
            if q_only_score == 0.0:
                contradict_correct.append(idx)

    print(len(acc))
    print("Average:", np.mean(acc))

    return stubborn_list, persuaded_list, contradict_correct


def clean_str(word):
    new_word = word.lower().replace("assistant:", "").strip()
    new_word = new_word.replace("answer", "")
    new_word = new_word.replace(":", "")

    return new_word


def evaluate(scorer, pred_str, gold_str):

    score_dict = scorer.score(pred_str, gold_str)

    tmp_f1 = score_dict["rougeL"].fmeasure

    if tmp_f1 > 0.3:
        return 1.0
    else:
        return 0.0


def persuasion_analysis(model_name, stub_list, pers_list, mode="static"):
    result = pickle.load(open("Output/sequences/%s/%s_%s_persuasion_output.pkl" % (mode, mode, model_name), "rb"))
    print(len(result))
    data = load_reference(mode)

    acc = []
    persuaded_scores = []
    stubborn_scores = []
    correct_stubborn = 0
    for idx, r in enumerate(result):
        if int(idx%2) == 1:
            if idx in stub_list:
                stubborn_scores.append(r["persuasion_score"])
            else:
                stubborn_scores.append(-1)

            if idx in pers_list:
                persuaded_scores.append(r["persuasion_score"])
            else:
                persuaded_scores.append(-1)

    return stubborn_scores, persuaded_scores




def check_cluster_correct(answer_list, cluster_info, gold_answer):
    members = [int(i) for i in cluster_info.split("-")]

    score_list = []
    for m in members:
        score = evaluate(scorer, answer_list[m], gold_answer)
        score_list.append(score)

    if sum(score_list) == 0.0:
        return False

    if int(sum(score_list)) >= (len(score_list)//2):
        return True

    else:
        return False





if __name__ == "__main__":

    # correlation
    # model_list = ["Mistral"]

    # for model_name in model_list:
    #     correlation_uncertainty(model_name, "static")


    # stubborn example count
    model_list = {"Qwen2":"Qwen2-7B-Instruct", "Llama":"Llama-2-7b-chat-hf", "Mistral":"Mistral-7B-Instruct-v0.1"}

    for short_name, long_name in model_list.items():
        for mode in ["temporal", "static", "dispute"]:
            print(long_name, "-", mode)
            # get indices
            stubborn_list, persuaded_list, contradict_correct = find_stubborn_example(long_name, mode=mode)

            # get scores from indices
            # if you want score list of contradict_correct,
            # do
            # stub_pers, pers_pers = persuasion_analysis(short_name, stubborn_list, contradict_correct, mode=mode)

            stub_pers, pers_pers = persuasion_analysis(long_name, stubborn_list, persuaded_list, mode=mode)
            print("\n# of Stubborn", len(stubborn_list))
            print("Persuasion Score on Stubborn", np.mean(stub_pers))
            pickle.dump(stub_pers, open("Output/final/%s_%s_persuasion_stubborn.pkl" % (mode, short_name), "wb"))

            print("\n# of Persuaded", len(persuaded_list))
            print("Persuasion Score on Persuaded", np.mean(pers_pers))
            pickle.dump(pers_pers, open("Output/final/%s_%s_persuasion_persuaded.pkl" % (mode, short_name), "wb"))
            print()
