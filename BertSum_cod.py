#Importing packages
import pandas as pd
from summarizer import Summarizer
model = Summarizer()

df=pd.read_csv(".../.../covid_dataset.csv")

#summarizing the given document
for i in df.url_content:
  result = model(i, min_length=30,max_length=300)
  summary = "".join(result)
  print(summary)  
  
  
from __future__ import division
from itertools import chain


def get_unigram_count(tokens):
    count_dict = dict()
    for t in tokens:
        if t in count_dict:
            count_dict[t] += 1
        else:
            count_dict[t] = 1

    return count_dict


class Rouge:
    beta = 1

    @staticmethod
    def my_lcs_grid(x, y):
        n = len(x)
        m = len(y)

        table = [[0 for i in range(m + 1)] for j in range(n + 1)]

        for j in range(m + 1):
            for i in range(n + 1):
                if i == 0 or j == 0:
                    cell = (0, 'e')
                elif x[i - 1] == y[j - 1]:
                    cell = (table[i - 1][j - 1][0] + 1, '\\')
                else:
                    over = table[i - 1][j][0]
                    left = table[i][j - 1][0]

                    if left < over:
                        cell = (over, '^')
                    else:
                        cell = (left, '<')

                table[i][j] = cell

        return table

    @staticmethod
    def my_lcs(x, y, mask_x):
        table = Rouge.my_lcs_grid(x, y)
        i = len(x)
        j = len(y)

        while i > 0 and j > 0:
            move = table[i][j][1]
            if move == '\\':
                mask_x[i - 1] = 1
                i -= 1
                j -= 1
            elif move == '^':
                i -= 1
            elif move == '<':
                j -= 1

        return mask_x

    @staticmethod
    def rouge_l(cand_sents, ref_sents):
        lcs_scores = 0.0
        cand_unigrams = get_unigram_count(chain(*cand_sents))
        ref_unigrams = get_unigram_count(chain(*ref_sents))
        for cand_sent in cand_sents:
            cand_token_mask = [0 for t in cand_sent]
            cand_len = len(cand_sent)
            for ref_sent in ref_sents:
                # aligns = []
                # Rouge.lcs(ref_sent, cand_sent, aligns)
                Rouge.my_lcs(cand_sent, ref_sent, cand_token_mask)

                # for i in aligns:
                #     ref_token_mask[i] = 1
            # lcs = []
            cur_lcs_score = 0.0
            for i in range(cand_len):
                if cand_token_mask[i]:
                    token = cand_sent[i]
                    if cand_unigrams[token] > 0 and ref_unigrams[token] > 0:
                        cand_unigrams[token] -= 1
                        ref_unigrams[token] -= 1
                        cur_lcs_score += 1

            lcs_scores += cur_lcs_score

        # print "lcs_scores: %d" % lcs_scores
        ref_words_count = sum(len(s) for s in ref_sents)
        # print "ref_words_count: %d" % ref_words_count
        cand_words_count = sum(len(s) for s in cand_sents)
        # print "cand_words_count: %d" % cand_words_count

        precision = lcs_scores / cand_words_count
        recall = lcs_scores / ref_words_count
        f_score = (1 + Rouge.beta ** 2) * precision * recall / (recall +
                                                                Rouge.beta ** 2 * precision + 1e-7) + 1e-6  # prevent underflow
        return precision, recall, f_score
        
        
r = Rouge()
for i range (0,10):
  [precision, recall, f_score] = r.rouge_l([temp[i]], [df.gold_sum[i])
  print("Precision is :"+str(precision)+"\nRecall is :"+str(recall)+"\nF Score is :"+str(f_score))

##ROUGE-1, ROUGE-2, ROUGE-L
from rouge_score import rouge_scorer
for i in range(0,10):
  print("Summary[",i,"]")
  print("Original Document:",df.url_content[i])
  print("Gold Summary:", df.gold_sum[i])
  print("Predicted summary:", temp[i])
  print("\n")
  print("*********ROUGE-1***********")
  rouge_1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
  rouge_1_scores = rouge_1.score(temp[i], df.gold_sum[i])
  print(rouge_1_scores)
  print("\n")
  print("*********ROUGE-2***********")
  rouge_2 = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
  rouge_2_scores = rouge_2.score(temp[i], df.gold_sum[i])
  print(rouge_2_scores)
  print("\n")
  print("*********ROUGE-L***********")
  rouge_L = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
  rouge_L_scores = rouge_L.score(temp[i], df.gold_sum[i])
  print(rouge_L_scores)
  #print("Precision is :"+str(precision)+"\nRecall is :"+str(recall)+"\nF Score is :"+str(f_score))
  print("-----------------------------------------------------------------------")
