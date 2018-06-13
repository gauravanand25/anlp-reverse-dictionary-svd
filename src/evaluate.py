import collections
import numpy as np
import math


def eval_rank(sent_vec, word_vec, y, idx_to_word=None, lengths=None):
    sent_to_word = sent_vec.dot(word_vec.T)  # (N, V)

    # calculate rank
    N, D = sent_vec.shape
    idx_rank = {}
    candidate_ids = sent_to_word.argsort(axis=1)[:, ::-1]
    for i in range(N):
        rank = np.where(candidate_ids[i] == y[i])[0][0]
        if lengths is not None and idx_to_word is not None:     #cross word
            cnt = 0
            for j in range(rank):
                if len(idx_to_word[candidate_ids[i][j]]) == lengths[i]:
                    candidate_ids[i][cnt] = candidate_ids[i][j]
                    # print idx_to_word[candidate_ids[i][j]], idx_to_word[y[i]]
                    cnt += 1

            rank = cnt
            print ("\n")

        if rank < 1000:
            idx_rank[i] = rank
            if lengths is not None and idx_to_word is not None and rank < 3:     #cross word
                cnt = 0
                for j in range(150):
                    if len(idx_to_word[candidate_ids[i][j]]) == lengths[i]:
                        candidate_ids[i][cnt] = candidate_ids[i][j]
                        print idx_to_word[candidate_ids[i][j]], idx_to_word[y[i]]
                        cnt += 1
                        if cnt > 5:
                            break
                rank = cnt
                print ("\n")

    # idx_rank = {k: v for (k, v) in idx_rank.items() if v < 1000}
    ranks = idx_rank.values()
    compute_median_variance(ranks)
    return idx_rank, candidate_ids

def compute_median_variance(ranks):
    print ("median ", np.median(ranks))
    top10 = sum(rank < 10 for rank in ranks)
    top100 = sum(rank < 100 for rank in ranks)
    print ("accuracy 10/100 ", 100 * top10 / float(len(ranks)), 100 * top100 / float(len(ranks)))
    print ("var ", math.sqrt(np.var(ranks)))


