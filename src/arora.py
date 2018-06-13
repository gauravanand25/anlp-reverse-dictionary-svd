from data_utils import ReverseDictionary, Crossword, token_ids_to_sentence
from embeddings import Embeddings
import config
import pickle
import tensorflow as tf
import os
import numpy as np
from evaluate import eval_rank
import math
from sklearn.decomposition import TruncatedSVD
from evaluate import eval_rank

class SIF():
    def __init__(self):
        pass

    def compute_pc(self, X, npc=1):
        """
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: component_[i,:] is the i-th pc
        """
        print ("Any Nan?", np.any(np.isnan(X)))
        print ("All finite?", np.all(np.isfinite(X)))
        if np.any(np.isnan(X)) or not np.all(np.isfinite(X)):
            for i in range(X.shape[0]):
                row = X[i]
                if np.any(np.isnan(row)) or not np.all(np.isfinite(row)):
                    print (i, row)
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(X)
        return svd.components_

    def remove_pc(self, X, npc=1):
        """
        Remove the projection on the principal components
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        pc = self.compute_pc(X, npc)
        if npc == 1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX

    def weighted_average(self, word_vec, word_prob, X):
        N, L = X.shape
        _, D = word_vec.shape

        X_vec = word_vec[X]
        assert X_vec.shape == (N, L, D)

        weight_prob_X = word_prob[X]
        assert weight_prob_X.shape == X.shape

        weighted_X = X_vec.reshape(-1, D) * weight_prob_X.reshape(-1, 1)
        weighted_X = weighted_X.reshape(N, L, D)

        sent_vec = np.sum(weighted_X, axis=1)       # (N, D)

        # normalize sent_vec
        # sent_vec /= np.linalg.norm(sent_vec, axis=1).reshape(-1, 1)

        return sent_vec

    # test
    def tests(self, orig_word_vec, word_prob, dataset, cw_dataset, embeddings):
        # for test_type, test_params in config.tests['reverse-dictionary'].iteritems():
        #     const_as = test_params['const_a']
        #     print test_type
        #     for const_a in const_as:
        #         print "\n", const_a
        #         file_name = test_params['file_name']
        #         normalize = test_params['normalize']
        #         word_vec = np.copy(orig_word_vec)
        #
        #         word_inv_prob = dataset.get_unigram_array(word_prob, dataset.word_to_idx, const_a)
        #
        #         data_path = os.path.join(config.data_dir, file_name)
        #         X, y = dataset.data_to_token_ids(data_path)
        #
        #         # TODO normalized word_vec before calling the below function???
        #         if normalize:
        #             word_vec = embeddings.normalize(word_vec)
        #         sent_vec = self.weighted_average(word_vec, word_inv_prob, X)
        #
        #         # normal rank
        #         print "Normal weighted averaging"
        #         eval_rank(sent_vec, word_vec, y)
        #
        #         if test_params['num_pc']:
        #             # sent_vec = self.remove_pc(sent_vec, test_params['num_pc'])
        #             concat_vec = self.remove_pc(np.vstack((word_vec[2:], sent_vec)), test_params['num_pc'])
        #             word_vec[2:], sent_vec = concat_vec[:word_vec.shape[0]-2], concat_vec[word_vec.shape[0]-2:]
        #
        #         # svd rank
        #         print "SVD ranks"
        #         eval_rank(sent_vec, word_vec, y)
        # print "\n################"
        rev_vocab = dataset.idx_to_word

        for test_type, test_params in config.tests['crossword'].iteritems():
            const_as = test_params['const_a']
            for const_a in const_as:
                file_name = test_params['file_name']
                normalize = test_params['normalize']
                word_vec = np.copy(orig_word_vec)

                word_inv_prob = dataset.get_unigram_array(word_prob, dataset.word_to_idx, const_a)

                data_path = os.path.join(config.data_dir, file_name)
                X, y, dig = cw_dataset.data_to_token_ids(data_path)

                word_vec = np.copy(orig_word_vec)
                word_inv_prob = dataset.get_unigram_array(word_prob, dataset.word_to_idx, const_a)
                sent_vec = self.weighted_average(word_vec, word_inv_prob, X)

                print "Normal weighted averaging"
                idx_rank, candidate_ids = eval_rank(sent_vec, word_vec, y, idx_to_word=dataset.idx_to_word, lengths=dig)

                if test_params['num_pc']:
                    # sent_vec = self.remove_pc(sent_vec, test_params['num_pc'])
                    concat_vec = self.remove_pc(np.vstack((word_vec[2:], sent_vec)), test_params['num_pc'])
                    word_vec[2:], sent_vec = concat_vec[:word_vec.shape[0]-2], concat_vec[word_vec.shape[0]-2:]

                print "SVD ranks"
                # svd_idx_rank, svd_candidate_ids = eval_rank(sent_vec, word_vec, y)
                #
                # top = 5
                # for idx, svd_rank in svd_idx_rank.items():
                #     # both got right
                #     if 0 <= svd_rank < 2 and idx in idx_rank and 2 > idx_rank[idx] >= 0:
                #         print("# both got right # correct word:", rev_vocab[y[idx]])
                #         print(token_ids_to_sentence(X[idx], rev_vocab))  # sentence
                #
                #         candidates = [rev_vocab[word_idx] for word_idx in candidate_ids[idx][:top]]
                #         print("\n Top 5 candidates from the SIF model:")
                #         for ii in range(5):
                #             print("%s: %s" % (ii + 1, candidates[ii]))
                #
                #         svd_candidates = [rev_vocab[word_idx] for word_idx in svd_candidate_ids[idx][:top]]  #
                #         print("\n Top 5 candidates from the SIF + SVD model:")
                #         for ii in range(5):
                #             print("%s: %s" % (ii + 1, svd_candidates[ii]))
                #         print("\n")
                #
                #     # svd got correct but without svd bad
                #     if 0 <= svd_rank < 2 and idx in idx_rank and 4 > idx_rank[idx] >= 2:
                #         print("# SVD better # correct word:", rev_vocab[y[idx]])
                #         print(token_ids_to_sentence(X[idx], rev_vocab))  # sentence
                #
                #         candidates = [rev_vocab[word_idx] for word_idx in candidate_ids[idx][:top]]
                #         print("\n Top 5 candidates from the SIF model:")
                #         for ii in range(5):
                #             print("%s: %s" % (ii + 1, candidates[ii]))
                #
                #         svd_candidates = [rev_vocab[word_idx] for word_idx in svd_candidate_ids[idx][:top]]  #
                #         print("\n Top 5 candidates from the SIF + SVD model:")
                #         for ii in range(5):
                #             print("%s: %s" % (ii + 1, svd_candidates[ii]))
                #         print("\n")

            print "\n################"

    def main(self):
        dataset = ReverseDictionary()
        dataset.load_embeddings_vocab()

        cw = Crossword(dataset)
        self.tests(dataset.word_vec, dataset.word_prob, dataset, cw, dataset.word_vec)

if __name__ == '__main__':
    sif = SIF()
    sif.main()