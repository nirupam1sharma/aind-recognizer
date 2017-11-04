import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, 
                                    verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states by model fit".format(
                    self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def __calculate_num_params__(self, n_components, n_features):
        '''
        Calculates a number of free parameters for HMM for BIC calculation
        See https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/4
        for details

        :param n_components: number of components (states) of the model
        :param n_features: number of features
        :return: calculated number of free parameters of the HMM
        '''
        return (n_components ** 2) + (2 * n_components * n_features)

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_BIC_score = 1000000
        bestModel = None
        num_data_points = len(self.X)
        log_num_data_points = np.log(num_data_points)

        # try different number of components
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            # create the model on the train set
            model = self.base_model(n_components)

            # check model for validity
            if model is not None:
                try:
                    # calculate the model score
                    logL = model.score(self.X, self.lengths)

                    # calculate BIC
                    num_params = self.__calculate_num_params__(n_components, model.n_features)
                    BIC_score = -2 * logL + num_params * log_num_data_points

                    # update a best model if we achieve a best BIC score
                    if best_BIC_score > BIC_score:
                        best_BIC_score = BIC_score
                        bestModel = model
                        if self.verbose:
                            print("best model for {} so far with {} states. BIC score is {}".format(
                                self.this_word, n_components, best_BIC_score))
                except:
                    if self.verbose:
                        print("failure on {} with {} states by model score".format(
                            self.this_word, n_components))

        return bestModel

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        """ select the best model for self.this_word based on
        DIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_DIC_score = -1000000
        bestModel = None

        # try different number of components
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            # create the model on the train set
            model = self.base_model(n_components)

            # check model for validity
            if model is not None:
                try:
                    # calculate the model score
                    logL = model.score(self.X, self.lengths)

                    # calculate scores for all words but not the current word
                    word_count = 0
                    sum_logL = 0
                    for word, Xlength in self.hwords.items():
                        if word != self.this_word:
                            try:
                                sum_logL += model.score(Xlength)
                                word_count += 1
                            except:
                                if self.verbose:
                                    print("failure on {} with {} states by model score".format(
                                        word, n_components))

                    # calculate DIC score
                    DIC_score = logL - sum_logL/(word_count - 1) if word_count > 1 else logL

                    # update a best model if we achieve a best DIC score
                    if best_DIC_score < DIC_score:
                        best_DIC_score = DIC_score
                        bestModel = model
                        if self.verbose:
                            print("best model for {} so far with {} states. DIC score is {}".format(
                                self.this_word, n_components, best_DIC_score))
                except:
                    if self.verbose:
                        print("failure on {} with {} states by model score".format(
                            self.this_word, n_components))

        return bestModel


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def __model__(self, n_components, X_train, lengths_train):
        '''
        Trains a gaussian HMM with specified parameters
        :param n_components: number of components (states) in the model
        :param X_train: train set of X as used in hmmlearn
        :param lengths_train: train set of lengths as used in hmmlearn
        :return: GausianHMM object or None if the model cannot be fitted
        '''
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                n_iter=1000,
                random_state=self.random_state,
                verbose=False).fit(X_train, lengths_train)

            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, n_components))
            return hmm_model

        except:
            if self.verbose:
                print("failure on {} with {} states by model fit".format(
                    self.this_word, n_components))
            return None

    def select(self):
        '''
        Selects best model based on average log Likelihood of cross-validation folds
        for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        '''
        # it is not a good practice to ignore warnings, but for the purpose of the exercise,
        # it is okay here
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # number of splits for k-fold. since we do not have too much training data for each word,
        # 3 is a reasonable choice
        no_of_splits = 3

        # initialize variables
        bestLogL = -1000000
        bestModel = None

        # try different number of components
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            seq_len = len(self.sequences)

            # average model score and number of created models
            avg_logL = 0
            model_count = 0

            # if we have less than 3 sequences for a word, we cannot do k-folding
            if seq_len < no_of_splits:
                for idx in range(seq_len):
                    # create the model on the train set
                    X_train, lengths_train = combine_sequences([idx], self.sequences)
                    model = self.__model__(n_components, X_train, lengths_train)

                     # check model for validity
                    if model is not None:
                        try:
                            # calculate the model score on the test set
                            idx_test = (idx + 1) % seq_len
                            X_test, lengths_test = combine_sequences([idx_test], self.sequences)
                            avg_logL += model.score(X_test, lengths_test)
                            model_count += 1
                        except:
                            if self.verbose:
                                print("failure on {} with {} states by model score".format(
                                    self.this_word, n_components))

            else:
                # use KFold split word sequences into train and test indices
                split_method = KFold(n_splits=no_of_splits)

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # create the model on the train set
                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    model = self.__model__(n_components, X_train, lengths_train)

                    # check model for validity
                    if model is not None:
                        try:
                            # calculate the model score on the test set
                            X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                            avg_logL += model.score(X_test, lengths_test)
                            model_count += 1
                        except:
                            if self.verbose:
                                print("failure on {} with {} states by model score".format(
                                    self.this_word, n_components))

            # check if we got any valid model
            if model_count > 0:
                # update average score
                avg_logL /= model_count

                # update a best model if we achieve a best score
                if bestLogL < avg_logL:
                    bestLogL = avg_logL
                    bestModel = self.__model__(n_components, self.X, self.lengths)
                    if self.verbose:
                        print("best model for {} so far with {} states".format(
                            self.this_word, n_components))

        return bestModel
