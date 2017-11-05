import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # lists to return
    probabilities = []
    guesses = []

    # constants for error cases
    lowest_score = -1000000

    # iterate over all test words
    for word_idx, _ in enumerate(test_set.wordlist):
        word_X, word_length = test_set.get_item_Xlengths(word_idx)
        current_probs = {}

        # go through the models, select a word, which model gives a best score
        # and store this word as a guess
        best_guess_score = -1000000
        guess = ""
        for train_word, model in models.items():
            try:
                # get the score
                train_word_score = model.score(word_X, word_length)

                # add score to probabilities dictionary
                current_probs[train_word] = train_word_score

                # change best guess if appropriate
                if best_guess_score < train_word_score:
                    best_guess_score = train_word_score
                    guess = train_word
            except:
                # print("failure on train word {} by model score".format(train_word))
                current_probs[train_word] = lowest_score

        # get the best guess
        guesses.append(guess)
        probabilities.append(current_probs)

        # print("word: {}, score: {}, guess: {}".format(word, score, guess))
    return (probabilities, guesses)
