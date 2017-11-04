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
    for word_idx, word in enumerate(test_set.wordlist):
        word_X, word_length = test_set.get_item_Xlengths(word_idx)

        # check whether the word is in models dictionary
        if word in models:
            model = models[word]
            try:
                # get the score
                score = model.score(word_X, word_length)
                probabilities.append({word: score})
            except:
                # print("failure on {} by model score".format(word))
                probabilities.append({word: lowest_score})
        else:
            # print("failure: for word {} no model exist".format(word))
            probabilities.append({word: lowest_score})

        # get the best guess
        guess = __get_best_guess__(models, word_X, word_length)
        guesses.append(guess)

        # print("word: {}, score: {}, guess: {}".format(word, score, guess))

    return (probabilities, guesses)

def __get_best_guess__(models: dict, word_X: list, word_length: list):
     # go through the words and select a model with the best score
    best_guess_score = -1000000
    guess = ""
    for train_word, model in models.items():
        try:
            train_word_score = model.score(word_X, word_length)
            if best_guess_score < train_word_score:
                best_guess_score = train_word_score
                guess = train_word
        except:
            # print("failure on train word {} by model score".format(train_word))
            pass

    return guess