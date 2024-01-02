from model.Ngram import Ngram
from model.Language import Language
import numpy as np
from typing import List, Dict

class Languages():
    """
    A Languages object stores a set of languages in a dictionary
    (keys are language str descriptions, values are Language objects).
    """

    def __init__(self):
        self.languages = {}

    def addLanguage(self, language : Language):
        self.languages[language.id] = language

    def get_language_ids(self) -> List[str]:
        return list(self.languages.keys())

    def get_language(self, lang_id : str) -> Language :
        return self.languages[lang_id]

    def get_loglk_ngram(self, langDesc : str, ngramDesc : str):
        if langDesc in self.languages:
            return self.languages[langDesc].getLoglkNgram(ngramDesc)

    def get_logllk_phrase(self, phrase: str, n: int = 3, lang_list: List[str] = None, activate_dialects: bool = False,
                          verbosity: int = 0) -> Dict[str, float]:
        # cut into ngrams
        ngrams = Languages.phrase_to_ngram(phrase, n)
        log_llk = {}
        for language in self.languages.values():
            # if current language is a dialect and if activate_dialects is False then we skip this language
            if language.is_dialect() and activate_dialects is False:
                continue
            # we compute for this language only if no list of lang is given
            # or if this language is in the passed lang_list
            if lang_list is None or language.id in lang_list:
                if verbosity > 2 :
                    print("[{}]: ".format(language.id))
                log_llk[language.id] = 0.0
                for ngram in ngrams:
                    loglk_ngram = self.get_loglk_ngram(language.id, ngram)
                    if verbosity > 2:
                        print("({}, {})".format(ngram, loglk_ngram))
                    log_llk[language.id] += loglk_ngram
        # normalise the log_llk by dividing with the length of ngrams in phrase
        scores = np.array(list(log_llk.values()))
        scores = scores / len(ngrams)
        # compute the softmax on the log_lk values to return probabilities
        # this is probably not correct mathematically but at least we have a final score between 0.0 and 1.0
        scores = Languages.softmax(scores)
        i = 0
        for language in self.languages.values():
            # same as above, check if current language is a dialect and if activate_dialects is False
            if language.is_dialect() and activate_dialects is False:
                continue
            # same as above, it could be that we have a limited list of languages
            if lang_list is None or language.id in lang_list:
                log_llk[language.id] = scores[i]
                i += 1
        return log_llk

    def get_winner_lang_id(self, scores : Dict[str, float]):
        tuple_list = [(value, key) for key, value in scores.items()]
        return max(tuple_list)[1]

    def add_language_from_file(self, filename : str, verbosity : int = 0) -> Language:
        language = Language.read_language_model(filename, verbosity)
        self.addLanguage(language)
        if verbosity > 0:
            print(language)
        return language

    def __str__(self):
        return "Languages: nbr_lang={}".format(len(self.languages))

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def getTestLanguages():
        testLanguages = Languages()
        testLanguage1 = Language("testlang1")
        testLanguage1.addNgram(Ngram("abc", -12.34))
        testLanguage1.addNgram(Ngram("bcd", -11.23))
        testLanguage1.addNgram(Ngram("cde", -10.12))
        testLanguages.addLanguage(testLanguage1)
        testLanguage2 = Language("testlang2")
        testLanguage2.addNgram(Ngram("abc", -13.45))
        testLanguage2.addNgram(Ngram("bcd", -14.56))
        testLanguage2.addNgram(Ngram("cde", -15.67))
        testLanguages.addLanguage(testLanguage2)
        return testLanguages
    
    @staticmethod
    def phrase_to_ngram(phrase : str, n : int) -> List[str]:
        """Returns a list of strings result of the slicing into
        sub-strings of length n. For example, phrase 'I eat the banana.'
        is converted with n=3 into ['I e', ' ea', 'eat', ..., 'na.'].

        :param phrase: The phrase to be sliced up.
        :param n: The length of n-gram
        :return: A list of strings where each string has length n
        """
        return [phrase[i:i + n] for i in range(0, len(phrase) + 1 - n)]


if __name__ == "__main__":
    testLanguages = Languages.getTestLanguages()
    print(testLanguages)
    print(testLanguages.get_loglk_ngram("testlang1", "abc"))
    print(testLanguages.get_loglk_ngram("testlang2", "abc"))
    print(testLanguages.get_logllk_phrase("abcd"))
    print(testLanguages.get_winner_lang_id(testLanguages.get_logllk_phrase("abcd")))