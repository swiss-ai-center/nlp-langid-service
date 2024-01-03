from model.Ngram import Ngram
import numpy as np
import pickle
from typing import Dict


class Language():
    """
    A Language object stores a language id and a dictionary of ngram
    (keys are ngram ids as strings, values are ngram objects). Language
    id is recommended to be iso 639-1 codes, i.e. 2 letters code such as
    fr for French. The language id can also be the iso 639-1 code followed
    by the 2 letter country code to identify dialect, e.g. de-CH for
    german spoken in Switzerland. No verifications are done to verify this.
    The description field is the plain english version of the iso code.
    """

    def __init__(self, id: str, description: str = "EMPTY_DESCRIPTION"):
        self.id = id
        self.description = description
        self.ngrams = {}
        self.default_log_lk_value = -100.0

    def addNgram(self, ngram: Ngram):
        self.ngrams[ngram.id] = ngram

    def getNgram(self, ngramDesc: str) -> Ngram:
        if ngramDesc in self.ngrams:
            return self.ngrams[ngramDesc]
        else:
            return None

    def getLoglkNgram(self, ngramDesc: str) -> float:
        if ngramDesc in self.ngrams:
            return self.ngrams[ngramDesc].loglk
        else:
            return self.default_log_lk_value  # pretty small log likelihood if not found

    def compute_default_log_lk_value(self, verbosity: int = 0):
        '''compute the default log likelihood value as the smallest value in the ngrams'''
        smallest_log_lk = np.min([ngram.loglk for ngram in self.ngrams.values()])
        self.default_log_lk_value = smallest_log_lk
        if verbosity > 1:
            print("Default for {} = {}".format(self.id, self.default_log_lk_value))

    def save_language_model(self, path: str = '', vervosity: int = 1):
        filename = path + '.p'
        if vervosity >= 1:
            print('Saving language in file: ', filename)
        pickle.dump(self, open(filename, "wb"))

    def getDict(self) -> Dict:
        return {
            "langid": self.id,
            "description": self.description
        }

    def is_dialect(self) -> bool:
        '''
        Determine wether the language is a dialect or not looking
        if the id has the form of 'aa-BB', i.e. 5 letters overall
        and a minus sign in the middle. The left part of the minus
        sign is expected to be the 2 letters language iso code and
        the right part is the 2 letter country code, representing the
        origin of the dialect.
        :return:
        '''
        if len(self.id) == 5:
            if self.id[2] == '-':
                return True
        return False

    def __str__(self):
        return "Language: id={} ".format(self.id) + \
            "description={} ".format(self.description) + \
            "nbr_ngrams={}".format(len(self.ngrams))

    @staticmethod
    def read_language_model(filename: str, verbosity: int = 0):
        if not filename.endswith('.p'):
            filename = filename + '.p'
        if verbosity > 1:
            print("Reading language model from : {}".format(filename))
        language = pickle.load(open(filename, "rb"))
        return language

    @staticmethod
    def getTestLanguage():
        testLanguage = Language("aa-BB")
        testLanguage.addNgram(Ngram("abc", -12.34))
        testLanguage.addNgram(Ngram("bcd", -11.23))
        testLanguage.addNgram(Ngram("cde", -10.12))
        return testLanguage


if __name__ == "__main__":
    test_language = Language.getTestLanguage()
    print(test_language)
    print(test_language.getNgram("abc"))
    print(test_language.getLoglkNgram("abc"))
    print(test_language.getNgram("xyz"))
    print(test_language.getLoglkNgram("xyz"))
    print('Is dialect: ', test_language.is_dialect())
    test_language.save_language_model(test_language.id)
    stored_test_language = Language.read_language_model(test_language.id)
    print(stored_test_language)
    print(stored_test_language.getNgram("abc"))
    print(stored_test_language.getLoglkNgram("abc"))
    print(stored_test_language.getNgram("xyz"))
    print(stored_test_language.getLoglkNgram("xyz"))
