
class Ngram():
    """One Ngram object stores the id of the ngram (string) and
     the log-likelihood of the ngram.
    """

    def __init__(self, id: str, loglk: float):
        self.id = id
        self.loglk = loglk

    def getDict(self):
        return {
            "id":self.id,
            "loglk":self.loglk
        }

    def __str__(self):
        return "Ngram: id={}".format(self.id) + " loglk={}".format(self.loglk)

    @staticmethod
    def getTestNgram():
        return Ngram("abc", -12.34)

if __name__ == "__main__":
    testNgram = Ngram.getTestNgram()
    print(testNgram)

