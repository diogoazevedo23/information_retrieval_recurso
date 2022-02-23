"""
    Recurso Recuperação Informação

    Autores:
    Diogo Azevedo nº 104654 / Ricardo Madureira nº 104624
    25/02/2022
"""

# Imports
import re                   # Regex
import Stemmer              # Stemmer

""" Main Class for the Tokenizer """
class Tokenizer:

    pos_tokens = {}
    pos_tokensArr = []

    def __init__(self, min_tamanho, tokenizer_mode, steemer, stopwords_file):
        self.tokenizer_mode = tokenizer_mode        # tokenizer (a/b)
        self.min_tamanho = min_tamanho              # minimum lenght of token(word)
        self.stopwords_file = stopwords_file        # Stopwords
        self.docsLength = []                        # Length of the documents

        if self.min_tamanho == "no":
            print("Minimo Tamanho palavra == 0")
        else:
            print("Minimo Tamanho palavra == ", self.min_tamanho)

        if self.stopwords_file == "no":
            self.stopwords = []
            print("Stopwords == No")
        elif self.stopwords_file == "yes":
            stopwords_file="files/snowball_stopwords_EN.txt"
            text = open(stopwords_file, 'r')
            self.stopwords = [word.strip() for word in text.readlines()]
            print("Stopwords == yes")
        else:
            try:
                print("sys.argv[4] -->", self.stopwords_file)
                self.stopwords_file = self.stopwords_file
                text = open(self.stopwords_file, 'r')
                self.stopwords = [word.strip() for word in text.readlines()]
                print("Stopwords == yes")
            except IOError:
                print("Error: File does not exist.")
                exit()

        if steemer == "yes":
            self.stemmer = Stemmer.Stemmer('english')
            self.steemWords = steemer

    def tokenize(self, input_string, index):
        final_tokens = []

        tokens = re.sub("[^a-zA-Z]+", " ", input_string).lower().split(" ")

        #if sys.argv[1] == "no":
        if "yes" == "no":
            for token in tokens:
                if len(token) < 0 or token in self.stopwords:
                    continue
                else:
                    final_tokens.append((token, index))
        else:
            tokens = [token for token in tokens if len(token) >= int(self.min_tamanho) and token not in self.stopwords]
            if self.steemWords == "yes":
                tokens = self.stemmer.stemWords(tokens)
            tokens = [(tokens[i],i) for i in range(0,len(tokens))]

        for token in tokens:
            final_tokens.append((token[0],index, token[1]))

        return final_tokens
