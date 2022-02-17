"""

    Autores:
    Diogo Azevedo nº 104654 / Ricardo Madureira nº 104624
    04/02/2022

"""

# Imports
import sys
import re
from turtle import up
from typing import final    # Regex
import Stemmer              # Stemmer

""" Main Class for the Tokenizer """
class Tokenizer:

    pos_tokens = {}
    pos_tokensArr = []

    def __init__(self, min_tamanho, tokenizer_mode, steemer, stopwords_file):
        self.stemmer = Stemmer.Stemmer('english')
        self.tokenizer_mode = tokenizer_mode
        self.min_tamanho = min_tamanho
        self.docsLength = []                    # Length of the documents

        if sys.argv[1] == "no":
            print("Minimo Tamanho palavra == 0")
        else:
            print("Minimo Tamanho palavra == ", self.min_tamanho)

        if sys.argv[4] == "no":
            self.stopwords = []
            print("Stopwords == No")
        elif sys.argv[4] == "yes":
            stopwords_file="files/snowball_stopwords_EN.txt"
            text = open(stopwords_file, 'r')
            self.stopwords = [word.strip() for word in text.readlines()]
            print("Stopwords == yes")
        else:
            try:
                print("sys.argv[4] -->", sys.argv[4])
                self.stopwords_file = sys.argv[4]
                text = open(self.stopwords_file, 'r')
                self.stopwords = [word.strip() for word in text.readlines()]
                print("Stopwords == yes")
            except IOError:
                print("Error: File does not exist.")
                exit()

        print("Stemmer == ", sys.argv[3])

    def tokenize(self, input_string, index):
        final_tokens = []

        tokens = re.sub("[^a-zA-Z]+", " ", input_string).lower().split(" ")

        #if sys.argv[3] == "yes":
        #    tokens = self.stemmer.stemWords(tokens)

        if sys.argv[1] == "no":
            for token in tokens:
                if len(token) < 0 or token in self.stopwords:
                    continue
                else:
                    final_tokens.append((token, index))
        else:
            tokens = [token for token in tokens if len(token) >= int(self.min_tamanho) and token not in self.stopwords]
            if sys.argv[3] == "yes":
                tokens = self.stemmer.stemWords(tokens)
            tokens = [(tokens[i],i) for i in range(0,len(tokens))]

        #print("\ntokens ->", tokens)

        for token in tokens:
            final_tokens.append((token[0],index, token[1]))
        
        self.docsLength.append(len(final_tokens))

        return final_tokens

    """
    def tokenize2(self, input_string):
        final_tokens = []

        tokens = re.sub("[^a-zA-Z]+", " ", input_string).lower().split(" ")

        if sys.argv[3] == "yes":
            tokens = self.stemmer.stemWords(tokens)

        for token in tokens:
            if len(token) < int(self.min_tamanho) or token in self.stopwords:
                continue
            else:
                final_tokens.append(token)

            final_tokens

        return final_tokens
    """
