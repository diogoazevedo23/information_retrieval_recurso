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
        self.min_tamanho = min_tamanho              # minimum lenght of token
        self.stopwords_file = stopwords_file        # Stopwords

        if self.min_tamanho == "no":            # If "no", min_tamanho == 0
            self.min_tamanho = 0

        print("Minimo Tamanho palavra ==", self.min_tamanho)

        if self.stopwords_file == "no":         # Dont use stopword
            self.stopwords = []
            print("Stopwords == No")
        elif self.stopwords_file == "yes":      # Use default stopword list
            stopwords_file="files/snowball_stopwords_EN.txt"
            text = open(stopwords_file, 'r')
            self.stopwords = [word.strip() for word in text.readlines()]
            print("Stopwords == yes")
        else:                                   # Try to open stopword inputted by user
            try:
                print("sys.argv[4] -->", self.stopwords_file)
                self.stopwords_file = self.stopwords_file
                text = open(self.stopwords_file, 'r')
                self.stopwords = [word.strip() for word in text.readlines()]
                print("Stopwords == yes")
            except IOError:
                print("Error: File does not exist.")
                exit()

        if steemer == "yes":                    # Use steemer
            self.stemmer = Stemmer.Stemmer('english')
            self.steemWords = steemer

    def tokenize(self, input_string, index):
        final_tokens = []

        if self.tokenizer_mode == "a":
            tokens = re.sub("[^a-zA-Z]+", " ", input_string).lower().split(" ")         # Only Alpha chars
            tokens = [token for token in tokens if len(token) >= int(self.min_tamanho) and token not in self.stopwords]

            if self.steemWords == "yes":
                tokens = self.stemmer.stemWords(tokens)
            tokens = [(tokens[i],i) for i in range(0,len(tokens))]

        elif self.tokenizer_mode == "b":
            tokens = re.sub("[^a-zA-Z0-9-_]+", " ", input_string).lower().split(" ")   # Only Alphanumeric chars, dash and underscores
            tokens = [token for token in tokens if len(token) >= int(self.min_tamanho) and token not in self.stopwords]

            if self.steemWords == "yes":
                tokens = self.stemmer.stemWords(tokens)
            tokens = [(tokens[i],i) for i in range(0,len(tokens))]

        for token in tokens:
            final_tokens.append((token[0],index, token[1]))

        return final_tokens
