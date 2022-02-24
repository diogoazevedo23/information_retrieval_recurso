"""
    Recurso Recuperação Informação

    Autores:
    Diogo Azevedo nº 104654 / Ricardo Madureira nº 104624
    25/02/2022
"""

# Imports
from itertools import combinations
import sys
from Tokenizer import Tokenizer         # Import of Tokenizer
from Merger import Merger               # Import of Merger
import time                             # Time functions
from datetime import datetime           # Time functions
from collections import OrderedDict     # Order dictionaries
import csv                              # Reading CSV files
import psutil                           # Checks memory
import math
from statistics import mean             # Mean for AVGDL
from cmath import sqrt
from turtle import pos

""" Main class """
class mainClass:

    """ Initialize some functions/variables """
    #def __init__(self, min_tamanho, tokenizer_mode, steemer, stopwords_file, chunksize=10000, ranker="tfidf", file='files/teste1.txt'):
    def __init__(self, min_tamanho, tokenizer_mode, steemer, stopwords_file, chunksize=10000, ranker="tfidf", file='files/testFile.tsv'):
        
        self.tokenizer = Tokenizer(min_tamanho, tokenizer_mode, steemer, stopwords_file)
        self.merger = Merger()
        
        self.chunksize = chunksize
        self.ranker = ranker
        self.file = file
        self.docsLength = []        # Length of the documents
        self.avgdl = 0              # Average Document Lenght
        self.N = 0                  # Size of corpus
        self.numberOfBlock = 0      # Number of Block of indexing
        self.indexed_words = {}
        self.dicionario = {}        # Dictionary (Term:IDF)
        self.arrayDocsIds = []      # Array with ID of each Doc
        self.indexDocs = 0          # Index of docs
        self.L = 0                  # Sum of all weights of a doc
        self.combination = ""

        # Questions
        self.indexingTime = 0       # Time taken to index
        self.mergeTime = 0          # Time taken to merge
        self.indexSizeOnDisk = 0    # Disk usage on index

    """ Function to send chunks of data to processing """
    def generateChunks(self, reader):
        chunk = []
        for i, line in enumerate(reader):
            if (i % self.chunksize == 0 and i > 0):
                yield chunk
                del chunk[:]
            chunk.append(line)
        yield chunk

    """ Processing of data with SPIMI implementation """
    def processFiles(self):
        startIndexing = datetime.now()

        with open(self.file, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)    # Reading csv/tsv files
            """ If you want to switch from tsv to csv files, change the  delimiter="\t", to  delimiter="," """
            print("\nReading tsv")

            for chunk in self.generateChunks(reader):
                print("\n\t\t** ITERATION Nº", self.numberOfBlock, "**\n")
                mem = psutil.virtual_memory().available
                print("\nMemory Available ->", mem >> 20, "mb")
                for row in chunk:
                    newTokens = []
                    index = row['review_id']

                    #appended_string = self.combinationIndex(row)

                    appended_string = row['product_title'] + " " + row['review_headline'] + " " + row['review_body']                # Join title, headline and body rows
                    newTokens = self.tokenizer.tokenize(appended_string, self.indexDocs) # Apply tokenizer of the tokens
                    #print("\nnewTokens:", newTokens, "\n")

                    self.calculateDictionarySize()
                    self.criarBlocos(newTokens)

                    cosineValue = round((math.sqrt(self.L)), 4)
                    self.cosineNormalization(self.indexDocs, cosineValue)

                    self.docsLength.append(len(newTokens))
                    newTokens.clear()

                    self.arrayDocsIds.append(index)     # array with id of Docs
                    self.N += 1                         # Size of Corpus
                    self.indexDocs += 1                 # Index of Doc

                print("Writing block")
                self.writeToBlock(self.numberOfBlock)
                self.numberOfBlock += 1
        
        self.indexingTime = (datetime.now() - startIndexing)

        self.writeDocsIds()
        self.avgdl = mean(self.docsLength)
        self.writeDocsLength()
        self.tokenizer.docsLength = []
        self.merger.getN(self.N)                                # Update size of colection in merger function                                        # Release memory

        self.createDicionario()                                 # Dicionary with df that we will turn into idf on merge
        startMerge = datetime.now()
        self.merger.merge_blocks(self.dicionario)               # Merge indexed blocks
        self.mergeTime = (datetime.now() - startMerge)          # Time taken to merge
        self.dicionario = {}                                    # Release memory

    """ Normalization of weights with cosine"""
    def cosineNormalization(self, indexDocs, cosineValue):
        for term in self.indexed_words:
            for doc in self.indexed_words[term]:
                if doc == indexDocs:
                    weight = self.indexed_words[term][indexDocs]['weight']
                    value = weight / cosineValue
                    self.indexed_words[term][indexDocs]['weight'] = value   # TF normalized

    """ Indexing of datachunks that were already processed """
    def criarBlocos(self, tokens):
        
        #print("Indexing block")
        
        for token in tokens:
            term = token[0]
            docID = token[1]
            position = token[2]

            #if sys.argv[6] == 'bm25':
            if term not in self.dicionario.keys():              # Dicionario que irá servir para o IDF
                self.dicionario[term] = str(docID)              # DF = Doc Freq. QTs = P
            else:
                if str(docID) not in self.dicionario[term]:
                    self.dicionario[term] += ("," + str(docID))

            if term not in self.indexed_words.keys():
                self.indexed_words[term] = { docID : { 'weight' : 1 , 'positions' : [position] }}
            else:
                value_dict = self.indexed_words[term]
                if docID not in value_dict.keys():
                    value_dict[docID] = { 'weight' : 1 , 'positions' : [position] }
                else:
                    value_dict[docID]['weight'] += 1
                    value_dict[docID]['positions'].append(position)

                self.indexed_words[term] = value_dict

        #print("self.indexed_words:", self.indexed_words)

        if sys.argv[6] == 'tfidf':          # Normalization of TF
            self.L = 0
            for term in self.indexed_words:
                for doc in self.indexed_words[term]:
                    if doc == docID:                    # It does the TF only for the present doc
                        #print("self.indexed_words[term]:", self.indexed_words[term][docID]['weight'])
                        weight = self.indexed_words[term][docID]['weight']
                        value = round((1 + math.log10(weight)), 4)
                        self.indexed_words[term][docID]['weight'] = value   # TF for word in document

                        #print("value:", value, "| math.pow(value, 2):", math.pow(value, 2))
                        self.L += math.pow(value, 2)    # Sum of all weights of a doc

        #print("self.L:", self.L)
        #print("self.indexed_words:", self.indexed_words)

    """ Sort the chunks that were already processed and indexed and write them to a block """
    def writeToBlock(self, numberOfBlock):
        # i++
        print("\nWriting to block")

        ordered_dict = sorted(self.indexed_words.items(), key = lambda kv: kv[0])
        with open("blocks/block" + str(numberOfBlock) + ".txt",'w+') as f:
            for term, value in ordered_dict:
                string = term + ':' + str(value) + '\n'
                f.write(string)

        self.indexed_words = {}
        print("\nFinished writing block")

    """ Write ID of docs to a text file """
    def writeDocsIds(self):
            with open("extras/idDocs.txt",'w+') as f:
                for id in self.arrayDocsIds:
                    string = id + '\n'
                    f.write(string)

    """ Write lenght of documents(doc:lenght) to text file"""
    def writeDocsLength(self):
        with open("extras/docsLength.txt",'w+') as f:
            string = str(self.avgdl) + '\n'
            f.write(string)
            for line in self.docsLength:
                string = str(line) + '\n'
                f.write(string)

    """ Dicionary with DF of words"""
    def createDicionario(self):
        for x, v in self.dicionario.items():
            self.dicionario[x] = len(v.split(","))

    """ Combinations to index ('product_title' + 'review_headline' + 'review_body') """
    def combinationIndex(self, row):
        if self.combination == "a":
            appended_string = row['product_title'] + " " + row['review_headline'] + " " + row['review_body']
        elif self.combination == "a":
            appended_string = row['product_title'] + " " + row['review_headline']
        elif self.combination == "c":
            appended_string = row['review_headline'] + " " + row['review_body'] 
        elif self.combination == "d":
            appended_string = row['product_title'] + " " + row['review_body']

        return appended_string

    """ We need to calculate in this way, since python only counts the size of keys and not the nested variables """
    def calculateDictionarySize(self):
        for key, value in self.indexed_words.items():
            self.indexSizeOnDisk += sys.getsizeof(value)

        self.indexSizeOnDisk + sys.getsizeof(self.indexed_words)

    """ Answer Questions """
    def answerQuestions(self):
        print("\n\n\t**Questions:**\n") 
        with open("answers/questions.txt", "w") as f:
                f.write("Indexing time (without merge) = {} (hh:mm:ss.ms)" .format(self.indexingTime))
                f.write("\nMerge time (hh:mm:ss.ms) = {} (hh:mm:ss.ms)" .format(self.mergeTime))
                f.write("\nTotal time (hh:mm:ss.ms) = {} (hh:mm:ss.ms)" .format((self.indexingTime + self.mergeTime)))
                f.write("\nTotal index size on disk = %.4f Mb." % (self.indexSizeOnDisk / 1024 / 1024))
                f.write("\nVocabulary size (number of terms) = {} terms" .format(self.merger.newTerm))
                f.write("\nNumber of temporary index segments written to disk (before merging) = {} blocks" .format(self.numberOfBlock))
                f.write("\nAmount of time taken to start up an index searcher = ???")

""" Main """
if __name__ == "__main__":

    # Default Values
    min_tamanho = 3
    tokenizer_mode = "a"
    steemer = "yes"
    stopwords_file = "yes"
    chunksize = 10000
    ranker = "tfidf"

    if len(sys.argv) < 7:
        print("Usage: py Main.py min_tamanho_palavra(no/4) tokenizerMode(a/b) stemmer('yes/no')"
        + "Stopwords('yes/no/filepath') chunksize(3) ranker(tfidf/bm25)"
              
              + "\n** CHOICES **"
              + "\nmin_tamanho_palavra: Can be choosen with a number or desativacted with 'no'"
              + "tokenizerMode(a/b)"
              + "\nstemmer: 'yes' or 'no'"
              + "\nstopwords: 'yes', 'no' or pathfile to the file that u want to use"
              + "\nchunkzise: integer")
        sys.exit(1)

    arrayArgs = [x for x in sys.argv]
    print("arrayArgs:", arrayArgs)
    with open('extras/metadados.txt', 'w') as f:
        for item in arrayArgs:
            f.write("%s\n" % item)

    mainclass = mainClass(min_tamanho, tokenizer_mode, steemer, stopwords_file, chunksize, ranker)

    mainclass.processFiles()
    mainclass.answerQuestions()

