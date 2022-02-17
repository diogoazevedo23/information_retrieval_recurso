"""
    Recurso Recuperação Informação

    Autores:
    Diogo Azevedo nº 104654 / Ricardo Madureira nº 104624
    25/02/2022

"""

# Imports
from turtle import pos
from Tokenizer import Tokenizer         # Import of Tokenizer
from Merger import Merger               # Import of Merger
import json                             # Json Functions
import time                             # Time functions
import sys
from collections import OrderedDict     # Order dictionaries
import csv                              # Reading CSV files
import psutil                           # Checks memory
import math
from statistics import mean             # Mean for AVGDL


""" Main class """
class mainClass:

    """ Initialize some functions/variables """
    def __init__(self, min_tamanho, tokenizer_mode, steemer, stopwords_file, chunksize=10000, ranker = "tfidf", file='files/teste1.txt'):
    #def __init__(self, min_tamanho, tokenizer_mode, steemer, stopwords_file, chunksize=10000, ranker = "tfidf", file='files/amazon_reviews_us_Digital_Video_Games_v1_00.tsv'):
        
        self.tokenizer = Tokenizer(min_tamanho, tokenizer_mode, steemer, stopwords_file)
        self.merger = Merger()
        
        self.chunksize = chunksize
        self.ranker = ranker
        self.file = file
        #self.docsLength = {}        # Length of the documents
        self.N = 0                  # Size of corpus
        self.numberOfBlock = 0      # Number of Block of indexing
        self.indexed_words = {}
        self.dicionario = {}        # Dictionary (Term:IDF)
        self.arrayDocsIds = []      # Array with ID of each Doc
        self.indexDocs = 0              # Index of docs

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

        with open(self.file, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)    # Reading csv/tsv files
            """ If you want to switch from tsv to csv files, change the  delimiter="\t", to  delimiter="," """
            print("\nReading tsv")

            for chunk in self.generateChunks(reader):
                print("\n\t\t** ITERATION Nº", self.numberOfBlock, "**\n")
                tokens = []                                                             # Words
                mem = psutil.virtual_memory().available
                print("\nMemory Available ->", mem >> 20, "mb")
                for row in chunk:
                    index = row['review_id']
                    appended_string = row['product_title'] + " " + \
                        row['review_headline'] + " " + row['review_body']               # Join title, headline and body rows
                    tokens += self.tokenizer.tokenize(appended_string, self.indexDocs)           # Apply tokenizer of the tokens

                    self.arrayDocsIds.append(index)
                    #self.docsLength[index] = (len(tokens))
                    #print("self.docsLength:", self.docsLength)
                    self.N += 1             # Size of Corpus
                    self.indexDocs += 1     # Index of Doc

                self.criarBlocos(tokens)
                print("Writing block")
                self.writeToBlock(self.numberOfBlock)
                self.numberOfBlock += 1
            
        #self.writeDicionario()
        #self.readDicionario()
        avgdl = mean(self.tokenizer.docsLength)
        print("avgdl:", avgdl)
        self.writeDocsLength()
        self.tokenizer.docsLength = []
        self.merger.updateColSize(self.N)                       # Update size of colection in merger function
        tokens = []                                             # Release memory

        self.createDicionario()
        self.merger.merge_blocks(self.dicionario)               # Merge indexed blocks
        self.dicionario = {}                                    # Release memory

    """ Indexing of datachunks that were already processed """
    def criarBlocos(self, tokens):
        
        print("Indexing block")
        
        for token in tokens:
            term = token[0]
            docID = token[1]
            position = token[2]

            if term not in self.dicionario.keys():          # Dicionario que irá servir para o IDF
                self.dicionario[term] = str(docID)               # DF = Doc Freq. QTs = P
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

        print("\n*DOING TF*")

        if sys.argv[6] == 'tfidf':
            for term in self.indexed_words:
                for docID in self.indexed_words[term]:
                    #print("self.indexed_words[term]:", self.indexed_words[term][docID]['weight'])
                    weight = self.indexed_words[term][docID]['weight']
                    self.indexed_words[term][docID]['weight'] = round((1 + math.log10(weight)), 4)   # TF for word in document

        for x, v in self.indexed_words.items():
            print(x, v)

    """ Sort the chunks that were already processed and indexed and write them to a block """
    def writeToBlock(self, numberOfBlock):

        print("\nWriting to block")

        ordered_dict = sorted(self.indexed_words.items(), key = lambda kv: kv[0])
        with open("blocks/block" + str(numberOfBlock) + ".txt",'w+') as f:
            for term, value in ordered_dict:
                string = term + ':' + str(value) + '\n'
                f.write(string)

        self.indexed_words = {}
        print("\nFinished writing block")

    """ Write lenght of documents(doc:lenght) to text file"""
    def writeDocsLength(self):
        with open("extras/docsLength.txt",'w+') as f:
            for line in self.tokenizer.docsLength:
                string = str(line) + '\n'
                f.write(string)

    def createDicionario(self):
        for x, v in self.dicionario.items():
            self.dicionario[x] = len(v.split(","))
        #return self.dicionario

    """ Load the dicitionary(term:idf) to memory """
    def readDicionario(self):
        with open("extras/dicionario.txt",'r') as f:
            loadDicionario = dict([line.split() for line in f])

        print("\n\nDicitonary ->", loadDicionario, "\n")


""" Main """
if __name__ == "__main__":

    if len(sys.argv) < 7:
        print("Usage: py Main.py min_tamanho_palavra(no/4) tokenizerMode(a/b) stemmer('yes/no')"
        + "Stopwords('yes/no/filepath') chunksize(4) ranker(tfidf/bm25)"
              
              + "\n** CHOICES **"
              + "\nmin_tamanho_palavra: Can be choosen with a number or desativacted with 'no'"
              + "tokenizerMode(a/b)"
              + "\nstemmer: 'yes' or 'no'"
              + "\nstopwords: 'yes', 'no' or pathfile to the file that u want to use"
              + "\nchunkzise: integer")
        sys.exit(1)

    mainClass = mainClass(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])

    totalIndexingTimeStart = time.time()
    mainClass.processFiles()

    """
    print("\n\nStarging to Merge")
    mergeStartTime = time.time()
    mainClass.pos_index()
    mergeEndTime = time.time()
    print("\nFinished Merging ==", mergeEndTime - mergeStartTime, "segundos")
    """
    totalIndexingTimeEnd = time.time()
    totalIndexingTimeFinal = totalIndexingTimeEnd - totalIndexingTimeStart
    print("Indexing Time ->", totalIndexingTimeFinal)

    """
    mainClass.printToFile(totalIndexingTimeFinal, mainClass.sizeOfDictInGb(), mainClass.pos_index2(), mainClass.BlockFilesNumber)
    #mainClass.saveTF(mainClass.len_doc)
    #mainClass.saveIDF()
    """