"""
    Recurso Recuperação Informação

    Autores:
    Diogo Azevedo nº 104654 / Ricardo Madureira nº 104624
    25/02/2022
"""

# Imports
from Tokenizer import Tokenizer         # Import of Tokenizer
import ast
import os
import re
import math
import itertools        # Slice dictionary for top results
from datetime import datetime
import statistics

class search:

    def __init__(self):

        self.avgdl = 0
        self.metadados = []
        self.docsLength = []
        self.dicionario = {}
        self.scoreDoc = {}
        self.L = 0              # Sum of all weights of a doc
        self.queryTimes = []

        with open("extras/metadados.txt",'r') as f:             # Load Metadados
            self.metadados = [line[:-1] for line in f][1:]

        #print("self.metadados:", self.metadados)

        with open("files/queries.txt",'r') as f:               # Load Queries
            self.queries = [line[:-1] for line in f]

        print("self.queries:", self.queries)

        self.indexBlocks = []
        for file in os.listdir("finalBlocks2/"):                 # Save block index for search
            #print("file:", file)
            removeTxt = file.split('.')[0]
            #print("removeTxt:", removeTxt)
            first, last = removeTxt.split('_')
            #print(first, "|", last)
            self.indexBlocks.append(first + "_" + last)

        #print("\nself.indexBlocks:", self.indexBlocks, "\n")

        min_tamanho = self.metadados[0]
        tokenizer_mode = self.metadados[1]
        steemer = self.metadados[2]
        stopwords_file = self.metadados[3]
        self.ranker = self.metadados[5]                         # tfidf / bm25

        self.tokenizer = Tokenizer(min_tamanho, tokenizer_mode, steemer, stopwords_file)

        self.cleanQueries = {}
        self.topResults = {}

    """ loadToMem """
    def loadToMem(self):

        #if self.ranker == "bm25":
        with open("extras/docsLength.txt",'r') as f:
            self.avgdl = f.readline()[:-1]                       # Load avgdl
            self.docsLength = [line[:-1] for line in f]         # Load dl

            #print("avgdl:", self.avgdl)
            #print("docsLength:", self.docsLength)

        with open("extras/dicionario.txt",'r') as f:
            self.dicionario = dict([line.split() for line in f])    # Load dicionary(IDF)

        #print("dicionario:", self.dicionario)

    """ loadQueries """
    def loadQueries(self, k1, b):

        for query in self.queries:
            queryStartTime = datetime.now()
            tfQuery = {}
            newTokens = self.tokenizer.tokenize(query, 1)
            #print("\nnewTokens:", newTokens, "\n")

            print("\nRanker:", self.ranker)
            if self.ranker == "tfidf":                      # If TF-IDF
            #if self.ranker == "bm25":

                for key in newTokens:                       # TF word
                    word = key[0]
                    if word not in tfQuery:
                        tfQuery[word] = 1
                    else:
                        tfQuery[word] += 1

                #print("tfQuery:", tfQuery)

                for tf in tfQuery:                          # Log(TF) | TF*IDF word
                    #print("TF:", tf)
                    tfNorm = (1 + math.log10(tfQuery[tf]))
                    if tf in self.dicionario.keys():
                        idf = self.dicionario[tf]
                        value = round((tfNorm * float(idf)),4)
                        tfQuery[tf] = value

                        self.L += math.pow(value, 2)        # Sum of all weights of a doc
                        #print("self.L:", self.L)

                cosineValue = round((math.sqrt(self.L)), 4) # Cosine Value

                for tf in tfQuery:                          # Normalization with cosine
                    tfQuery[tf] /= cosineValue

                for term in newTokens:                      # Open right file | get tf values | lnc*ltc
                    word = term[0]
                    #print("\n", word)

                    for doc in self.indexBlocks:
                        first, last = doc.split("_")
                        #print(first, "|", last)
                        if word >= first and word <= last:
                            tfidfDocs = {}
                            #print(word, "Its betwen", first, "and", last)
                            with open("finalBlocks2/" + first + "_" + last + ".txt", "r") as f:
                                #print("File:", f)
                                for line in f.readlines():
                                    #print("line:", line)
                                    term_file,value = re.split(':', line.rstrip('\n'), maxsplit=1)
                                    if term_file == word:
                                        tfidfDocs[term_file] = ast.literal_eval(value)

                            #print("\ntfidfDocs:", tfidfDocs, "\n")
                                    
                            for key, value in tfidfDocs[word].items():
                                tf_doc = value['weight']
                                #print("word:", word, "|", tfQuery[word], "|", tf_doc)

                                newScore = tfQuery[word] * tf_doc
                                #print("newScore:", newScore)

                                if key not in self.scoreDoc:
                                    self.scoreDoc[key] = newScore
                                else:
                                    self.scoreDoc[key] += newScore

            elif self.ranker == "bm25":                     # If BM25
            #elif self.ranker == "tfidf":
                print("bm25")

                for term in newTokens:  # Open right file | get tf values | BM25
                    word = term[0]

                    print("\nTerm:", word, "\n")

                    for doc in self.indexBlocks:
                        first, last = doc.split("_")
                        if word >= first and word <= last:
                            tfidfDocs = {}
                            with open("finalBlocks2/" + first + "_" + last + ".txt", "r") as f:
                                #print("File:", f)
                                for line in f.readlines():
                                    term_file,value = re.split(':', line.rstrip('\n'), maxsplit=1)
                                    if term_file == word:
                                        tfidfDocs[term_file] = ast.literal_eval(value)

                            #print("\ntfidfDocs:", tfidfDocs, "\n")
                                    
                            for key, value in tfidfDocs[word].items():
                                #print("key:", key, "| value:", value, "| value['weight']:", value['weight'])
                                tf_doc = value['weight']             # TF
                                #print("tf_doc", tf_doc)
                                numerator = int(((k1 + 1) * tf_doc))
                                dl = int(self.docsLength[key])
                                #print("word", word, "| key:", key, "| value", value, "| dl:", dl)
                                denominator = ((k1 * ((1 - b) + (b*(dl/float(self.avgdl))))) + tf_doc)
                                bm25Doc = (float(self.dicionario[word]) * (numerator / denominator))     # IDF is in dicionry
                                #print("Word:", word, "| tf_doc:", tf_doc, "| Numerator:", numerator, "| denominator:", denominator, "| bm25Doc:", bm25Doc)

                                # print("bm25Doc:", bm25Doc)

                                if key not in self.scoreDoc:
                                    self.scoreDoc[key] = bm25Doc
                                else:
                                    self.scoreDoc[key] += bm25Doc

            # ------------------------------------

            queryTime = (datetime.now() - queryStartTime)
            self.queryTimes.append(queryTime)

            #print("\nself.scoreDoc:", self.scoreDoc, "\n")

            #self.writeToFile()

            sortedDocs = dict(sorted(self.scoreDoc.items(), key=lambda item: item[1], reverse=True))
            #print("sortedDocs:", sortedDocs)
            self.scoreDoc.clear()

            top5 = dict(itertools.islice(sortedDocs.items(), 5))
            top3 = dict(itertools.islice(top5.items(), 3))
            top2 = dict(itertools.islice(top3.items(), 2))

            topXresults = { "top5": top5, "top3": top3, "top2": top2}   # 50/20/10
            self.topResults[query] = topXresults

        for k, v in self.topResults.items():
            print("\n", k, v)

        medianQueryLatency = statistics.median(self.queryTimes)
        with open("answers/questions.txt", "a") as f:
            f.write("\nMedian query latency = {} (hh:mm:ss.ms)" .format(medianQueryLatency))


        """                                     # Open Docs id for printing results
        with open("extras/idDocs.txt") as f:
            idDocs = [line[:-1] for line in f]

        for key in sortedDocs:
            print("Best:", idDocs[key])
        """


""" Main """
if __name__ == "__main__":

    # Default values
    k1 = 1.2
    b = 0.75

    searcher = search()
    searcher.loadToMem()
    searcher.loadQueries(k1, b)
