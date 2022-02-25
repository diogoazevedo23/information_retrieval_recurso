"""
    Recurso Recuperação Informação

    Autores:
    Diogo Azevedo nº 104654 / Ricardo Madureira nº 104624
    25/02/2022
"""

# Imports
from Tokenizer import Tokenizer         # Import of Tokenizer
import ast
import pandas as pd
import getopt
import sys
import os
import re
import math
import itertools        # Slice dictionary for top results
from datetime import datetime
import statistics
import csv

class search:

    def __init__(self, k1, b, boost):

        self.k1 = k1
        self.b = b
        self.boost = boost
        self.avgdl = 0
        self.metadados = []
        self.docsLength = []
        self.dicionario = {}
        self.scoreDoc = {}
        self.L = 0              # Sum of all weights of a doc
        self.queryTimes = []
        self.finaltime = 0      # Just to answer question about index searcher

        with open("extras/metadados.txt",'r') as f:             # Load Metadados
            self.metadados = [line[:-1] for line in f]

        #print("self.metadados:", self.metadados)

        with open("files/queries1.txt",'r') as f:               # Load Queries
            self.queries = [line[:-1] for line in f]

        #print("self.queries:", self.queries)

        self.indexBlocks = []
        for file in os.listdir("finalBlocks/"):                 # Save block index for search
            removeTxt = file.split('.')[0]
            first, last = removeTxt.split('_')
            self.indexBlocks.append(first + "_" + last)

        #print("\nself.indexBlocks:", self.indexBlocks, "\n")
        
        with open("extras/idDocs.txt") as f:        # Open Docs id for printing results
            self.idDocs = [line[:-1] for line in f]

        min_tamanho = self.metadados[1]
        tokenizer_mode = self.metadados[2]
        steemer = self.metadados[3]
        stopwords_file = self.metadados[4]
        self.ranker = self.metadados[6]                         # tfidf / bm25

        self.tokenizer = Tokenizer(min_tamanho, tokenizer_mode, steemer, stopwords_file)

        self.cleanQueries = {}
        self.topResults = {}
        self.metricsDict = {}

    """ loadToMem """
    def loadToMem(self):

        #if self.ranker == "bm25":
        with open("extras/docsLength.txt",'r') as f:
            self.avgdl = f.readline()[:-1]                       # Load avgdl
            self.docsLength = [line[:-1] for line in f]          # Load dl

            #print("avgdl:", self.avgdl)
            #print("docsLength:", self.docsLength)

        with open("extras/dicionario.txt",'r') as f:
            self.dicionario = dict([line.split() for line in f])    # Load dicionary(IDF)

        #print("dicionario:", self.dicionario)

    """ loadQueries """
    def loadQueries(self):

        for query in self.queries:                          # For each query
            queryStartTime = datetime.now()
            tfQuery = {}
            newTokens = self.tokenizer.tokenize(query, 1)   # Tokenize query
            #print("\nnewTokens:", newTokens, "\n")

            print("\nRanker:", self.ranker)
            if self.ranker == "tfidf":                      # If TF-IDF

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

                boostCount = {}
                for term in newTokens:                      # Open right file | get tf values | lnc*ltc
                    word = term[0]
                    #print("\n", word)

                    indexSearcherTime = datetime.now()

                    for doc in self.indexBlocks:            # Choose the correct file
                        first, last = doc.split("_")
                        #print(first, "|", last)
                        if word >= first and word <= last:
                            tfidfDocs = {}
                            #print(word, "Its betwen", first, "and", last)
                            with open("finalBlocks/" + first + "_" + last + ".txt", "r") as f:
                                #print("File:", f)
                                for line in f.readlines():
                                    #print("line:", line)
                                    term_file,value = re.split(':', line.rstrip('\n'), maxsplit=1)
                                    if term_file == word:
                                        tfidfDocs[term_file] = ast.literal_eval(value)  # Get the correct posting lists

                                        self.finaltime = datetime.now() - indexSearcherTime

                            #print("\ntfidfDocs:", tfidfDocs, "\n")
                                    
                            for key, value in tfidfDocs[word].items():      # Calculate the score
                                tf_doc = value['weight']
                                #print("word:", word, "|", tfQuery[word], "|", tf_doc)

                                newScore = tfQuery[word] * tf_doc
                                #print("newScore:", newScore)

                                if key not in self.scoreDoc:
                                    boostCount[key] = 1
                                    self.scoreDoc[key] = newScore
                                else:
                                    boostCount[key] += 1
                                    self.scoreDoc[key] += newScore

                if self.boost == "on":
                    for k, v in boostCount.items():             # Boost
                        if boostCount[k] >= len(newTokens):     # If it has the terms of the query
                            value = (0.1 * boostCount[k]) / int(self.docsLength[k])
                            self.scoreDoc[k] += value

            elif self.ranker == "bm25":                     # If BM25
                
                boostCount = {}
                for term in newTokens:  # Open right file | get tf values | BM25
                    word = term[0]

                    #print("\nTerm:", word, "\n")

                    for doc in self.indexBlocks:            # Open right file
                        first, last = doc.split("_")
                        if word >= first and word <= last:
                            tfidfDocs = {}
                            with open("finalBlocks/" + first + "_" + last + ".txt", "r") as f:
                                #print("File:", f)
                                for line in f.readlines():
                                    term_file,value = re.split(':', line.rstrip('\n'), maxsplit=1)
                                    if term_file == word:
                                        tfidfDocs[term_file] = ast.literal_eval(value)  # Get postings with tf value

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
                                    boostCount[key] = 1
                                    self.scoreDoc[key] = bm25Doc
                                else:
                                    boostCount[key] += 1
                                    self.scoreDoc[key] += bm25Doc

                if self.boost == "on":
                    for k, v in boostCount.items():             # Boost
                        if boostCount[k] >= len(newTokens):     # If it has the terms of the query
                            value = (0.1 * boostCount[k]) / int(self.docsLength[k])
                            self.scoreDoc[k] += value

            # ------------------------------------

            queryTime = (datetime.now() - queryStartTime)
            self.queryTimes.append(queryTime)

            #print("\nself.scoreDoc:", self.scoreDoc, "\n")

            sortedDocs = dict(sorted(self.scoreDoc.items(), key=lambda item: item[1], reverse=True))
            self.scoreDoc.clear()
            top5 = {}

            i = 0
            while i < 5:    # Get Top 50 and the id of Doc
                for k, v in sortedDocs.items():
                    top5[self.idDocs[k]] = v
                    i += 1

            top3 = dict(itertools.islice(top5.items(), 3))      # Top 20
            top2 = dict(itertools.islice(top3.items(), 2))      # Top 10

            topXresults = { "top5": top5, "top3": top3, "top2": top2}   # 50/20/10
            self.topResults[query] = topXresults

        with open("answers/topResults.txt", "w") as f:
            for k, v in self.topResults.items():
                f.write("Query:" + str(k) + "\n")
                for a, b in v.items():
                    f.write(str(a) + "\n")
                    f.write(str(b) + "\n")
                f.write("\n")

        self.medianQueryLatency(self.queryTimes)    # Write to file median query latency 

    """ Calculate the median query latency and print to file question"""
    def medianQueryLatency(self, queryTimes):
        mQL = statistics.median(queryTimes)
        with open("answers/questions.txt", "a") as f:
            f.write("\nAmount of time taken to start up an index searcher = {} (hh:mm:ss.ms)" .format(self.finaltime))
            f.write("\nMedian query latency = {} (hh:mm:ss.ms)" .format(mQL))

    """ Do the metrics """
    def metrics(self):
        arrDict = {}

        with open('files/queries.relevance1.txt', 'r') as csvfile:
            csvReader = csv.reader(csvfile, delimiter='\t')
            for row in csvReader:
                if not (row):
                    continue
                else:
                    if row[0].startswith("Q:"):
                        key = row[0]
                        newKey = str(key.replace('Q:', ''))
                        arrDict.update({newKey: {}})
                    else:
                        arrDict[newKey].update({row[0]: row[1]})

        #print("\narrDict:", arrDict)
        #print("\nself.topResults:", self.topResults, "\n")

        for key, value in arrDict.items():
            #print(key)
            tp = 0
            fp = 0
            for a, b in self.topResults.items():                # dict  = query : dic2
                for top, dictX in b.items():                    # dict2 = topX  : dict3
                    docsPrecision = []
                    for x in dictX:                             # dict3 = idDoc : score | x = idDoc
                        if x in arrDict[key].keys():            # if x in queries.relevance.txt
                            tp += 1         # Relevant retrieved
                        else:
                            fp += 1         # Non relevant retrieved

                        fn = len(x) - tp    # All documents minus Relevant retrieved equals Relevant not retrieved

                    try:
                        precision = (round((tp/(tp + fp)), 4))
                    except ZeroDivisionError:
                        precision = 0
                    docsPrecision.append(precision)

                    try:
                        recall = (round(tp/(tp + fn), 4))
                    except ZeroDivisionError:
                        recall = 0
                    
                    try:
                        fMeasure = (round((2 * recall * precision) / (recall + precision), 4))
                    except ZeroDivisionError:
                        fMeasure = 0

                    try:
                        averagePrecision = sum(docsPrecision)/len(docsPrecision)
                    except ZeroDivisionError:
                        averagePrecision = 0

                    arrayMeasures = {}
                    arrayMeasures.update({"precision": precision})
                    arrayMeasures.update({"recall": recall})
                    arrayMeasures.update({"fMeasure": fMeasure})
                    arrayMeasures.update({"averagePrecision": averagePrecision})

                    if key not in self.metricsDict:
                        self.metricsDict.update({key: {top: arrayMeasures}})
                    else:
                        if a not in self.metricsDict[key]:
                            self.metricsDict[key].update({top: arrayMeasures})

        print("\nmetricsDict:", self.metricsDict)

    """ Print metrics to file """
    def topMetrics(self):
        df = pd.DataFrame(self.metricsDict).T
        df.fillna(0, inplace=True)

        with open('answers/metrics.txt', 'w') as f:
            dfAsString = df.to_string(header=True, index=True)
            f.write(dfAsString)
            f.write("\n")

""" Main """
if __name__ == "__main__":

    # Default values
    k1 = 1.2
    b = 0.75
    boost = "on"

    try:
        opts, args = getopt.getopt(sys.argv[1:], "k:b:o:")
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit()
    
    if len(sys.argv) < 1:
        sys.exit()

    for short, arg in opts:
        if short == "-k":   # K1
            try:
                k1 = float(arg)
            except:
                print("ERROR")
                sys.exit(1)
        if short == "-b":   # B
            try:
                b = float(arg)
            except:
                print("ERROR")
                sys.exit(1)
        if short == "-o":   # Boost
            try:
                boost = str(arg)
            except:
                print("ERROR")
                sys.exit(1)

    searcher = search(k1, b, boost)
    searcher.loadToMem()
    searcher.loadQueries()
    searcher.metrics()
    searcher.topMetrics()
