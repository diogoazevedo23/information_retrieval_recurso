"""

    Autores:
    Diogo Azevedo nº 104654 / Ricardo Madureira nº 104624
    04/02/2022

"""

# Imports
import csv
from io import StringIO
from itertools import islice
from os import read
import re
import sys
import json
import math
from Tokenizer import Tokenizer    # Import of Tokenizer
import time
import ast
import pandas as pd

from operator import methodcaller

""" Main Class """


class Ranked:

    lenDocs = {}
    tfDict = {}
    tokens = []
    finalTF = {}
    array_queries = []
    numDocs = 0
    tfidfDocs = {}
    tfidfQuery = {}
    bm25Final = {}
    line2 = ""
    posWord = 0
    posWordDict = {}
    top50ResultsDict = {}
    top20ResultsDict = {}
    top10ResultsDict = {}
    fullTopResults = {}
    boostDict = {}

    full_path1 = "extras/dicionario.txt"
    dictionario_palavras = json.load(open(full_path1))
    full_path2 = "finalBlock/completeText.txt"
    dictionary_final = json.load(open(full_path2))
    full_path3 = "extras/lenDocs.txt"
    lenDocs = json.load(open(full_path3))

    def __init__(self, query, tokenizer_mode, steemer, ranker, boost):
        self.query = query
        self.ranker = ranker
        self.boost = boost
        self.tokenizer = Tokenizer("4", tokenizer_mode, steemer)

    """ Ler as queries e por cada query fazer a função tf-idf / bm25 """

    def readQuery(self):
        print("\n\t* Reading Queries / Tokenizer *\n")
        with open(self.query, 'r') as f:
            for line in f:
                self.line2 = line
                print("line -->", line)
                self.tokens = self.tokenizer.tokenize2(line)
                self.array_queries.append(line)

                # print("\nFinals Tokens -->", self.tokens)

                self.things()
                self.tfidf_Queries()
                if self.ranker == 'tfidf':
                    # print("\n\tself.ranker ->", self.ranker, "\n")
                    self.tf_idfFinal(line)
                elif self.ranker == 'bm25':
                    # print("\n\tself.ranker ->", self.ranker, "\n")
                    self.bm25(line)
                else:
                    print("Choose a valid ranker")
                    exit(0)

                self.tokens.clear()

    def things(self):
        start = time.time()
        print("\n\t\t* Things *\n")
        for x in self.tokens:
            for i in self.lenDocs:
                if x not in self.tfDict:
                    self.tfDict[x] = {i: 0}
                else:
                    self.tfDict[x].update({i: 0})

        self.numDocs = len(self.lenDocs)

        # print("\nself.tfDict -->", self.tfDict)
        end = time.time()
        print("Time things =", end - start)

    """Fazer o tf-idf dos docs"""

    def tfidf_Docs(self):
        start = time.time()
        print("\t\t* TFIDF Docs *")
        self.things()

        for k, v in self.dictionary_final.items():
            for x in v.items():
                # print("k", k, "| x[0]", x[0], "| x[1]", x[1], "| dictionario_palavras[k]", self.dictionario_palavras[k], "| Mult =", x[1] * self.dictionario_palavras[k])
                if k not in self.tfidfDocs:
                    self.tfidfDocs[k] = {x[0]: x[1] * 1}
                else:
                    if x not in self.tfidfDocs[k]:
                        self.tfidfDocs[k].update({x[0]: x[1] * 1})
                    else:
                        self.tfidfDocs[k][x[0]].update(x[1] * 1)

        # print("\nself.tfidfDocs ->", self.tfidfDocs, "\n")

        sumVals2 = 0
        for k in self.tfidfDocs.values():
            for x in k.values():
                sumVals2 += math.pow(x, 2)

        sumVals2 = math.sqrt(sumVals2)
        # print("sumVals2", sumVals2, "\n")

        # print("Afer Cosine")
        for k, v in self.tfidfDocs.items():
            # print("k:", k, "| v:", v)
            for x in v.items():
                # print("x:", x)
                self.tfidfDocs[k].update({x[0]: x[1] / sumVals2})

        # print("\nself.tfidfDocs ->", self.tfidfDocs, "\n")
        end = time.time()
        print("Time TFIDFDocs =", end - start)

    """Fazer o tf-idf das queries"""

    def tfidf_Queries(self):
        start = time.time()
        print("\n\t\t* TFIDF Queries *\n")
        tfQuery = {}
        idfQuery = {}

        for k, v in self.dictionario_palavras.items():
            # print("k:", k, "| v:", v)
            tfQuery[k] = 0

        for k in self.tokens:
            if k not in tfQuery:
                tfQuery[k] = 1
            else:
                tfQuery[k] += 1

        # print("tfQuery ->", tfQuery)

        for k, v in self.dictionary_final.items():
            # print("k:", k, "| v:", v, "| len(v):", len(v))
            idfQuery[k] = math.log10(self.numDocs / len(v))

        for k in self.tokens:
            if k not in idfQuery:
                # print(k, "is not")
                idfQuery[k] = 0
            else:
                # print(k, "is in")
                pass

        # print("\nidfQuery ->", idfQuery, "\n")

        for k, v in tfQuery.items():
            # print("self.tfidfQuery[k] = v", v , "* idfQuery[k]", idfQuery[k])
            self.tfidfQuery[k] = v * idfQuery[k]

        # print("\nself.tfidfQuery ->", self.tfidfQuery, "\n")

        sumVals = 0
        for x in self.tfidfQuery.values():
            sumVals += math.pow(x, 2)
        sumVals = math.sqrt(sumVals)

        # print("\nAfter Cosine")

        for x, v in self.tfidfQuery.items():
            self.tfidfQuery.update({x: v / sumVals})

        # print("\nself.tfidfQuery ->", self.tfidfQuery, "\n")
        end = time.time()
        print("Time TFIDFQueries =", end - start)

    """Fazer o tf-idf final no formato lnc.lct"""

    def tf_idfFinal(self, line):
        start = time.time()

        print("\n\t\t* TFIDF Final *")

        tdidfFinal = {}

        print("self.tfidfQuery->", self.tfidfQuery, "\n")
        print("self.tfidfDocs->", self.tfidfDocs, "\n")

        for key, value in self.tfidfQuery.items():
            for key2, value2 in self.tfidfDocs.items():
                if key == key2:
                    # print("key:", key, "value:", value, "key2:", key2, "value2:", value2)
                    for k, v in value2.items():
                        # print("key:", key, "value:", value, "key2:", key2, "k:", k, "v:")
                        # print("key:", key, "value:", value, "value2:", v, "result =", value * v, "k:", k)
                        if key not in tdidfFinal:
                            tdidfFinal[key] = {k: value * v}
                        else:
                            if k not in tdidfFinal[key]:
                                tdidfFinal[key].update({k: value * v})
                            else:
                                tdidfFinal[key][k].update(value * v)

        # print("\ntdidfFinal ->", tdidfFinal, "\n")
        # print("\nFinalSum")

        finalSum = {}
        for key, subdict in tdidfFinal.items():
            for k, v in subdict.items():
                finalSum[k] = finalSum.get(k, 0) + v

        # print("\ntdidfFinal ->", finalSum, "\n")
        # print("\nTop Results")

        top100Results = {k: v for k, v in sorted(
            finalSum.items(), key=lambda item: item[1], reverse=True)[0:100]}

        #print("top100Results ->", top100Results)

        """ Boost On/Off"""
        if sys.argv[5] == 'yes':
            for x in self.tokens:
                for k, v in self.tfidfDocs.items():
                    #print("k->", k, "| v ->" ,v)
                    if x in k:
                        #print("x->", x, "v->", self.tfidfDocs[x].keys())
                        for doc in self.tfidfDocs[x].keys():
                            if doc not in self.boostDict:
                                self.boostDict.update({doc : 1})
                            else:
                                self.boostDict[doc] += 1

            print("self.boostDict->", self.boostDict, "\n")

            for x, v in self.boostDict.items():
                if x in top100Results:
                    print("x->", x, "| k ->", v, " |", top100Results[x])
                    #boost = int(v) * 0.1
                    #print("boost ->", boost)
                    top100Results[x] += (v * 0.1)
                    print("top100Results[k] ->", top100Results[x])

        cleanLine = line.rstrip("\n")

        top50Results = list(top100Results.keys())[0:50]
        top20Results = top50Results[0:20]
        top10Results = top20Results[0:10]

        for x in top50Results:
            if cleanLine not in self.top50ResultsDict:
                self.top50ResultsDict.update({cleanLine: []})
            else:
                self.top50ResultsDict[cleanLine].append(x)

        for x in top20Results:
            if cleanLine not in self.top20ResultsDict:
                self.top20ResultsDict.update({cleanLine: []})
            else:
                self.top20ResultsDict[cleanLine].append(x)

        for x in top10Results:
            if cleanLine not in self.top10ResultsDict:
                self.top10ResultsDict.update({cleanLine: []})
            else:
                self.top10ResultsDict[cleanLine].append(x)

        # print("top50ResultsDict ->", self.top50ResultsDict, "\n")
        # print("top20ResultsDict ->", self.top20ResultsDict, "\n")
        # print("top10ResultsDict ->", self.top20ResultsDict, "\n")

        self.fullTopResults.update({"top50Results": self.top50ResultsDict})
        self.fullTopResults.update({"top20Results": self.top20ResultsDict})
        self.fullTopResults.update({"top10Results": self.top10ResultsDict})

        # print("self.fullTopResults ->", self.fullTopResults)

        # print("top100Results ->\n", top100Results)
        print("\n\n\t** Top 100 documentos **")
        with open('finalResult/resultsTFIDF.txt', 'a') as f:
            line = "Q:", self.line2
            f.write(f'{line}\n')
            for k, v in top100Results.items():
                f.write(f'{k}' + "\t" + f'{v} \n')

        end = time.time()
        print("Time TFIDFFinal =", end - start)

    """Calculo do BM25"""

    def bm25(self, line, k1=1.2, b=0.75):
        start = time.time()

        print("\n\t\t* BM25 *\n")

        idfQuery = {}
        tfQuery = {}
        self.bm25Final = {}
        finalSum = {}
        idfWord = 0
        numerator = 0
        denominator = 0
        bm25Doc = 0

        # print("\nself.tfDict -->", self.tfDict, "\n")

        for k, v in self.dictionario_palavras.items():
            # print("k:", k, "| v:", v)
            tfQuery[k] = 0

        for k in self.tokens:
            if k not in tfQuery:
                tfQuery[k] = 1
            else:
                tfQuery[k] += 1

        # print("tfQuery ->", tfQuery, "\n")

        # print("self.lenDocs:", self.lenDocs, "\n")
        avdl = sum(self.lenDocs.values()) / self.numDocs
        # print("self.numDocs:", self.numDocs, "SUM:", avdl)

        for k, v in self.dictionary_final.items():
            # print("k:", k, "| v:", v, "| len(v):", len(v))
            idfQuery[k] = math.log10(self.numDocs / len(v))

        # print("\nidfQuery ->", idfQuery, "\n")

        for k, v in self.dictionary_final.items():
            # print("Termo(k):", k, "IDF(v):", v)
            for x in v.items():
                idfWord = idfQuery[k]
                numerator = ((k1 + 1) * x[1])
                denominator = (
                    (k1 * ((1 - b) + (b*(self.lenDocs[x[0]]/avdl)))) + x[1])
                # print("x[0]:", x[0], "x[1]:", x[1], "idfQuery[k]:", idfQuery[k], "idfWord:", idfWord, "numerator:", numerator, "self.lenDocs[x[0]]", self.lenDocs[x[0]])
                bm25Doc = (idfWord * (numerator / denominator))
                # print("bm25Doc:", bm25Doc)

                if k not in self.bm25Final:
                    self.bm25Final[k] = {x[0]: bm25Doc}
                else:
                    if x[0] not in self.bm25Final[k]:
                        self.bm25Final[k].update({x[0]: bm25Doc})
                    else:
                        self.bm25Final[k][x[0]].update(bm25Doc)

        print("\n\n\t ** Final ** \n")
        # print("self.bm25Final ->", self.bm25Final, "\n")

        for key, subdict in self.bm25Final.items():
            for k, v in subdict.items():
                finalSum[k] = finalSum.get(k, 0) + v

        # print("\nBM25FinalSum ->", finalSum, "\n")

        top100Results = {k: v for k, v in sorted(
            finalSum.items(), key=lambda item: item[1], reverse=False)[0:100]}

        """ Boost On/Off"""
        if sys.argv[5] == 'yes':
            for x in self.tokens:
                for k, v in self.tfidfDocs.items():
                    #print("k->", k, "| v ->" ,v)
                    if x in k:
                        #print("x->", x, "v->", self.tfidfDocs[x].keys())
                        for doc in self.tfidfDocs[x].keys():
                            if doc not in self.boostDict:
                                self.boostDict.update({doc : 1})
                            else:
                                self.boostDict[doc] += 1

            print("self.boostDict->", self.boostDict, "\n")

            for x, v in self.boostDict.items():
                if x in top100Results:
                    print("x->", x, "| k ->", v, " |", top100Results[x])
                    #boost = int(v) * 0.1
                    #print("boost ->", boost)
                    top100Results[x] += (v * 0.1)
                    print("top100Results[k] ->", top100Results[x])

        cleanLine = line.rstrip("\n")

        top50Results = list(top100Results.keys())[0:50]
        top20Results = top50Results[0:20]
        top10Results = top20Results[0:10]

        for x in top50Results:
            if cleanLine not in self.top50ResultsDict:
                self.top50ResultsDict.update({cleanLine: []})
            else:
                self.top50ResultsDict[cleanLine].append(x)

        for x in top20Results:
            if cleanLine not in self.top20ResultsDict:
                self.top20ResultsDict.update({cleanLine: []})
            else:
                self.top20ResultsDict[cleanLine].append(x)

        for x in top10Results:
            if cleanLine not in self.top10ResultsDict:
                self.top10ResultsDict.update({cleanLine: []})
            else:
                self.top10ResultsDict[cleanLine].append(x)

        # print("top50ResultsDict ->", self.top50ResultsDict, "\n")
        # print("top20ResultsDict ->", self.top20ResultsDict, "\n")
        # print("top10ResultsDict ->", self.top20ResultsDict, "\n")

        self.fullTopResults.update({"top50Results": self.top50ResultsDict})
        self.fullTopResults.update({"top20Results": self.top20ResultsDict})
        self.fullTopResults.update({"top10Results": self.top10ResultsDict})

        # print("self.fullTopResults ->", self.fullTopResults)

        print("\n\n\t** Top 100 documentos **")
        with open('finalResult/resultsBM25.txt', 'a') as f:
            line = "Q:", self.line2
            f.write(f'{line}\n')
            for k, v in top100Results.items():
                f.write(f'{k}\n')

        end = time.time()
        print("Time Bm25 =", end - start)

    """Escrever para um ficheiro o top x docs (ranking)"""

    def writeToFile(self, finalDict):
        start = time.time()

        print("\nWrinting To File")
        with open('finalResult/writeToFile.txt', 'a') as f:
            for key, value in finalDict.items():
                string = ""
                for x, y in value.items():
                    string += str(((x, y))) + str((self.search(key, x)))
                line2 = key+"|"+str(self.dictionario_palavras[key])+"|"+string
                f.write(f'{line2}\n')
            f.write('\n')

        end = time.time()
        print("Time writeToFile =", end - start)

    """ Procurar as posições dos tokens"""

    def search(self, word, doc):
        st = time.time()
        posWordArr = []
        with open('extras/docPositions.txt', 'r') as fp:
            for line in fp:
                d = ast.literal_eval(line)
                if word in d and doc in d[word]:
                    # print("found!")
                    posWordArr.append({d[word][doc]})

        print("Time ->", (time.time() - st))

        return posWordArr

    """ Calculo das metricas (precision, recall, etc...) """

    def metrics(self):
        print("\t* Doing Metrics*\n")
        arrDict = {}
        metricsDict = {}

        with open('queries.relevance.txt', 'r') as csvfile:
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

        # print("arrDict->", arrDict, "\n")

        for key, value in arrDict.items():
            print(key)
            tp = 0
            fp = 0
            for a, b in self.fullTopResults.items():
                # print("a->", a, "b->", b)
                if key in b.keys():
                    for x in b[key]:
                        if x in arrDict[key].keys():
                            # print(x, "in", arrDict[key].keys())
                            tp += 1
                        else:
                            # print(x, "not in", arrDict[key].keys())
                            fp += 1

                fp = 0.1
                fn = len(x) - fp

                precision = (round((tp/(tp + fp)), 3))
                recall = (round(tp/(tp + fn), 3))
                fMeasure = (round((2 * recall * precision) /
                                  (recall + precision + 0.1), 3))

                arrayMeasures = {}
                arrayMeasures.update({"precision": precision})
                arrayMeasures.update({"recall": recall})
                arrayMeasures.update({"fMeasure": fMeasure})

                if key not in metricsDict:
                    metricsDict.update({key: {a: arrayMeasures}})
                else:
                    if a not in metricsDict[key]:
                        metricsDict[key].update({a: arrayMeasures})

        # print("self.fullTopResults->", self.fullTopResults, "\n")
        # print("metricsDict->",  metricsDict, "\n")

        df = pd.DataFrame(metricsDict).T
        df.fillna(0, inplace=True)

        with open('finalResult/metrics.txt', 'a') as f:
            dfAsString = df.to_string(header=True, index=True)
            f.write(dfAsString)
            f.write("\n")


""" Main """

if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage: py ranked.py ('path to queries.txt') ('yes') ('yes') ('tfidf'), ('yes')"
              + "\n** CHOICES **"
              + "\npath = queries.txt"
              + "\ntokenizer = yes/no"
              + "\nstemmer = yes/no"
              + "\nranker = tfidf/bm25"
              + "\nboost = yes/no")
        sys.exit(1)

    try3 = Ranked(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    start = time.time()
    try3.tfidf_Docs()
    try3.readQuery()
    try3.writeToFile(try3.tfidfDocs)
    end = time.time()
    print("Total Time Spent was =", end - start)
    # print(try3.posWordDict)
    try3.metrics()
    # try3.search("diogo", "3C")
