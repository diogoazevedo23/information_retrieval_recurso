"""

    Autores:
    Diogo Azevedo nº 104654 / Ricardo Madureira nº 104624
    04/02/2022

"""

# Imports
import sys
import os
import json  # Json functions
import time  # Time functions
import math
import psutil
import regex as re
import ast
from statistics import mean

""" Main Class """


class Merger:
    
    def __init__(self):
        self.indexed_words = {}       
        self.N = 0                  # Size of corpus


    def updateColSize(self, N):
        self.N = N


    def merge_blocks(self, dicionario):

        print("\n\t\t** Merger **\n")
        self.temp_index = {}
        last_term = ""
        block_files = os.listdir("blocks/")
        block_files = [open("blocks/" + block_file) for block_file in block_files]
        lines = [block_file.readline()[:-1] for block_file in block_files]
        """
        lines = []
        for block_file in block_files:
            lines.append(block_file.readline()[:-1])
        """
        print("self.N:", self.N)
        print("dicionario:", dicionario)

        mem_initial = psutil.virtual_memory().available

        while len(block_files) > 0:
            min_index = lines.index(min(lines)) # vai buscar o index do array (linha do array) -> 0 , i++
            #print("\nmin_index ->", min_index)
            line = re.split(":",lines[min_index].rstrip('\n'),maxsplit=1)
            #print("\nline ->", line)          # line -> ['abstract', " {'1A': {'weight': 1, 'positions': [2]}}"]
            current_term = line[0]          # current_term -> abstract
            current_postings = line[1]      # current_postings -> {'1A': {'weight': 1, 'positions': [2]}}

            # we check initially, so we dont put the same term in two diff files
            mem_used = mem_initial - psutil.virtual_memory().available
            #print("mem_used ->", mem_used)
            if mem_used > 3000000 and current_term!=last_term:
                print("Max mem reached")
                self.write_partition_index()
                mem_initial = psutil.virtual_memory().available
            
            if current_term != last_term:
                json_dict = ast.literal_eval(current_postings)


                #TF-IDF (LNC) No idf/df(1) or just DF??

                #print("json_dict:", json_dict)
                #if sys.argv[6] == 'tfidf':
                    #for x in json_dict.values():
                        #print("x['weight']:", x['weight'], "| dicionario[current_term]:", dicionario[current_term], "| term:", current_term)
                        #print(dicionario[current_term])
                        #x['weight'] *= dicionario[current_term]

                self.temp_index[current_term] = json_dict
                print("self.temp_index[current_term]", self.temp_index[current_term])
                last_term = current_term # abstract

            else:
                json_dict = ast.literal_eval(current_postings)
                #if sys.argv[6] == 'tfidf':
                    #for x in json_dict.keys():
                        #print("x['weight']:", x['weight'], "| dicionario[current_term]:", dicionario[current_term], "| term:", current_term)
                        #print(dicionario[current_term])
                        #x['weight'] *= dicionario[current_term]     # TFIDF = TF * DF (LNC)
                
                tmp_dict = self.temp_index[current_term]
                new_val = {**json_dict, **tmp_dict}             # merging the two dicts
                self.temp_index[current_term] = new_val

            lines[min_index] = block_files[min_index].readline()[:-1]

            
            if lines[min_index] == "":
                    block_files[min_index].close()
                    block_files.pop(min_index)
                    lines.pop(min_index)
            

        self.write_partition_index()

        # IDF
        for x, v in dicionario.items():
            dicionario[x] = round((self.N / dicionario[x]),4)

        self.writeDicionario(dicionario)

        # Lengh Normalization
        # dividing each of its components by its length –
 
        # BM25:
            # DL (Document Lenght)              - Done
            # AVGDL (Abverage Document Length)  - Done in Main
            # avgdl = mean(dl)                  - Done in Main

    def write_partition_index(self):
        ordered_dict = sorted(self.temp_index.items(), key = lambda kv: kv[0])
        first = ordered_dict[0][0]
        last = ordered_dict[-1][0]
        with open("finalBlocks/" + f"{first}_{last}.txt",'w+') as f:
            for term, posting in ordered_dict:
                string = f"{term}:{str(posting)}\n"
                f.write(string)
        self.temp_index = {}
        f.close()

    """ Write dicionary(term:idf) to text file"""
    def writeDicionario(self, dicionario):
        with open("extras/dicionario.txt",'w+') as f:
            for term, value in dicionario.items():
                string = term + ' ' + str(value) + '\n'
                f.write(string)


""" Main """
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: py teste1.py term('diogo')"
              + "\n** CHOICES **"
              + "\nterm = term you need to search on index")
        sys.exit(1)

    try2 = Merger()

    indexSearchStart = time.time()
    try2.index_searcher(sys.argv[1])
    indexSearchEnd = time.time()
    indexSearchFinal = indexSearchEnd - indexSearchStart

    with open("finalResult/finalAnswers.txt", "a") as f:
        print("\ne) Amount of time taken to start up an index searcher, after the final index is written to disk ==",
              indexSearchFinal, "s.", file=f)
        print("\nThe term", sys.argv[1], "is in", len(
            try2.contagem[0]), "documents", file=f)
