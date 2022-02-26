<h1>Information Retrieval work RECURSO</h1>

Autores:
<p>Diogo Azevedo nº 104654 / Ricardo Madureira nº 104624
<p>25/02/2022

<h4>!!! IMPORTANT !!!</h4>

<p>Before starting the program, you need to install these two modules.

<p>pip install PySteemer
<p>pip install psutil

---------------

<h5>How to execute:</h5>

<h5>py Main.py -c a -m 3 -t a -s yes -f yes -k 10000 -r tfidf</h5>

<p>-c: Combinations (a, b, c, d) 
<p>-m: Minimum lenght of term (3)
<p>-t: Tokenizer (a/b)
<p>-s: Steemer (yes/no)
<p>-f: Stopfiles (yes/no/pathfile)
<p>-k: chunckizse (1000)
<p>-r: ranker (tfidf/bm25)

<p>This program will read a .tsv and index it.

---------------

<p>Then we can do the ranker with:

<h5>py Searcher.py -k 1.2 -b 0.75 -o on</h5>

<p>-k: k1 (1.25)
<p>-b: b (0.75)
<p>-o: boost(on/off)

---------------

<h3>TL;DR</h3>

<h4>Imports</h4>

```jsx
pip install PyStemmer
```

```jsx
pip install psutil
```

<h4>Index:</h4>

```jsx
py Main.py -c a -m 3 -t a -s yes -f yes -k 10000 -r tfidf
```
<p>
<h4>Ranker:</h4>

```jsx
py Searcher.py -k 1.2 -b 0.75 -o on
```

