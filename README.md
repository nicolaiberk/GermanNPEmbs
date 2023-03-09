# GermanNPEmbs

Word embedding model of over 2.2M German newspaper articles 2013-2021 (Bild, Frankfurter Allgemeine, Spiegel Online, Süddeutsche Zeitung, TAZ, Welt).

The model can be downloaded [here](https://www.dropbox.com/sh/q0fjjwbhzcfhq8k/AACqKEybJsDhZyHNwJhUZxKla?dl=1).


After download, load with

```python
import gensim
model = gensim.models.Word2Vec.load("np_emb")
```

and access word vectors with

```python
model.wv.vectors
```