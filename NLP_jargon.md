## Syntax vs. Semantics

In short, syntax refers to grammar, while semantics refers to meaning. **Syntax** is the set of rules needed to ensure a sentence is grammatically correct. **Semantics** is how one’s lexicon, grammatical structure, tone, and other elements of a sentence coalesce to communicate its meaning. 

## Language Model

In many Natural Language tasks you have a language (L) and want to build a model (M) for the language. In NLP we think of language L as a process for generating text, where the process is a collection of probability distributions. Given a history $h$ consisting of a series of previous words in a sentence, the language $L$ is the probability that the next word is $w$. Similarly, given an entire sentence $s$, we can evaluate $L(s)$ as the probability of the sentence occurring. A language model M will try to replicate this language, which we can then use for various tasks. 

## Source Channel Framework

![alt text](https://github.com/louisds/NLP-projects/blob/main/images/noisy_channel_model.png)

The source channel framework or noisy channel framework is a framerwork used in spell checking, speech recognition, translation, POS tagging, etc. It consists of a message (input X) that is encoded and fed into a (noisy) channel. The output from the noisy channel (output Y) is then decoded (decoded message X'), attempting to reconstruct the original message based on this noisy output. In NLP we do not usually act on encoding. The problem is reduced to decoding the noisy output Y for getting the most likely input given the output. 

An example of a noisy channel for spell checking is a channel that scrambles some letters in an input sentence (and thus producing spelling errors). In this case, the correct language (real language) is the input X, while the output Y is the language with errors (observed language). The probability P(X) is the probability of the word sequence in the correct language (language model probability). The probability P(Y|X) is the probability that we have a specific scrambled output sentence Y, giving a specific input sentence X (noisy channel probability). The goal of the framework is the obtain X', based on the probability P(X|Y): 

$$
X' = argmax P(x|Y) = argmax \frac{p(x) p(Y|x)}{P(Y)} \prop argmax p(w) p(Y|x)
$$


## Perplexity



## Entropy (in language)

## Dependency (Syntactic) vs. Constituency Parsing

## Ontologies

## Information Extraction

## Entity Recognition vs. Disambiguation vs. Linking

## Document Retrieval and Search

## Relation Extraction

## Inverse Document Frequency (IDF)

## Term Frequency–Inverse Document Frequency (TF-IDF)

## N-grams

## Skip gram

## Bag of words

## Pragmatic/Lexical/Syntactic/Semantic Ambiguity

## Masked Language Model

## POS tagging

## Ensemble Method

## Name Entity Recognition (NER)

## Lemmatization

## Stemming

## Tokenization

## Soundex

## Cosine Similarity

## Keyword Normalization

## Latent Semantic Indexing

## Latent Dirichlet Allocation (LDA)

## Transfer Learning

## Multi Task Learning

## Feature-Extraction vs. Fine Tuning

## Token, Segment and Position embedding

## Attention mechanism

## Transformer

## Permutation-Based Language Modelling

## Scheduling Learning Rate

## Label Smoothening

## Parameter Sharing

## Sentence Order Prediction (SOP)

## Embedding Factorization

## Precision

## Recall

## F1 score

## AUC
