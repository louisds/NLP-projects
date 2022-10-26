# 1) Linguistics and NLP Tasks

## Lexicon


## Pragmatics vs. Semantics vs. Syntax vs. Morphology vs. Phonology vs. Phonetics

![](https://github.com/louisds/NLP-projects/blob/main/images/semantic_pragmatic.png | width=1)

**Pragmatics** identifies the meaning of words and phrases based on how language is used to communicate (i.e. emphasis on their context as well). Unlike semantics, which only concerns the meaning of the words, pragmatics goes a step further by looking at the same word in relation to its context. Thus, pragmatics explains how language users are able to overcome apparent ambiguity since it explains the meaning relies on the manner, time, place, etc. of an utterance.

**Semantics** is the literal meaning of words and phrases. It has two main categories: lexical semantics and phrasal semantics. Lexical semantics concerns the meanings of words and the meaning of relationships among words, i.e. analyze words and see how they can be related to each other with relations to synonyms, antonyms, homonyms, polysemy, figures of speech. Phrasal semantics concerns the meaning of syntactic units, i.e. concepts such as paraphrase, contradiction, ambiguity, mutual entailment, etc.

For example, the sentence – “He is so cool”. Semantically, this sentence can be interpreted as – He is very nice, a compliment to the person, which is the literal meaning. But under pragmatics, this sentence suggests the context: the positive attitude of the speaker towards the person. This is the intended or the inferred meaning in the sentence.

**Syntax** is the set of rules needed to ensure a sentence is grammatically correct (article, noun, verb, ...). Syntax refers to grammar, while semantics refers to meaning. 

**Morphology** analyzes the structure of words and parts of words such as stems, root words, prefixes, and suffixes (i.e. how words are formed).

**Phonology** concerns the study of more complex and abstract sound patterns and structures (syllables, intonation, etc.), i.e. different patterns of sounds in different positions in words etc.

**Phonetcs** studies how humans produce and perceive sounds. 

## Dependency (Syntactic) vs. Constituency Parsing

Both methods are types of sentence parsing, which is done by taking a sentence and breaking it down into different parts of speech. The words are placed into distinct grammatical categories, and then the grammatical relationships between the words are identified. This creates a parse tree that highlights the syntactical structure of a sentence according to a formal grammar. Both methods use different types of grammars and have different assumptions.

**Dependency parsing** expresses the syntax of the sentence in terms of dependencies between words. The parse tree is a graph G = (V, E) where the set of vertices V contains the words in the sentence, and each edge in E connects two words. Each edge in E has a type, which defines the grammatical relation that occurs between the two words.

**Constituency Parsing** is based on the formalism of context-free grammars. In this type of tree, the sentence is divided into constituents, that is, sub-phrases that belong to a specific category in the grammar. In English, for example, the phrases “a dog”, “a computer on the table” and “the nice sunset” are all noun phrases, while “eat a pizza” and “go to the beach” are verb phrases. This a difference with dependency parsing, which looks at the individual words.

## Lexical vs. Structural Ambiguity

Lexical ambiguity occurs when a word has more than one meaning, while structural ambiguity arises when the sentence’s meaning is unclear because of how words are related. Some examples:

**Lexical ambiguity**: “I went to the bank” There are several dictionary meanings for the word “bank” Here, you can’t know if “bank” is a place where money is held or a riverbank.

**Structural ambiguity**: “The chicken is ready to eat” You can’t tell if the chicken will do the eating or if the chicken will be eaten. 


## Taxonomy and Ontology

Taxonomy and ontology work in different ways to apply structure to language. 

**Taxonomy** identifies the hierarchical relationships among concepts and specifies the term to be used to refer to each, i.e. it prescribes structure and terminology. A taxonomy is static and classifies words into categories and sub-categories (i.e. chicken is a bird which is an animal which is a living organism on earth).

**Ontology** identifies and distinguishes concepts and their relationships, i.e. it describes content and relationships. An ontology is dynamic and domain-centric and provides types, properties, and inter-relationships of the words.

Together, they help disambiguate language so programs can perform with more accuracy. In the example “The chicken is ready to eat”, the taxonomy can classify the chicken as an animal, while ontology can apply relational understanding to further define the term “chicken” as either an animal or a type of food to be eaten.


## Information Extraction

## Entity Recognition vs. Disambiguation vs. Linking

## Document Retrieval and Search

## Relation Extraction

# 2) NLP Basics

## Language Model

In many Natural Language tasks you have a language (L) and want to build a model (M) for the language. In NLP we think of language L as a process for generating text, where the process is a collection of probability distributions. Given a history $h$ consisting of a series of previous words in a sentence, the language $L$ is the probability that the next word is $w$. Similarly, given an entire sentence $s$, we can evaluate $L(s)$ as the probability of the sentence occurring. A language model M will try to replicate this language, which we can then use for various tasks. 

$$
p(w_1, ..., w_n) = p(w_1) \cdot p(w_2|w_1) ... p(w_n|w_1, ..., w_{n-1})
$$

## Source Channel Framework

![alt text](https://github.com/louisds/NLP-projects/blob/main/images/noisy_channel_model.png)

The source channel framework or noisy channel framework is a framerwork used in spell checking, speech recognition, translation, POS tagging, etc. It consists of a message (input X) that is encoded and fed into a (noisy) channel. The output from the noisy channel (output Y) is then decoded (decoded message X'), attempting to reconstruct the original message based on this noisy output. In NLP we do not usually act on encoding. The problem is reduced to decoding the noisy output Y for getting the most likely input given the output. 

An example of a noisy channel for spell checking is a channel that scrambles some letters in an input sentence (and thus producing spelling errors). In this case, the correct language (real language) is the input X, while the output Y is the language with errors (observed language). The probability P(X) is the probability of the word sequence in the correct language (language model probability). The probability P(Y|X) is the probability that we have a specific scrambled output sentence Y, giving a specific input sentence X (noisy channel probability). The goal of the framework is the obtain X', based on the probability P(X|Y): 

$$
X' = argmax_x \ P(x|Y) = argmax_x \ \frac{p(x) p(Y|x)}{P(Y)} \propto argmax_x \ p(w) p(Y|x)
$$

## N-grams

If we have a sequence of $n$ words, the number of histories (consisting of a series of previous words) grows as $|V|^{n-1}$, while the number of parameters in our model grows as $|V|^{n}$, where V is our vocabulary. To reduce the number of parameters, we group histories with a particular grouping method. N-gram language modeling is a popular grouping method, where the probability of a given sequence of N words is predicted. For example in case of a trigram (3-gram), we ignore everything except for the previous two words when we try to predict the next word (number of parameters is of the order $|V|^{3}$).

Despite the fact that a higher-order n-gram model, in theory, contains more information about a word's context, it cannot easily generalize to other data sets (i.e. overfitting) because the number of events (i.e. n-grams) it has seen during training becomes progressively less as n increases. On the other hand, a lower-order model lacks contextual information and so may underfit your data.

## Perplexity

We want our language model $M$ to assign high probabilities to sentences that are real and synthetically correct. The best model is the one that assigns the highest probability to the test set. The perplexity is a measure to quantify how "good" a language model is, based on a test (or validation) set. The perplexity on a sequence $s$ of words is defined as:

$$
Perplexity(M) = M(s)^{(-1/n)} = \left( p(w_1, ..., w_n)\right)^{(-1/n)}
$$

The intuition behind this metric is that if a model assigns a high probability to a test set, it is not surprised to see it (not perplexed by it), which means the model M has a good understanding of how the language L works. Hence, a good model has a lower perplexity. The exponent (-1/n) in the formula is just a normalizing strategy (geometric average), because adding more sentences to a test set would otherwise introduce more uncertainty (i.e. larger test sets would have lower probability). So by introducing the geometric average, we have a metric that is independent of the size of the test set. 

As an example: a perplexity of 100 means that, on average, the model predicts as if there were 100 equally likely words to follow. In theory, the optimal situation would be a perplexity of 1, where the model always knows exactly which word to predict. However, this means there could only be one possible sentence in a language, which is quite boring. The generated text is either repetitive or incoherent. Overall, it is hard to provide a benchmark for perplexity, as this depends on the vocabulary size and also the fact that we rather stay closer to human perplexity (instead of 1). 

## Pointwise Mutual Information (PMI)

PMI is a metric that quantifies how likely specific words/clusters are to co-occur together within some window, compared to if they were independent. It helps to understand whether two (or more) words actually form a unique concept. If this is the case, we can reduce the dimensionality of our task, since that couple of words (called bigram- or n-grams in the general case for n words) can be considered as a single word. Hence, we can remove one vector from our computations. The bigram "New York" is an example: the two words form a whole, but "new" probably also appears in a lot of other parts within the text. The idea of PMI is that we want to quantify the likelihood of co-occurrence of two words, taking into account the fact that it might be caused by the frequency of the single words. It is defined as:

$$
PMI(w_1, w_2) = log\left( \frac{p(w_1, w_2)}{p(w_1) p(w_2)}  \right) 
$$

So PMI computes the (log) probability of co-occurrence scaled by the product of the single probability of occurrence. When the ratio equals 1 (hence the log equals 0), it means that the two words together don’t form a unique concept: they co-occur by chance. On the other hand, if either one of the words (or even both of them) has a low probability of occurrence if singularly considered, but its joint probability together with the other word is high, it means that the two are likely to express a unique concept.

## Cosine Similarity

## Inverse Document Frequency (IDF)

## Term Frequency–Inverse Document Frequency (TF-IDF)

## Bag of words (BOW)

## Skip gram

## Pragmatic/Lexical/Syntactic/Semantic Ambiguity

## Masked Language Model

## POS tagging

## Ensemble Method

## Name Entity Recognition (NER)

## Lemmatization

## Stemming

## Tokenization

## Soundex

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
