# Understanding Automatic Text Summarization

## Introduction

Text summarization is the process of shortening a long piece of text with the goal of keeping its meaning and effect as intact as possible. There are mainly four types of summaries:

1. Single Document Summary: Summary of a Single Document
2. Multi-Document Summary: Summary from multiple documents
3. Query Focused Summary: Summary of a specific query
4. Informative Summary: It includes a summary of the full information

There are also two main approaches to perform text summarization:

1. Extraction-based Summarization
2. Abstraction-based Summarization

**Extraction-based Summarization** picks up the most important sentences and lines from the documents and combines them to create a summary. So every sentence in the summary actually belongs to the original text. This means extractive summarizations don’t require natural language generations and semantic representations (which makes them less complicated).

**Abstraction-based Summarization** uses new phrases and terms, different from the actual document, keeping the points the same (just like how we actually summarize). This approach is mainly based on deep learning, which makes it more complex than the extraction-based approach. 

Lastly, we generally use two types of evaluation methods:

1. Human evaluation
2. Automatic evaluation

Human evaluation speaks for itself. Automatic evaluation, however, is a little bit more complicated. A popular method used in this category is ROUGE (Recall-Oriented Understudy for Gisting Evaluation), where the quality of the summary is determined by comparing it to other summaries made by humans as a reference. The intuition behind this is if a model creates a good summary, then it must have common overlapping portions with the human references. Common versions of ROUGE are: ROUGE-n, ROUGE-L, and ROUGE-SU.




