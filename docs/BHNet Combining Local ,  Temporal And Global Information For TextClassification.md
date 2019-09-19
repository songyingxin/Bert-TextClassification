# BHNet: Combining Local ,  Sequential And Global Information For TextClassification

tags: 我的论文

---

## Abstract

Pre-training language model has achieved amazing results in many textclassification tasks. In Particular, BERT (Bidirectional Encoder Representations from Transformers) create a new era in NLP tasks. Despite the success, these model perform well at Global Information but weak on local and Sequential information.  In this paper, we conduct exhaustive experiments of  classical text classification models upon BERT in text classification task and provide a general guide for BERT+ models. Finally, we proposed a new text classification model called MIHNet(Multi-dimension Information Integration using Highway network)  , which integrates Global, Local and Sequential information together. Notably, our model obtains new state-of-the-art results on eight widely-studied text classification datasets.

## 1. Introduction

Text Classification is a classic and important problem in Natural Language Processing(NLP). The task is to assign predefined categories to a given text sequence.  In recent years, neural network models that can make use of text representation have been show to be effective for text classification, including convolution models[1] [2] [3] [4] [5] [6], recurrent models[7] [8] [9], and attention mechanisms[10] [11]. 

In Natural Language Processing, pre-trined language representations are beneficial for text classification and other tasks, which can avoid training a new model from scratch. And there are two existing strategies for applying pre-trained language representation to downstream tasks: feature-based and fine-tuning. The featured-based approach including  word embeddings ,such as word2vec[12] , Glove[13],  Cove[14] and ELMo[15]. The fine-tuning approach, such as BERT[16], OpenAI GPT[17], has brought a new breakthrough in Natural Language Processing(NLP) and  become one of the hottest research topics.

Although BERT has achieved amazing results in many natural language understanding(NLU) tasks, its potential has yet to be fully explored. There is little research to explore what can we do upon BERT to improve the performance on target tasks futher.

In this paper, we investigate how to maximize the utilization of BERT+ models for the text classification task. We explore several text classification models upon BERT to enhance its performance on text classification task. We design exhaustive experiments to make a detailed analysis of BERT+ models and proposed  a new model called MIHNet , which integrate  Global, Local, Sequential information together to enhance performance further.

The contributions of our paper are as follows:

- We analyzed the effect of text length on BERT and proposed a advice to choose the appropriate length based on the length distribution.
- We investigate the performance of several text classification models upon BERT and propose a general solution to use those models upon BERT for different datasets.
- We propose a new model named MIHNet and achieve the state-of-the-art results on seven widely-studied english text classification datasets and one Chinese news classification dataset.

## 2. Related Work

Transfer learing that learn knowledge from the other tasks has a rising interest in the field of NLP and achieve the new the-state-of-art result in most text classification datasets. We briefly review the two field: pre-trained language model  and Text classification.

### 2.1 Pre-trained Language  model

Feature-based approaches that learning widely applicable representations of words has been an active area of research for decades, including Word2Vec[12], Glove[13].  Those pre-trained word embeddings are now an important  part of modern  NLP systems, offering significant improvements over embeddings learned from scratch. But, there is a serious problem in those classicial word embedding approch : Word ambiguity problem. 

ELMo[15] propose to extract context-sentive features from deep language model to solve word ambiguity problem and achieve new state-of-the-art results in many tasks. This approch prove that deep neural language model can achivee better performance than shadow networks. 

Recently, the method of pre-trained language models on deep language model with a large amount of unlabeled data and fine-tuning in down-stream tasks has made a big breakthrough in several natural language understanding tasks, such as OpenAI GPT[17], Bert[16], GPT 2.0[18]. BERT is now the hottest pre-trained  language model in NLP. It is pre-trained on Masked Bidirectional Language Model Task and Next Sentence Prediction via a large cross-domain corpus. BERT is the first fine-tuning based representation model that achieves state-of-the-art results for a range of NLP tasks. 

Pre-trained Language Model is the hottest research topic in NLP,  some recent models explore language model research along different dimension. ERNIE[19] and ERNIE[20] propose to integrate knowledge information into pre-trained language model. MASS[21] and UNILM[22] explore how to use pre-trained language model in Natural Language Generation. But BERT is still the best pre-trained language model in Natural Language Understanding.

In this paper, we have futher explored how to add different models upon BERT for text classification.

### 2.2 Text classification





## 3. Text Classification models upon Bert

### 3.1 The effect of text length

For text classification, choosing the right text length is critical for BERT. Proper text length settings can reduce training time while achieving optimal results. In the experiment, we limited the text length to 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, respectively, and tested on multiple data sets, and the text length distribution of these data sets themselves is also very different.

### 3.2   BERT+ for Text Classification

When adapt BERT fine-tuning strategy to text classification task, we always get a better result than classic text classification models in most datasets. To improve model performance further, a proper strategy combined with BERT is desired. In this paper, we look for the proper combination strategy in the following models.

There are two kinds of BERT models, which have a big difference in model size and poarameter quantity. BERT-base model contains an encoder with 12 Transformer blocks, 12 self-attention heads, 110M parameters and the hidden size of 768. BERT-large model contains an encoder with 24 Transformer blocks, 16 self-attention heads, 340M parameters and the hidden size of 1024.

BERT takes an input of a sequence of noT more than 512 tokens and outputs the representation of the sequence. In the section, the input sentence is like that: `[CLS] + sentence`, where [CLS] contains the special classification embedding.  
$$
h, O = BERT(S)
$$
The output of BERT consists of two parts, where $h \in R^d $ is the final hidden state of the first token [CLS] and $O \in R^{|S| \times d}$ is the context representation matrix of the sequence. Beyond on this, i conducted a detailed experiment that explored the performance of different models.

1) **BERT : ** In this model, we taks the final hidden state $h$ as the representation of the whole sequence. A simple softmax classifier is added to the top of BERT to predict the probability of label c:
$$
p(c|h) = \text{softmax}(Wh)
$$
where $W$ is the task-specific parameter matrix. We fine-tune all the parameters from BERT as well as W jointly by maximizing the log-probability of the correct label.

2) **BERT + TextCNN**: In this model, we tasks all the final hidden state matrix $O$ of the setence tokens as the representation of the whole sequence. Then we apply TextCNN to extract the local features of $O$  to get the final representation $h_{cnn}$.  
$$
h_{cnn} = \text{TextCNN}(O)  \\
p(c|h_{cnn}) = \text{softmax}(W_{cnn}h_{cnn})
$$
3) **BERT + LSTM: **  We apply Bi-LSTM to enhance the sequential information considering that Transformer adopts Position Embedding to capture sequential information. Finally, we take the last token hidden state $h_{lstm}$ of LSTM as the final representation and feed it into softmax classifier. The output layer is like that:
$$
h_{lstm} = \text{LSTM}(O) \\
p(c|h_{lstm}) = \text{softmax}(W_{lstm}h_{lstm})
$$
where $W_{lstm}$ is the specific parameter matrix for BERT + LSTM.

4) **BERT + RCNN:**  We use bidirectional LSTM to get the contextual representation of each token: $c^l_i, c_i^r$. To get the final representation  of token, we concat the output of bidirectional LSTM and the output of BERT:
$$
x_i = [c_i^l; o_i, c_i^r]
$$
We then apply a tanh activation function and max-pooling to ge the final representation of the input sentence:
$$
h_i = tanh(W_hx_i + b_h) \\
y = max_{i=1}^n h_i
$$
Finally, we use softmax classifier to get the probability of each class:
$$
p(c|y) = \text{softmax}(W_{rcnn}y)
$$
5) **BERT + Attention: ** We add an Attention Layer upon $O$ to get the final representation of the sentence and feed the representation $v$  into the softmax classifier.
$$
\alpha_i = \frac{exp(O_i^Tu_s)}{\sum_i exp(O_i^Tu_s)} \\
v = \sum_i \alpha_i O_i \\
p(c|v) = softmax(W_v v)
$$
6) **BERT +  DPCNN: **



7) **BERT + VDCN: **

### 3.2 Dealing with long texts

The first question of applying BBERT to text classification is how the performance of the model changes as the text length window expands. Since the maximum length of the BERT is 512, we try the following text lengths:  [50, 100, 150, 200, 250, 300, 350, 400, 450, 500].

As mentioned above, the maximum sequence length of BERT is 512.  How to processing the text with a length larger than 512 is important. We try the following ways for dealing with long articles different from [28].

### 3.2  BHNet



## 5. Experiments

### 5.1 Datasets

 We explore all approach on nine widely-studied datasets.These datasets have varying numbers of documents and varying document lengths, covering three common text classification tasks: sentiment analysis, question classification, and topic classification. We show the statistics for each dataset in Table 1.

**Sentiment analysis** For sentiment analysis, we use the binary film review IMDB dataset[23],  two-class version of the SST dataset[24] and five-class version of the Yelp review dataset[25].

**Question classification** For question classification, we evaluate our method on the six-class version of the TREC dataset[26] and Yahoo! Answers dataset[27].

**Topic classification**  For topic classification, we use large-scale AG's News and DBPedia as English  datasets. To test the effectiveness of BERT for Chinese text, we create the chinese dataset using the THUCNews corpus.



### 5.2  Hyperparameters



### 5.3 Results



## 6.  Conclusions





## 论文结构参考



## Reference

[1] A convolutional neural network for modelling sentences.

[2] Character-level convolutional networks for text classifification.

[3] Very deep convolutional networks for natural language processing.

[4] Deep pyramid convolutional neural networks for text categorization.

[5] Deconvolutional paragraph representation learning.

[6] Deconvolutional latent-variable model for text sequence matching.

[7] Recurrent neural network for text classifification with multi-task learning.

[8] Generative and discriminative text classifification with recurrent neural networks.

[9] Neural speed reading via skimrnn.

[10] Attention is all you need.

[11] A structured self-attentive sentence  embedding

[12] Distributed representations of words and phrases and their compositionality.

[13] Glove: Global vectors for word representation.

[14] Learned in translation: Contextualized word vectors.

[15] Deep contextualized word representations.

[16] BERT: Pre-training of Deep Bidirectional Transformers for  Language Understanding

[17] Improving language understanding with unsupervised learning.

[18] Language Models are Unsupervised Multitask Learners

[19] ERNIE: Enhanced Language Representation with Informative Entities

[20] ERNIE: Enhanced Representation through Knowledge Integration

[21] MASS: Masked Sequence to Sequence Pre-training for Language Generation

[22] Unifified Language Model Pre-training for  Natural Language Understanding and Generation

[23] Learning word vectors for sentiment analysis

[24] Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank

[25] Character-level convolutional networks for text classification.

[26] The trec-8question answering track evaluation.

[27] Character-level convolutional networks for text classification.

[28] How to Fine-Tune BERT for Text Classification?