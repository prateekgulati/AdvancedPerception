
 # Time Sequences: Transformers
*Prateek Gulati*


**Abstract**
This project is an attempt to interpret a transformer by visualizing embeddings from different layers. In this project, I use tweets from TweetEval, a Twitter-specific classification task dataset as sequential data. I feed these tweets to different transformer-based networks and extract embeddings from different layers. These embeddings are transformed into a lower dimensional space for visualization and comparison. The text is preprocessed and tokenized into a list of pairs of sequences before feeding to the model. We see that the embeddings in later layers of a transformer are symmetrical and spread out compared to the initial few layers.

## 1. Introduction

Since the introduction of transformers in the 'Attention is all you need' paper, they have gained a lot of popularity. Multiple different versions and enhancements of transformers have been designed since then. They are used in varied kinds of data modalities like text, speech, audio, image, and video. But a question like how a transformer captures the relationship between its inputs is widely asked. What different layers in a transformer see in a sequence, and what do the different embeddings in each layer look like? In this project, I attempt to push in a direction to answer some of these questions.

The transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output. This method doesn't involve the usage of sequence-aligned methods like RNN or Convolution. The Transformer has a component Multi-Head Attention, that averages attention-weighted positions which reduces the learning of dependency between distant positions to a constant number of operations.

Where d is the dimension

I use three different transformers in this paper – a) BERT: Bidirectional Encoder Representations from Transformers, b) ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, and c) RoBERTa: A Robustly Optimized BERT Pretraining Approach. The pretrained weights for these models are obtained from the hugging face community. For the data I use one of the text attributes from TWEETEVAL: Unified Benchmark and Comparative Evaluation for Tweet Classification with binary labels. For dimensionality reduction, I use t-SNE: t-distributed stochastic neighbor embedding and UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.

The text from the tweets is preprocessed and tokenized so that it can be consumed by the transformer. These byte embeddings are fed to one of the transformers. The embeddings from different layers of the transformer are extracted and reduced to a lower dimension. These low-dimension embeddings are visualized for comparison across different layers, primarily, the initial two and final two layers. With this visualization, the embeddings between different transformers can also be compared.

For the evaluation, the results of these methods are compared with each other. We see that the embeddings from the initial layers of a transformer represent a smaller area in the embedding space while the later layers have more spread-out and uniform embeddings.

## 2. Methods

### 2.1 Dataset

The dataset used for this project is taken from TWEETEVAL: Unified Benchmark and Comparative Evaluation for Tweet Classification. This set has seven different datasets with their corresponding labels. For ease of visualization, I use Irony Detection in English Tweets since it has two labels. The task can be described as a binary irony classification task to define, for a given tweet, whether irony is expressed. I also experimented with some other sets including sentiment analysis (positive, neutral, or negative), Multilingual Detection of Hate Speech Against Immigrants and Women on Twitter (hateful or not hateful), and Identifying and Categorizing Offensive Language in social media (Offensive or not offensive). But the results have been reported for the irony data.

In this project, the focus is not to establish state-of-the-art speech detection or classification, so the labels have been used for visualizing the embedding space from different layers in the transformer.

### 2.2 Data Preparation

Data preparation is a critical component of any task that includes textual data. All the tweets used in this project go through a series of preprocessing steps – a) URLs: removing any links or hyperlinks b) encoding mentions or hashtags, c) decoding emojis: translating Unicode emoji to emoji names, d) string formatting: removal of extra spaces and converting to lowercase

#### 2.2.1 Tokenize

A tokenizer is in charge of preparing the inputs for a model by splitting strings into sub-word token strings. For this, I use the pre-trained tokenizer from the hugging face library. Each transformer has a different tokenizer. For BERT, the tokenizer uses the base uncased version, for Roberta, the tokenizer is base-emotion and for Albert, it's the v2 base.

### 2.3 Transformer

#### 2.3.1 BERT

BERT: that is designed to pre-train deep bidirectional representations from the unlabeled text. It performs joint conditioning on both the left and right contexts in all the layers. I use the base version of this model which has 12 Transformer blocks, 12 self-attention heads, and 768 hidden sizes.

#### 2.3.2 RoBERTa

RoBERTa has almost similar architecture as BERT. It modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates. It uses a byte-level BPE as a tokenizer. The parameters are: β1 = 0.9, β2 = 0.999, ǫ = 1e-6 and L2 weight decay of 0.01

#### 2.3.3 ALBERT

ALBERT builds over BERT by enabling parameter-sharing across the layers, i.e., the same layer is applied on top of each other. This approach slightly diminishes the accuracy, but the more compact size is well worth the tradeoff. Instead of learning unique parameters for each of the 12 layers, it only learns parameters for the first block and reuses the block in the remaining 11 layers. It is trained using the Sentence Order Prediction task. This forces the model to learn finer-grained distinctions about discourse-level coherence properties.

Unless stated otherwise, for all the experiments, I use a batch size of 64 and a maximum sequence length of 50.

### 2.4 Dimensionality Reduction

#### 2.4.1 t-SNE

t-distributed stochastic neighbor embedding (t-SNE) is a statistical method for visualizing high-dimensional data by giving each data point a location in a two or three-dimensional map. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. For ease of representation, I use n\_components as 2, a learning rate of 200, with random initialization.

#### 2.4.2 UMAP

UMAP is a dimension reduction technique that can be used for visualization similarly to t-SNE, but also for general non-linear dimension reduction. It seeks to learn the manifold structure of the data and find a low-dimensional embedding that preserves the essential topological structure of that manifold. For a fair comparison, similar parameters as tSNE are maintained.

### 2.5 Experimentation

After preprocessing all the tweets, the text is tokenized into byte pair encodings. This is fed to the respective. Since the goal of this project is to learn the semantic representation of a transformer, there is no need for training, hence, I freeze the weights of the transformer. It is initialized with pre-trained weights, in the evaluation mode. The model output is captured from all the layers and transformed to a two-dimensional space using TSNE as well as UMAP separately. These embeddings are plotted for comparison.

For the experimentation, I have used an 8 CPU with a v100 Intel GPU enabled with Cuda/11.2 on a Linux operating system.

## 3 Results

![**Figure 1**: Embeddings from BERT, Layer 1, 2, 11 and 12; Row 1: Dimensionality reduction by tSNE, Row 2: Dimensionality reduction by UMAP](Transformer Interpretation/static/Figure 1.png)

 
![**Figure 2**: Embeddings from RoBERTa, Layer 1, 2, 11 and 12; Row 1: Dimensionality reduction by tSNE, Row 2: Dimensionality reduction by UMAP](Transformer Interpretation/static/Figure 2.png)

![**Figure 3**: Embeddings from ALBERT, Layer 1, 2, 11 and 12; Row 1: Dimensionality reduction by tSNE, Row 2: Dimensionality reduction by UMAP](Transformer Interpretation/static/Figure 3.png)

BERT: tSNE results in a more spread-out dimensionality reduction for all four layers. But in both representations, the embeddings in the last layer seem more clustered in comparison to layers 1 and 2.

ALBERT: UMAP results seem more interesting. The embeddings in layer 1 look like a thin cape utilizing a very small area in the embedding space. The last layer embeddings appear more spread out.

RoBERTa: The results of RoBERTa seem most interesting in lower dimensionality space. With each layers 1 to 11, we can see the embeddings gradually spread more evenly in the space. The first two layers result in very unique curves.

## 4 Reflection and Acknowledgments

Deep learning has been very successful, especially in tasks that involve images and texts such as image classification and language translation. But their interpretation has always been a widely asked question. In this work, I take a step towards this. I have experimented with different transformer architectures and tried to represent the embeddings from different layers in a lower dimensional space. A clear distinction can be noticed in the embeddings in the initial layers when compared with embeddings in the penultimate layers. In the first two layers, we see that the embeddings representing our data are only a fraction of the embedding space, yet they all seem to form a curve and are close to each other. This might be due to the fact that every sequence in our data belongs to a similar domain. In the embedding space, a major fraction (white space) exists in the language model but doesn't exist in our data. It is likely that the first few layers try to capture a wide variety of sequential data. In the last two layers, the embeddings are more widely spread and seem more clustered and symmetrical. This likely means that towards the last layer, the transformer's attention is focused primarily on our dataset. Since the model is a sequence classifier, it is more likely for the model to classify sequences if the data is spread more uniformly.

I would like to acknowledge the hugging face community for making the model weight available publicly. I would also like to acknowledge Tanmay Garg [https://github.com/codebuff95](https://github.com/codebuff95) for code to preprocess text.

## References

Devlin, J., Chang, M.W., Lee, K., & Toutanova, K.. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R.. (2019). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V.. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.

Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., & Neves, L. (2020). TweetEval:Unified Benchmark and Comparative Evaluation for Tweet Classification. In Proceedings of Findings of EMNLP.

Laurens van der Maaten, & Geoffrey Hinton (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research, 9(86), 2579–2605.

McInnes, L., Healy, J., & Melville, J.. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.

Tamay Garg [https://github.com/codebuff95](https://github.com/codebuff95)
