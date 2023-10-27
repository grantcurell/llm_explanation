
- [\\end{pmatrix}](#endpmatrix)
      - [Output Embedding: " de nada"](#output-embedding--de-nada)
- [\\end{pmatrix}](#endpmatrix-1)
      - [Adding Them Together](#adding-them-together)
    - [Encoder Stack Processing](#encoder-stack-processing)
    - [Prepare and Embed Target Sequence for Decoder](#prepare-and-embed-target-sequence-for-decoder)
    - [Decoder Stack Processing with Encoder Output](#decoder-stack-processing-with-encoder-output)
    - [Output Layer for Word Probabilities](#output-layer-for-word-probabilities)
    - [Loss Function and Back-Propagation](#loss-function-and-back-propagation)
    - [Supplemental Information](#supplemental-information)
      - [Word Embeddings](#word-embeddings)

## Overview

The purpose of this whitepaper is to explain at a medium to low level how LLMs work such that we can make informed decisions about their performance and architectural decisions.

## How Transformers Work

We start by covering how transformers work. Transformers are the core of an LLM model. Below we discover the traditional transformer architecture. GPT-style models do not use the encoders and have what is called a decoder only architecture, but it is difficult to understand that architecture without understanding where it comes from so for this section we cover how the encoder works as well as the decoder.

[This article series](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452) does a fantastic job of explaining the overarching LLM learning process but there were several parts I didn't immediately understand and moreover it did not provide concrete mathematical examples to illustrate the concepts. I go through and explain the pieces of the article which didn't make sense to me and I have added concrete examples to illustrate the process mathematically.

The below image is an overview model learning and inference process. There are some minor differences between the two processes which I will explain below but at this level they are the same. To illustrate how LLMs work, we will use the example of asking an LLM to translate, "You are welcome" to "De Nada".

![](images/2023-10-26-11-41-04.png)

[Image Source](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)

These are the high level steps which must take place.

1. The input sequence is converted into Embeddings (with Position Encoding) and fed to the Encoder.
2. The stack of Encoders processes this and produces an encoded representation of the input sequence.
3. The target sequence is prepended with a start-of-sentence token, converted into Embeddings (with Position Encoding), and fed to the Decoder.
4. The stack of Decoders processes this along with the Encoder stack’s encoded representation to produce an encoded representation of the target sequence.
5. The Output layer converts it into word probabilities and the final output sequence.
6. The Transformer’s Loss function compares this output sequence with the target sequence from the training data. This loss is used to generate gradients to train the Transformer during back-propagation.

This overview was taken from [here](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)

### 1 - Input Sequence to Encoder Embeddings

Let's assume you have a word embedding model that maps each word in the sentence "You are welcome" to a 4-dimensional vector. I use arbitrary numbers for demonstration.

1. **Tokenization**:
    - The sentence "You are welcome" is tokenized into `['You', 'are', 'welcome']`.

2. **[Word Embeddings](#word-embeddings)**:
    - See [Word Embeddings](#word-embeddings) for an explanation of how these work.
    - Assume the embedding model maps 'You' to $[0.1, 0.2, -0.1, 0.4]$.
    - 'are' is mapped to $[-0.3, 0.5, 0.1, -0.2]$.
    - 'welcome' is mapped to $[0.4, -0.3, 0.2, 0.1]$.

3. **Input Matrix \(X\)**:
    - The vectors are stacked to form the input matrix \(X\):
    $$
    X = \begin{pmatrix}
    0.1 & 0.2 & -0.1 & 0.4 \\
    -0.3 & 0.5 & 0.1 & -0.2 \\
    0.4 & -0.3 & 0.2 & 0.1
    \end{pmatrix}
    $$

This \(X\) matrix serves as the input to the neural network, and each row corresponds to the embedding of a word in the sentence "You are welcome". 

Now we need to do the same thing for the output embedding. In many sequence-to-sequence models like Transformers used for tasks like machine translation, a special start token (often denoted as `<s>`, `<start>`, or `[START]`) and sometimes an end token (e.g., `<e>`, `<end>`, or `[END]`) are added to sequences. These tokens provide signals for the beginning and end of sequences and help in the generation process.

1. **Tokenization with Start Token**:
    - The phrase "de nada" becomes `['<start>', 'de', 'nada']`.

2. **Word Embeddings**:
    - Assume our embedding model maps `<start>` to $[0.0, 0.0, 0.0, 0.0]$ (just as a placeholder; in practice, it would have a unique representation).
    - 'de' is mapped to $[-0.2, 0.4, 0.3, 0.1]$.
    - 'nada' is mapped to $[0.5, -0.1, -0.4, 0.3]$.

3. **Output Matrix \(Y\)** with the start token:
    $$
    Y = \begin{pmatrix}
    0.0 & 0.0 & 0.0 & 0.0 \\
    -0.2 & 0.4 & 0.3 & 0.1 \\
    0.5 & -0.1 & -0.4 & 0.3
    \end{pmatrix}
    $$

The inclusion of the start token helps the model recognize the beginning of the output sequence. If there's an end token, it can similarly indicate the end of the sequence, especially useful in generation tasks. What I don't show here is a padding but in the actual models you would likely also have a pad. Ex: `['<start>', 'de', 'nada', '<pad>', '<pad>']` to make sure that the input sequences are the same size. This is a feature of the traditional transformer model but will not appear in the GPT-style models.

### 1.5 Position Encoding

In previous models (Recurrent Neural Networks [RNNs] typically), the position of words in a sentence and their mathematical importance were fixed by virtue of the fact that those models operated on each word sequentially. Transformers on the other hand process all words in a batch at the same time drastically reducing training/inference time. This presents a problem because word order matters. Ex: "The cat sat on the mat." is not the same as "The mat sat on the cat." It is important for us to include the position within the sentence as a value within our framework.

To remedy this, Transformers incorporate a mechanism called Position Encoding. This mechanism is designed to infuse the sequence with positional information, ensuring the model can distinguish between "cat sat on the mat" and "mat sat on the cat".

Just as there was an embedding layer for input and output there are also position encoding layers for both input and output. Importantly, the Position Encoding doesn't rely on the specific words in a sequence. Instead, it assigns a unique encoding to each possible position in the sequence. These encodings are predetermined and consistent across all sequences. These encodings are generated using the following formulas:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

- where sin is used for even positions and cos is used for odd positions.
- $pos$: Position of the word in the sequence
- $d_{\text{model}}$: Dimensionality of the embeddings
- $i$: Index for the dimension ranging from 0 to $d_{\text{model}}/2 - 1$

The brilliance of this design lies in its ability to provide rich positional context without being tied to the content of the sequence. This ensures that no matter what words are present, the Transformer always understands their order. To continue our other examples:

Alright, let's compute the position encodings using the given equations for both input and output embeddings.

Given:
- $d_{\text{model}}$ is the dimension of the model, which in our case from the earlier examples is 4.
- The $pos$ variable represents the position of the word in the sequence.
- The $i$ variable ranges from 0 to $d_{\text{model}}/2-1$. Since $d_{\text{model}}$ is 4, $i$ will range from 0 to 1.

#### Input Embedding: "You are welcome"

For the input sequence, we have 3 words, so the positions are $pos = 0, 1, 2$.

Using the given equations:

1. For $pos = 0$:
$$
PE_{(0, 0)} = \sin\left(\frac{0}{10000^{2(0)/4}}\right) = 0\space,\space
PE_{(0, 1)} = \cos\left(\frac{0}{10000^{2(0+1)/4}}\right) = 1\space,\space
PE_{(0, 2)} = \sin\left(\frac{0}{10000^{2(1)/4}}\right) = 0\space,\space
PE_{(0, 3)} = \cos\left(\frac{0}{10000^{2(1+1)/4}}\right) = 1\space,\space
$$

To break that down further:

- $pos = 0$
- $d_{\text{model}} = 4$

1. **For $PE_{(0,0)}$**
    Using the formula for even indices (2i):
    $$i = 0$$
    $$PE_{(0, 2(0))} = \sin\left(\frac{0}{10000^{2(0)/4}}\right)$$
    Since $\sin(0)$ is 0, the value is 0.

2. **For $PE_{(0,1)}$**
    Using the formula for odd indices (2i+1):
    $$i = 0$$
    $$PE_{(0, 2(0)+1)} = \cos\left(\frac{0}{10000^{2(0+1)/4}}\right)$$
    Since $\cos(0)$ is 1, the value is 1.

3. **For $PE_{(0,2)}$**
    Using the formula for even indices (2i):
    $$i = 1$$
    $$PE_{(0, 2(1))} = \sin\left(\frac{0}{10000^{2(1)/4}}\right)$$
    Again, since $\sin(0)$ is 0, the value is 0.

4. **For $PE_{(0,3)}$**
    Using the formula for odd indices (2i+1):
    $$i = 1$$
    $$PE_{(0, 2(1)+1)} = \cos\left(\frac{0}{10000^{2(1+1)/4}}\right)$$
    Once more, since $\cos(0)$ is 1, the value is 1.

Following the same pattern for $pos = 1$ and $pos = 2$, we get:

$$
PE_{\text{input}} = 
\begin{pmatrix}
0 & 1 & 0 & 1 \\
\sin\left(\frac{1}{10000^0}\right) & \cos\left(\frac{1}{10000^{0.5}}\right) & \sin\left(\frac{1}{10000^2}\right) & \cos\left(\frac{1}{10000^{2.5}}\right) \\
\sin\left(\frac{2}{10000^0}\right) & \cos\left(\frac{2}{10000^{0.5}}\right) & \sin\left(\frac{2}{10000^2}\right) & \cos\left(\frac{2}{10000^{2.5}}\right)
\end{pmatrix}
=
\begin{pmatrix}
0 & 1 & 0 & 1 \\
0.8415 & 0.99995 & 0.0001 & 1 \\
0.9093 & 0.9998 & 0.0002 & 1
\end{pmatrix}
$$



#### Output Embedding: "<start> de nada"

For the output sequence, we also have 3 words/tokens. The positions again are $pos = 0, 1, 2$.

Using the equations similarly:

$$
PE_{\text{output}} = 
\begin{pmatrix}
0 & 1 & 0 & 1 \\
\sin\left(\frac{1}{10000^0}\right) & \cos\left(\frac{1}{10000^{0.5}}\right) & \sin\left(\frac{1}{10000^2}\right) & \cos\left(\frac{1}{10000^{2.5}}\right) \\
\sin\left(\frac{2}{10000^0}\right) & \cos\left(\frac{2}{10000^{0.5}}\right) & \sin\left(\frac{2}{10000^2}\right) & \cos\left(\frac{2}{10000^{2.5}}\right)
\end{pmatrix}
=
\begin{pmatrix}
0 & 1 & 0 & 1 \\
0.8415 & 0.99995 & 0.0001 & 1 \\
0.9093 & 0.9998 & 0.0002 & 1
\end{pmatrix}
$$

Finally, to incorporate these position encodings into our embeddings, you would simply add the corresponding position encoding to each row of the embedding matrices $X$ and $Y$.

#### Adding Them Together

Given:

$$
X = \begin{pmatrix}
0.1 & 0.2 & -0.1 & 0.4 \\
-0.3 & 0.5 & 0.1 & -0.2 \\
0.4 & -0.3 & 0.2 & 0.1
\end{pmatrix}
$$

$$
Y = \begin{pmatrix}
0.0 & 0.0 & 0.0 & 0.0 \\
-0.2 & 0.4 & 0.3 & 0.1 \\
0.5 & -0.1 & -0.4 & 0.3
\end{pmatrix}
$$

The position encodings (from our previous calculations) are:

$$
PE_{\text{input/output}} = \begin{pmatrix}
0 & 1 & 0 & 1 \\
0.8415 & 0.99995 & 0.0001 & 1 \\
0.9093 & 0.9998 & 0.0002 & 1
\end{pmatrix}
$$

Adding position encodings to $X$:

$$
X + PE_{\text{input}} = \begin{pmatrix}
0.1 + 0 & 0.2 + 1 & -0.1 + 0 & 0.4 + 1 \\
-0.3 + 0.8415 & 0.5 + 0.99995 & 0.1 + 0.0001 & -0.2 + 1 \\
0.4 + 0.9093 & -0.3 + 0.9998 & 0.2 + 0.0002 & 0.1 + 1
\end{pmatrix}
= \begin{pmatrix}
0.1 & 1.2 & -0.1 & 1.4 \\
0.5415 & 1.49995 & 0.1001 & 0.8 \\
1.3093 & 0.6998 & 0.2002 & 1.1
\end{pmatrix}
$$

Adding position encodings to $Y$:

$$
Y + PE_{\text{output}} = \begin{pmatrix}
0.0 + 0 & 0.0 + 1 & 0.0 + 0 & 0.0 + 1 \\
-0.2 + 0.8415 & 0.4 + 0.99995 & 0.3 + 0.0001 & 0.1 + 1 \\
0.5 + 0.9093 & -0.1 + 0.9998 & -0.4 + 0.0002 & 0.3 + 1
\end{pmatrix}
= \begin{pmatrix}
0.0 & 1.0 & 0.0 & 1.0 \\
0.6415 & 1.39995 & 0.3001 & 1.1 \\
1.4093 & 0.8998 & -0.3998 & 1.3
\end{pmatrix}
$$

These new matrices incorporate both the embeddings and the position information, and they will be used as input to subsequent layers of the Transformer model.


### Encoder Stack Processing

Now we have created our input matrix \(X\) and are ready to start 

TODO - From here, the matrix \(X\) would be used to calculate the Query, Key, and Value matrices in a Transformer model, for example, as part of the attention mechanism and subsequent layers for various NLP tasks.



### Prepare and Embed Target Sequence for Decoder
### Decoder Stack Processing with Encoder Output
### Output Layer for Word Probabilities
### Loss Function and Back-Propagation

### Supplemental Information

#### Word Embeddings

The first question(s) I asked when I saw these matrices is, "Where do these values come from and what are they?" What followed soon after was, "Why do they exist?" Why they exist lends context to where do they come from and what are they.

*Why do they exist*

The article [Large language models, explained with a minimum of math and jargon](https://www.understandingai.org/p/large-language-models-explained-with) does a fantastic job of answering this in a non-mathematical way. The goal here is that we want to describe mathematically the relationship between some given word and other words because this will then allow us to more accurately predict what words should go in the answers our LLM produces. Words that are similar should have similar vector values. This concept was pioneered at Google in [a paper called Word2vec](https://en.wikipedia.org/wiki/Word2vec). You can see this visually at [this website](http://vectors.nlpl.eu/explore/embeddings/en/MOD_enwiki_upos_skipgram_300_2_2021/airplane_NOUN/). For example, I wanted to see what airplanes are likely to be related to airplanes:

![](images/2023-10-26-12-54-59.png)

The percentage similarity to "airplane":

- **aeroplane**: NOUN, 0.8153
- **aircraft**: NOUN, 0.7992
- **plane**: NOUN, 0.7429
- **airliner**: NOUN, 0.7398
- **helicopter**: NOUN, 0.7006
- **aircraave**: NOUN, 0.6568
- **biplane**: NOUN, 0.6540
- **airship**: NOUN, 0.6404
- **dc-3**: NOUN, 0.6297
- **seaplane**: NOUN, 0.6271


*Where do these values come from?*

The answer is it depends. There are three potential places these values may originate from at first:

1. Random Initialization: Here, the embeddings are initialized with small random values. These values then get adjusted during training to capture semantic meanings.
2. Pre-trained Embeddings: Instead of starting with random values, it's common to initialize the embeddings with vectors from pre-trained models like Word2Vec, GloVe, or FastText. These embeddings are trained on vast corpora and capture general language semantics. Depending on the task and the dataset, these pre-trained embeddings can either be kept static during training or be fine-tuned.
3. Training from Scratch: In some cases, especially when the domain-specific language is very different from general language (e.g., medical texts or legal documents), embeddings might be trained from scratch along with the rest of the model.

The values in the embedding are updated during the training phase. Most models today will start with one of the sets of values defined above so you aren't starting from scratch. That also gets you a much more accurate model much faster.

*What are they?*

Realistic answer? We don't really fully understand. We understand parts, but the simple answer is that the model learns values it "thinks" are sensible. There are entire papers written to explain just fractions of the relationships.