
- [Overview](#overview)
- [How Transformers Work](#how-transformers-work)
  - [1 - Input Sequence to Encoder Embeddings](#1---input-sequence-to-encoder-embeddings)
  - [1.5 Position Encoding](#15-position-encoding)
    - [Input Embedding: "You are welcome"](#input-embedding-you-are-welcome)
    - [Output Embedding: " de nada"](#output-embedding--de-nada)
    - [Adding Them Together](#adding-them-together)
  - [Encoder Stack Processing](#encoder-stack-processing)
    - [Self Attention](#self-attention)
      - [A Concrete Example](#a-concrete-example)
    - [Attention Heads and Multi Attention Heads](#attention-heads-and-multi-attention-heads)
    - [Layer Normalization](#layer-normalization)
    - [Feed Forward Neural Net](#feed-forward-neural-net)
  - [Prepare and Embed Target Sequence for Decoder](#prepare-and-embed-target-sequence-for-decoder)
  - [Decoder Stack Processing with Encoder Output](#decoder-stack-processing-with-encoder-output)
  - [Output Layer for Word Probabilities](#output-layer-for-word-probabilities)
  - [Loss Function and Back-Propagation](#loss-function-and-back-propagation)
  - [Supplemental Information](#supplemental-information)
    - [Word Embeddings](#word-embeddings)
    - [Weight Matrices](#weight-matrices)
    - [Softmax](#softmax)
    - [Matrix Multiplication](#matrix-multiplication)
      - [Calculating Y](#calculating-y)
    - [Calculate Attention Score](#calculate-attention-score)
    - [Calculate Multi-Attention Head Output](#calculate-multi-attention-head-output)
    - [Calculate Layer Normalization](#calculate-layer-normalization)
    - [Linear Regression Code](#linear-regression-code)
    - [Plot XOR](#plot-xor)




## Overview

The purpose of this whitepaper is to explain at a medium to low level how LLMs work such that we can make informed decisions about their performance and architectural decisions.

## How Transformers Work

We start by covering how transformers work. Transformers are the core of an LLM model. Below we discover the traditional transformer architecture. GPT-style models do not use the encoders and have what is called a decoder only architecture, but it is difficult to understand that architecture without understanding where it comes from so for this section we cover how the encoder works as well as the decoder.

[This article series](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452) does a fantastic job of explaining the overarching LLM learning process but there were several parts I didn't immediately understand and moreover it did not provide concrete mathematical examples to illustrate the concepts. I go through and explain the pieces of the article which didn't make sense to me and I have added concrete examples to illustrate the process mathematically.

The below image is an overview model learning and inference process. There are some minor differences between the two processes which I will explain below but at this level they are the same. To illustrate how LLMs work, we will use the example of asking an LLM to translate, "You are welcome" to "De Nada".

![](images/2023-10-30-16-38-47.png)

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
0.1 & 0.2 & -0.1 & 0.4 \\\\
-0.3 & 0.5 & 0.1 & -0.2 \\\\
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
0.0 & 0.0 & 0.0 & 0.0 \\\\
-0.2 & 0.4 & 0.3 & 0.1 \\\\
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

For $pos = 0$:

$$
PE_{(0, 0)} = \sin\left(\frac{0}{10000^{2(0)/4}}\right) = 0\space,\space
PE_{(0, 1)} = \cos\left(\frac{0}{10000^{2(0+1)/4}}\right) = 1\space,\space
PE_{(0, 2)} = \sin\left(\frac{0}{10000^{2(1)/4}}\right) = 0\space,\space
PE_{(0, 3)} = \cos\left(\frac{0}{10000^{2(1+1)/4}}\right) = 1\space,\space
$$

To break that down further:

- $pos = 0$
- $d_{\text{model}} = 4$

- **For $PE_{(0,0)}$**
    Using the formula for even indices (2i):
$$i = 0$$
$$PE_{(0, 2(0))} = \sin\left(\frac{0}{10000^{2(0)/4}}\right)$$
    Since $\sin(0)$ is 0, the value is 0.

- **For $PE_{(0,1)}$**
    Using the formula for odd indices (2i+1):
$$i = 0$$
$$PE_{(0, 2(0)+1)} = \cos\left(\frac{0}{10000^{2(0+1)/4}}\right)$$
    
Since $\cos(0)$ is 1, the value is 1.

- **For $PE_{(0,2)}$**
    Using the formula for even indices (2i):
$$i = 1$$
$$PE_{(0, 2(1))} = \sin\left(\frac{0}{10000^{2(1)/4}}\right)$$
    
Again, since $\sin(0)$ is 0, the value is 0.

- **For $PE_{(0,3)}$**
    Using the formula for odd indices (2i+1):
$$i = 1$$
$$PE_{(0, 2(1)+1)} = \cos\left(\frac{0}{10000^{2(1+1)/4}}\right)$$
    
Once more, since $\cos(0)$ is 1, the value is 1.

Following the same pattern for $pos = 1$ and $pos = 2$, we get:

$$ PE_{\text{input}} = \begin{pmatrix} 0 & 1 & 0 & 1 \\\ \sin\left(\frac{1}{10000^0}\right) & \cos\left(\frac{1}{10000^{0.5}}\right) & \sin\left(\frac{1}{10000^2}\right) & \cos\left(\frac{1}{10000^{2.5}}\right) \\\ \sin\left(\frac{2}{10000^0}\right) & \cos\left(\frac{2}{10000^{0.5}}\right) & \sin\left(\frac{2}{10000^2}\right) & \cos\left(\frac{2}{10000^{2.5}}\right) \end{pmatrix} = \begin{pmatrix} 0 & 1 & 0 & 1 \\\ 0.8415 & 0.99995 & 0.0001 & 1 \\\ 0.9093 & 0.9998 & 0.0002 & 1 \end{pmatrix} $$

#### Output Embedding: "<start> de nada"

For the output sequence, we also have 3 words/tokens. The positions again are $pos = 0, 1, 2$.

Using the equations similarly:

$$ PE_{\text{output}} = \begin{pmatrix} 0 & 1 & 0 & 1 \\\ \sin\left(\frac{1}{10000^0}\right) & \cos\left(\frac{1}{10000^{0.5}}\right) & \sin\left(\frac{1}{10000^2}\right) & \cos\left(\frac{1}{10000^{2.5}}\right) \\\ \sin\left(\frac{2}{10000^0}\right) & \cos\left(\frac{2}{10000^{0.5}}\right) & \sin\left(\frac{2}{10000^2}\right) & \cos\left(\frac{2}{10000^{2.5}}\right) \end{pmatrix} = \begin{pmatrix} 0 & 1 & 0 & 1 \\\ 0.8415 & 0.99995 & 0.0001 & 1 \\\ 0.9093 & 0.9998 & 0.0002 & 1 \end{pmatrix} $$

Finally, to incorporate these position encodings into our embeddings, you would simply add the corresponding position encoding to each row of the embedding matrices $X$ and $Y$.

#### Adding Them Together

Given:

$$
X = \begin{pmatrix}
0.1 & 0.2 & -0.1 & 0.4 \\\\
-0.3 & 0.5 & 0.1 & -0.2 \\\\
0.4 & -0.3 & 0.2 & 0.1
\end{pmatrix}
$$

$$
Y = \begin{pmatrix}
0.0 & 0.0 & 0.0 & 0.0 \\\\
-0.2 & 0.4 & 0.3 & 0.1 \\\\
0.5 & -0.1 & -0.4 & 0.3
\end{pmatrix}
$$

The position encodings (from our previous calculations) are:

$$
PE_{\text{input/output}} = \begin{pmatrix}
0 & 1 & 0 & 1 \\\\
0.8415 & 0.99995 & 0.0001 & 1 \\\\
0.9093 & 0.9998 & 0.0002 & 1
\end{pmatrix}
$$

Adding position encodings to $X$:

$$
X + PE_{\text{input}} = \begin{pmatrix}
0.1 + 0 & 0.2 + 1 & -0.1 + 0 & 0.4 + 1 \\\\
-0.3 + 0.8415 & 0.5 + 0.99995 & 0.1 + 0.0001 & -0.2 + 1 \\\\
0.4 + 0.9093 & -0.3 + 0.9998 & 0.2 + 0.0002 & 0.1 + 1
\end{pmatrix}
= \begin{pmatrix}
0.1 & 1.2 & -0.1 & 1.4 \\\\
0.5415 & 1.49995 & 0.1001 & 0.8 \\\\
1.3093 & 0.6998 & 0.2002 & 1.1
\end{pmatrix}
$$

Adding position encodings to $Y$:

$$
Y + PE_{\text{output}} = \begin{pmatrix}
0.0 + 0 & 0.0 + 1 & 0.0 + 0 & 0.0 + 1 \\\\
-0.2 + 0.8415 & 0.4 + 0.99995 & 0.3 + 0.0001 & 0.1 + 1 \\\\
0.5 + 0.9093 & -0.1 + 0.9998 & -0.4 + 0.0002 & 0.3 + 1
\end{pmatrix}
= \begin{pmatrix}
0.0 & 1.0 & 0.0 & 1.0 \\\\
0.6415 & 1.39995 & 0.3001 & 1.1 \\\\
1.4093 & 0.8998 & -0.3998 & 1.3
\end{pmatrix}
$$

These new matrices incorporate both the embeddings and the position information, and they will be used as input to subsequent layers of the Transformer model. One more thing you should take from this is that the contents of the input are independent of the position embedding. The only thing that matters is the position of the word for the position embedding. What we now have is a matrix that contains information on both the relationships of the word and its position in the sentence.

This is visualized well in [this post](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34)

![](images/2023-10-27-12-19-11.png)

It's worth noting in our example we are only showing a single line of text but in reality you would batch multiple sets of text together in the traditional transformer model. The shape of the matrix will remain unchanged until we reach the final output layer. The other bit of complexity you see in the figure above that we don't here is that realistically the word vector matrix describing the word's relationships would be multidimensional but here we used a basic, two dimensional, matrix.

Now we are ready to move onto encoding.

### Encoder Stack Processing

The encoding process looks like this at a high level as illustrated [here](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34):

![](images/2023-10-27-12-26-04.png)

At a high level this is what each component does:

- **Self-Attention**: This mechanism allows the model to weigh the importance of different words in the sequence relative to each other. It outputs a matrix of the same size which is a combination of input vectors based on their attention scores.
- **Layer Norm**: After the self-attention mechanism, the output undergoes a layer normalization to stabilize the activations and assist with training.
- **Feed-forward**: The normalized output is passed through a feed-forward neural network. This is present in each transformer block and helps in further transformation of the data.
  - One of the things that wasn't immediately obvious to me was why this exists. It does a couple of things. First, this is where most of the "learning" happens. It is in the feedforward network that the complex relationships between different tokens (words) and general ideas are really stored. Each word is independently transformed by some linear transformation (IE: a matrix operation of some variety be it addition, subtraction, multiplication, division). Secondly, it introduces non-linearity. If you aren't heavy into math it may not be immediately obvious why you care about this. If you have a linear model its performance ends up being roughly the same as some sort of linear regression. That's fine for basic statistical analysis but that's what we have been doing for decades and is hardly going to do anything earth shattering for you. By making the model non-linear it allows it to create a function which more closely approximates complex relationships. Said in a non-mathy way, it is the juice that gets you the cider that is something as fantastic as ChatGPT. Third, it provides depth. What is depth is the question I first had immediately following this explanation. Imagine your model is evaluating pictures. The lowest level might learn rough outlines, maybe some colors or textures, but that's it. Maybe the picture is of a face and at the moderate levels of depth it starts to learn what an eye or a nose look like. Finally, at the deepest levels the model figures out who the picture is of or identifies whether the person is happy or sad.
- **Layer Norm**: Post feed-forward operation, another layer normalization is performed.

#### Self Attention

The purpose of the self attention portion of the algorithm is explained fantastically in [this paper by Troy Wang](https://www.cis.upenn.edu/wp-content/uploads/2021/10/Tianzheng_Troy_Wang_CIS498EAS499_Submission.pdf#page=11).

> Self-attention is one of the key differentiating characteristics of the transformer model. It is a
> critical component that enables the Transformer to comprehend language contexts intelligently
> [VSP+17].
> The objective of attention can be clearly illustrated by a simple example. Given an input such as
> “Professor Marcus gave some really good advice to Troy, one of his students, because he has
> extensive experiences in the academia.” For a human reader, it is clear that the word “he” in the
> second half of the sentence refers to “Professor Marcus,” instead of his student. But for a computer
> program, it’s not so apparent. There are many such circumstances where grammatical ambiguities
> legally exist, such that rule-based and hard-coded logic would not be sufficient for effective
> language analysis and comprehension [VSP+17][Alammar18].
> This is where self-attention comes into play. When the model processes each token input, self-
> attention enables the model to associate the meaning of the current token with other tokens, such
> as associating “he” with “Professor Marcus” in our previous example, in order to gain better
> knowledge of this current input. In other words, the transformer model learns how to pay attention
> to the context thanks to the self-attention mechanism. It turns out that natural language has a lot of
> sequential dependencies, and thus the ability to incorporate information from previous words in
> the input sequence is critical to comprehending the input [VSP+17][Alammar18].

The actual math of this is a bit confusing so I found it was best to read it first from [our friend Troy Wang](https://www.cis.upenn.edu/wp-content/uploads/2021/10/Tianzheng_Troy_Wang_CIS498EAS499_Submission.pdf#page=11) and then I'll provide a more detailed explanation where I really break down what he is saying.

> Now, let's breakdown the process of computing self-attention. Before computing self-attention, each individual input token is first being converted into a vector using the embedding algorithm that we discussed earlier. Then, for each embedding, we calculate its query vector ($\mathbf{Q}$), key vector ($\mathbf{K}$), and value vector ($\mathbf{V}$) by multiplying the embedding vector with three pre-trained matrices $\mathbf{W_Q}$, $\mathbf{W_K}$, $\mathbf{W_V}$ intended for calculating the three matrices respectively. Notice that the three vectors all have the same dimension, and it does not have to be equal with the dimension of the embedding vectors. The dimensions of the three matrices are the same, and they are all length of the embedding vectors by the length of the ($\mathbf{Q}$), ($\mathbf{K}$), and ($\mathbf{V}$) vectors. Then, for each input embedding, we dot multiply its own $\mathbf{Q}$ vector with every other input embedding's $\mathbf{K}$ vector. At this point, for every input embedding, we have calculated a set of score corresponding to each and every embedding in the input sequence, including itself. Then for each of the embedding, we divide all its scores by the square root of the dimension of the key vectors for more stable gradients and pass the scores through the softmax function for normalization. After this, we multiply each $\mathbf{V}$ vector in the input sequence with its respective softmax score, and finally add up those weighted value vectors. This resulting sum vector is the self-attention of this particular input embedding \[VSP+17\] \[Alammar18\]\[IYA16\].
>
> Although we described the process in terms of vectors, in practice it is implemented by means of matrices. This is because the computation process for each vector independent and identical. We would stack our input embeddings as rows in an input matrix, multiply this matrix with learned weight matrices $W_Q$, $W_K$, $W_V$ and get $(Q)$, $(K)$, and $(V)$ vectors respectively, feed the three resulting matrices into the softmax function as:
>
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \times K^T}{\sqrt{d_k}}\right) \times V$$

Ok, let's start with explaining what exactly are Q, K, and V because that wasn't immediately obvious to me when I read Wang's paper.

Of course! Let's dive deeper into the roles of the Query (Q), Key (K), and Value (V) matrices in the self-attention mechanism:

1. **Query (Q)**:
      - **Purpose**: The Query matrix is used to represent the current word or token we are focusing on. Think of it as asking a question: "Which words in the sequence are most relevant to this one?" 
      - **Function**: In the self-attention mechanism, the Q vector of a token is used to score how much other tokens should be attended to. By taking the dot product of Q with every K (explained below), we obtain a score that determines how much focus each token in the sequence should have in relation to the current token.
2. **Key (K)**:
      - **Purpose**: The Key matrix represents all the tokens that we will check our current token against to determine their level of similarity or relevance.
      - **Function**: The K vectors are matched against the Q vector to produce attention scores. The intuition is that if a Q vector has a high dot product with a K vector, then the corresponding tokens are relevant to each other. This score indicates the weight or level of attention the model should give to the token represented by that particular K when considering the token represented by Q.
3. **Value (V)**:
      - **Purpose**: The Value matrix represents the actual content of the tokens. While Q and K are used to determine the relationships and relevance among tokens, V provides the content we want to extract based on those relationships.
      - **Function**: Once we have our attention scores (from the Q-K dot products), these scores determine how much of each V vector we take into the final output. If a token is deemed highly relevant, its V vector contributes more to the final representation of the current token.

**An Analogy**:

Imagine you're in a spy in a room with multiple people having conversations. You're eavesdropping on one person (the "query"), but you also want to gather context from what everyone else (the "keys") are saying to understand as much as you can.

- The **Q** (Query) is you asking: "Who in this room is relevant to what the target (query) is saying?"
- The **K** (Keys) are the topics each person in the room is talking about. By comparing your query to each topic (taking the dot product of Q and K), you determine who is talking about things most relevant to the person you're focusing on.
- The **V** (Values) are the actual words or content of what each person is saying. Once you've identified the most relevant people based on your query and keys, you take a weighted sum of their words (the V vectors) to get the complete context.

The self-attention mechanism uses this approach to weigh the importance of different tokens in a sequence relative to a particular token, resulting in a rich contextual representation for each token in the input.

Ok, now that we better understand Q, K, and V let's look at the total process from beginning to end and go through an example.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \times K^T}{\sqrt{d_k}}\right) \times V$$

1. **Calculating Q, K, and V**:
    - **What happens**: Each embedding vector gets transformed into three different vectors: Query (Q), Key (K), and Value (V).
    - **How it happens**: Multiply the embedding vector with three distinct pre-trained matrices ($W_Q$, $W_K$, and $W_V$).
    - **Why it matters**: These vectors serve distinct purposes. Q is used to fetch the attention score, K provides a set of keys to match against, and V serves as a pool of values to fetch from based on those scores.
    - **Relationship to equation**: This step produces the Q, K, and V vectors which are central elements in the equation.

2. **Dot Product between Q and K**:
    - **What happens**: Each Q vector is dot-multiplied with every K vector.
    - **How it happens**: This is simple vector multiplication between Q of a particular token and the K of every other token, resulting in a score.
    - **Why it matters**: This score represents how much attention should be paid to other tokens when encoding information about a particular token.
    - **Relationship to equation**: This step corresponds to the $Q \times K^T$ part of the equation.

3. **Scaling the Scores**:
    - **What happens**: The scores from the previous step are scaled down.
    - **How it happens**: Each score is divided by the square root of the dimension of the K vectors.
    - **Why it matters**: This step ensures stable gradients. Without this scaling, the gradients could be too small for effective learning, especially when the dimensionality (or depth) of the keys is large.
    - **Relationship to equation**: This step is represented by the division by $\sqrt{d_k}$ in the equation.

4. **Applying Softmax**:
    - **What happens**: The scores for each token are turned into a probability distribution using the softmax function.
    - **How it happens**: Softmax normalizes the scores so they're between 0 and 1 and sum up to 1.
    - **Why it matters**: This provides a clear set of attention "weights" for each token. The higher the softmax output, the more attention the model pays to the corresponding token.
      - I wasn't immediately clear on why softmax. TODO
    - **Relationship to equation**: This step is captured by the $\text{softmax}(\cdot)$ operation in the equation.

5. **Calculating Weighted Values**:
    - **What happens**: The V vectors are multiplied by the softmax scores.
    - **How it happens**: Each token's V vector is weighted by the token's respective softmax score from the previous step.
    - **Why it matters**: This step essentially picks out values from the V vectors proportional to the attention scores. Tokens deemed more relevant contribute more to the final output.
    - **Relationship to equation**: This step corresponds to the $\times V$ at the end of the equation, where the weighted values from the softmax operation are combined with V.

6. **Summing Weighted Values**:
    - **What happens**: The weighted V vectors are summed up.
    - **How it happens**: A simple vector summation.
    - **Why it matters**: The resulting vector is the final output for a particular token, which is a combination of features from other tokens based on the attention scores.
    - **Relationship to equation**: This summation is implied in the matrix multiplication in the equation. The result of the $\text{softmax}(\cdot) \times V$ operation is the summed attention output for each token.

7. **Matrix Computations in Practice**:
    - **What happens**: The operations described above, while explained using vectors, are in practice executed using matrices.
    - **How it happens**: Instead of processing tokens one-by-one, all tokens are processed simultaneously by stacking embeddings into a matrix and using matrix multiplication for efficiency.
    - **Why it matters**: Matrix operations are highly optimized and parallelizable, making the computations significantly faster, especially on hardware like GPUs.
    - **Relationship to equation**: The use of Q, K, and V in the equation reflects these matrix computations.

8. **The Attention Output**:
    - The equation one last time:
    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \times K^T}{\sqrt{d_k}}\right) \times V
    $$
    - This formula captures the essence of the self-attention mechanism. The product of Q and $K^T$ gives attention scores, which after scaling and softmax, are used to weigh the V values.

##### A Concrete Example

Given the above steps for calculating self-attention, let's break it down:
1. **Create Query (Q), Key (K), and Value (V) Matrices**: We obtain these by multiplying the input embeddings (with position encodings) with the weight matrices $W_Q$, $W_K$, and $W_V$ respectively. 

For the sake of this example, we will assume these weight matrices are:

$$
W_Q = \begin{pmatrix} 1 & 0 & 0 & 1 \\\\ 0 & 1 & 1 & 0 \\\\ 0 & 1 & 0 & 1 \\\\ 1 & 0 & 1 & 0 \end{pmatrix}
$$

$$
W_K = \begin{pmatrix} 0 & 1 & 1 & 0 \\\\ 1 & 0 & 0 & 1 \\\\ 1 & 0 & 1 & 0 \\\\ 0 & 1 & 0 & 1 \end{pmatrix}
$$

$$
W_V = \begin{pmatrix} 1 & 1 & 0 & 0 \\\\ 0 & 0 & 1 & 1 \\\\ 0 & 1 & 1 & 0 \\\\ 1 & 0 & 0 & 1 \end{pmatrix}
$$

2. **Compute Q, K, and V**: Multiply the input matrix $X$ with these weights:
  $$
  Q = X \times W_Q
  $$
  $$
  K = X \times W_K
  $$
  $$
  V = X \times W_V
  $$
3. **Calculate Attention Scores**: This is done by multiplying $Q$ with $K^T$, then dividing by the square root of the dimension of the key vectors $d_k$. In our example, $d_k$ is 4.
  $$
  Score = \frac{Q \times K^T}{\sqrt{d_k}}
  $$
4. **Apply Softmax to the Scores**: This will give each word's attention score.
  $$
  SoftmaxScore = \text{softmax}(Score)
  $$
5. **Multiply Softmax Score with V**: This will give the weighted representation of the input with respect to other words in the sentence.
  $$
  AttentionOutput = SoftmaxScore \times V
  $$

Let's calculate the matrices Q, K, V, Score, SoftmaxScore, and AttentionOutput:

To begin, we will compute the $Q$, $K$, and $V$ matrices.

Using:
$$ X + PE_{\text{input}} = \begin{pmatrix} 0.1 & 1.2 & -0.1 & 1.4 \\\ -0.3 + \sin\left(\frac{1}{10000^0}\right) & 0.5 + \cos\left(\frac{1}{10000^{0.5}}\right) & 0.1 + \sin\left(\frac{1}{10000^2}\right) & -0.2 + \cos\left(\frac{1}{10000^{2.5}}\right) \\\ 0.4 + \sin\left(\frac{2}{10000^0}\right) & -0.3 + \cos\left(\frac{2}{10000^{0.5}}\right) & 0.2 + \sin\left(\frac{2}{10000^2}\right) & 0.1 + \cos\left(\frac{2}{10000^{2.5}}\right) \end{pmatrix} $$

Given the weight matrices:

$$ W_Q = \begin{pmatrix} 1 & 0 & 0 & 1 \\\ 0 & 1 & 1 & 0 \\\ 0 & 1 & 0 & 1 \\\ 1 & 0 & 1 & 0 \end{pmatrix} $$
$$ W_K = \begin{pmatrix} 0 & 1 & 1 & 0 \\\ 1 & 0 & 0 & 1 \\\ 1 & 0 & 1 & 0 \\\ 0 & 1 & 0 & 1 \end{pmatrix} $$
$$ W_V = \begin{pmatrix} 1 & 1 & 0 & 0 \\\ 0 & 0 & 1 & 1 \\\ 0 & 1 & 1 & 0 \\\ 1 & 0 & 0 & 1 \end{pmatrix} $$

1. Compute $Q$:
  $$
  Q = (X + PE_{\text{input}}) \times W_Q
  $$

2. Compute $K$:
  $$
  K = (X + PE_{\text{input}}) \times W_K
  $$

3. Compute $V$:
  $$
  V = (X + PE_{\text{input}}) \times W_V
  $$

Let's calculate these values:

First, let's compute the `Q` matrix using our input matrix \( X + PE_{\text{input}} \) and the \( W_Q \) matrix:

$$
Q = (X + PE_{\text{input}}) \times W_Q
$$

Given:

$$ X + PE_{\text{input}} = \begin{pmatrix} 0.1 & 1.2 & -0.1 & 1.4 \\\ -0.3 + \sin\left(\frac{1}{10000^0}\right) & 0.5 + \cos\left(\frac{1}{10000^{0.5}}\right) & 0.1 + \sin\left(\frac{1}{10000^2}\right) & -0.2 + \cos\left(\frac{1}{10000^{2.5}}\right) \\\ 0.4 + \sin\left(\frac{2}{10000^0}\right) & -0.3 + \cos\left(\frac{2}{10000^{0.5}}\right) & 0.2 + \sin\left(\frac{2}{10000^2}\right) & 0.1 + \cos\left(\frac{2}{10000^{2.5}}\right) \end{pmatrix} $$

$$ W_Q = \begin{pmatrix} 1 & 0 & 0 & 1 \\\ 0 & 1 & 1 & 0 \\\ 0 & 1 & 0 & 1 \\\ 1 & 0 & 1 & 0 \end{pmatrix} $$

When you multiply the above matrices, you get:

$$ Q = \begin{pmatrix} 1.5 & 1.1 & 2.6 & 0 \\\ 1.3415 & 1.6 & 2.3 & 0.6415 \\\ 2.4093 & 0.8998 & 1.7998 & 1.5093 \end{pmatrix} $$

If your matrix multiplication is rusty see [the matrix math behind this calculation](#matrix-multiplication).

We compute the $K$ matrix using our input matrix $X + PE_{\text{input}}$ and the $W_K$ matrix:

$$
K = (X + PE_{\text{input}}) \times W_K
$$

Given:

$$ X + PE_{\text{input}} = \begin{pmatrix} 0.1 & 1.2 & -0.1 & 1.4 \\\ -0.3 + \sin\left(\frac{1}{10000^0}\right) & 0.5 + \cos\left(\frac{1}{10000^{0.5}}\right) & 0.1 + \sin\left(\frac{1}{10000^2}\right) & -0.2 + \cos\left(\frac{1}{10000^{2.5}}\right) \\\ 0.4 + \sin\left(\frac{2}{10000^0}\right) & -0.3 + \cos\left(\frac{2}{10000^{0.5}}\right) & 0.2 + \sin\left(\frac{2}{10000^2}\right) & 0.1 + \cos\left(\frac{2}{10000^{2.5}}\right) \end{pmatrix} $$

$$ W_Q = \begin{pmatrix} 1 & 0 & 0 & 1 \\\ 0 & 1 & 1 & 0 \\\ 0 & 1 & 0 & 1 \\\ 1 & 0 & 1 & 0 \end{pmatrix} $$

When you multiply the above matrices, you get:

$$ Q = \begin{pmatrix} 1.5 & 1.1 & 2.6 & 0 \\\ 1.3415 & 1.6 & 2.3 & 0.6415 \\\ 2.4093 & 0.8998 & 1.7998 & 1.5093 \end{pmatrix} $$

Finally we do the math for $V$. $X + PE_{\text{input}}$ and the $W_V$ matrix:

$$
V = (X + PE_{\text{input}}) \times W_V
$$

Given:

$$ X + PE_{\text{input}} = \begin{pmatrix} 0.1 & 1.2 & -0.1 & 1.4 \\\ -0.3 + \sin\left(\frac{1}{10000^0}\right) & 0.5 + \cos\left(\frac{1}{10000^{0.5}}\right) & 0.1 + \sin\left(\frac{1}{10000^2}\right) & -0.2 + \cos\left(\frac{1}{10000^{2.5}}\right) \\\ 0.4 + \sin\left(\frac{2}{10000^0}\right) & -0.3 + \cos\left(\frac{2}{10000^{0.5}}\right) & 0.2 + \sin\left(\frac{2}{10000^2}\right) & 0.1 + \cos\left(\frac{2}{10000^{2.5}}\right) \end{pmatrix} $$

$$ W_V = \begin{pmatrix} 0 & 0 & 1 & 1 \\\ 1 & 1 & 0 & 0 \\\ 0 & 1 & 0 & 1 \\\ 1 & 0 & 1 & 0 \end{pmatrix} $$

Multiplying the matrices, we get:

$$ V = \begin{pmatrix} 1.5 & 0 & 1.1 & 2.6 \\\ 1.3415 & 0.6416 & 1.6001 & 2.3 \\\ 2.4093 & 1.5095 & 0.9 & 1.7998 \end{pmatrix} $$

We perform the [same calculations](#calculating-y) for $Y$. Here are the results:

$$ \text{Q} = \begin{pmatrix} 1 & 1 & 2 & 0 \\\ 1.7415 & 1.7001 & 2.5 & 0.9416 \\\ 2.7093 & 0.5 & 2.1998 & 1.0095 \end{pmatrix} $$
$$ \text{K} = \begin{pmatrix} 1 & 1 & 0 & 2 \\\ 1.7001 & 1.7415 & 0.9416 & 2.5 \\\ 0.5 & 2.7093 & 1.0095 & 2.1998 \end{pmatrix} $$
$$ \text{V} = \begin{pmatrix} 1 & 0 & 1 & 2 \\\ 1.7415 & 0.9416 & 1.7001 & 2.5 \\\ 2.7093 & 1.0095 & 0.5 & 2.1998 \end{pmatrix} $$

This gives us the $V$ matrix. With $Q$, $K$, and $V$ matrices in hand, you're ready to compute the attention scores and proceed with the self-attention mechanism.

Next we need to calculate the attention score with:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \times K^T}{\sqrt{d_k}}\right) \times V$$

I did this with [this python code](#calculate-attention-score):

$$ \text{Attention}(Q, K, V) = \begin{pmatrix} 2.11372594 & 1.21488963 & 1.06582258 & 1.96465889 \\\ 2.10729705 & 1.1957622 & 1.06288922 & 1.97442407 \\\ 1.82115196 & 0.90157546 & 1.21880742 & 2.13838393 \end{pmatrix} $$

#### Attention Heads and Multi Attention Heads

I will not do all the math here as it is an exact repeat of what we have above, but the actual models will use multiple attention heads. Each attention head "pays attention" to different parts of the sentence. The goal being to avoid a few tokens (words) having an outsized impact on the output. We want to pay attention to the totality of the sentence.

Consider a sentence like "Jane, who recently graduated from Harvard, is starting her new job at Google." In this sentence, the words "Harvard" and "Google" are likely to have high attention scores because they are proper nouns and often important in text. 

- **Single-Head Attention**: If you're using single-head attention to find out where Jane is starting her job, the model might give high attention to both "Harvard" and "Google". This could be misleading because the word "Harvard" isn't relevant to the query about Jane's new job, even though it's generally an important token.
- **Multi-Head Attention**: In contrast, one head could focus on "Jane" and "job," while another head could focus on "Harvard" and "Google," and yet another head could focus on "recently graduated" and "starting." This way, the model can capture both the important context provided by "Harvard" and the fact that "Google" is where she is starting her new job, without letting the importance of "Harvard" overshadow the relevance of "Google" to the query.

Returning to [Troy Wang's paper](https://www.cis.upenn.edu/wp-content/uploads/2021/10/Tianzheng_Troy_Wang_CIS498EAS499_Submission.pdf#page=12):

> One problem of the self-attention layer is that by only using a single set of trained matrices \( Q, K, \) and \( V \), the self-attention could be dominated by just one or a few tokens, and thereby not being able to pay attention to multiple places that might be meaningful. Therefore, by using multi-heads, we aim to linearly combine the results of many independent self-attention computations, and thereby expand the self-attention layer's ability to focus on different positions \[VSP+17\] \[Alammar18\].
>
> More concretely, we use multiple sets of mutually independent \( (Q), (K), \) and \( (V) \) matrices, each being randomly initialized and independently trained. With multiple \( (Q), (K), \) and \( (V) \) matrices, we end up with multiple resulting vectors for every input token vector. Nonetheless, the feedforward neural network in the next step is designed to only accept one vector per word input. In order to combine those vectors, we concatenate them into a single vector and then multiply it with another weight vector which is trained simultaneously.
>
> Formally, this multi-head attention is defined as 
>
> $$
> MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
> $$
> where $head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)$

What this would actually look like. Here we just make up some matrix and assume that it is the output from head 2; it would have been generated exactly as we did the output from the first head.

1. **Compute for Head 2**:  
   First, let's assume the output of the second attention head (`head_2`) is:
$$
\text{Attention}_2(Q, K, V) = 
\begin{pmatrix}
1.5 & 0.8 & 1.2 & 2.0 \\\\
1.6 & 0.9 & 1.1 & 2.1 \\\\
1.4 & 0.7 & 1.3 & 1.9
\end{pmatrix}
$$

2. **Concatenate Outputs**:  
   Now, we concatenate the outputs of `head_1` and `head_2`:
$$
\text{Concat}(\text{Attention}_1, \text{Attention}_2) = 
\begin{pmatrix}
2.11372594 & 1.21488963 & 1.06582258 & 1.96465889 & 1.5 & 0.8 & 1.2 & 2.0 \\\\
2.10729705 & 1.1957622 & 1.06288922 & 1.97442407 & 1.6 & 0.9 & 1.1 & 2.1 \\\\
1.82115196 & 0.90157546 & 1.21880742 & 2.13838393 & 1.4 & 0.7 & 1.3 & 1.9
\end{pmatrix}
$$

3. **Final Linear Projection**:  
   Finally, we multiply the concatenated matrix with a learned projection matrix $W_O$. For the sake of simplicity in this example, let's assume $W_O$ is an 8x4 matrix filled with 0.5. In a real-world scenario, this matrix would have learned values.

I generated random values for $W_O$ but in reality this would start random and the model would train these values over time.

$$
W_O = 
\begin{pmatrix}
0.37738326 & 0.83274845 & 0.37280978 & 0.14584743 \\\\
0.28706851 & 0.29072609 & 0.69116998 & 0.20106682 \\\\
0.26764653 & 0.12058646 & 0.82634382 & 0.60818759 \\\\
0.44329703 & 0.4425581  & 0.89811744 & 0.24551412 \\\\
0.9186323  & 0.40029736 & 0.17636762 & 0.06896409 \\\\
0.41921272 & 0.0495383  & 0.77792527 & 0.4354529  \\\\
0.14791365 & 0.66822966 & 0.48313699 & 0.94127396 \\\\
0.11604641 & 0.51794357 & 0.62942357 & 0.76420883
\end{pmatrix}
$$

Multiply the two together:

The final multi-head attention output will be:
$$
\text{Concat}(\text{Attention}_1, \text{Attention}_2)\times W_{O}=
\text{MultiHead}(Q,K,V)=
\begin{pmatrix}
4.42554033 & 5.589241 & 6.99844643 & 4.79288192 \\\\
4.55176485 & 5.6122494 & 7.09923364 & 4.82144704 \\\\
4.21254506 & 5.31988824 & 6.84520426 & 4.79017392
\end{pmatrix}
$$

I don't want to get too into the weeds on this, but it is worth making a brief note on why this is better than RNN. The short version is it's faster. For starters, all the calculations for each attention head can run in parallel completely independently.

The other great part is that the calculation is independent of input length. You could feed in 1000 words or 500 and the attention calculation runs at the same speed.


#### Layer Normalization

As the model trains itself, problems frequently arise. For example:

Without layer normalization, the model could suffer from issues related to the internal covariate shift. Here's a simple example:

Let's say we have a neural network for binary classification and a layer that takes two features $x_1$ and $x_2$ as input. Initially, $x_1$ and $x_2$ are both in the range of [0, 1].

1. **First Epoch**: The model learns some weights and biases based on $x_1$ and $x_2$.
2. **Second Epoch**: We add new features, or the features themselves change distribution, such that $x_1$ is now in the range of [0, 1000] while $x_2$ remains in [0, 1].

Without layer normalization, this change in feature scale will make the previously learned weights and biases less useful, and the model may need a lot of time to adjust to this new scale. It might even diverge and fail to train.

In contrast, if we use layer normalization, the inputs are rescaled to have zero mean and unit variance, making it easier for the model to adapt to the new data distribution. Layer normalization keeps the scales consistent, making training more stable.

After layer normalization is applied the results won't be restricted to a specific range like [0, 1] or [-1, 1] as in the case of some other normalization techniques. Instead, layer normalization will center the data around zero and will scale based on the standard deviation, but there's no hard constraint on the range of the output values.

However, after layer normalization, the mean of each row (or each example, in the context of a neural network) will be approximately 0, and the standard deviation will be approximately 1. The actual values can be positive or negative and can exceed the range of [-1, 1], depending on the original data and its distribution.

Layer normalization is applied to each data point within a given example, rather than across examples in the dataset (which is what batch normalization does). 

Given matrix $M$:

$$
M = 
\begin{pmatrix}
4.42554033 & 5.589241   & 6.99844643 & 4.79288192 \\\\
4.55176485 & 5.6122494  & 7.09923364 & 4.82144704 \\\\
4.21254506 & 5.31988824 & 6.84520426 & 4.79017392
\end{pmatrix}
$$

The layer normalization for each row (example) in $M$ is calculated as:

$$
\text{LN}(x_i) = \gamma \times \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

- $x_i$ is the input vector (a row in $M$ in our case).
- $\mu$ is the mean of $x_i$.
- $\sigma^2$ is the variance of $x_i$.
- $\gamma$ and $\beta$ are learnable parameters, which can be set to 1 and 0 respectively for simplification.
- $\epsilon$ is a small constant added for numerical stability (can be set to $1e-6$ or similar).

I don't show the breakdown of applying the formula here but what really matters is understanding what the formula does. See [this python code](#calculate-layer-normalization) for how I got the results.

$$
\text{LN}(x_i)=
\begin{pmatrix}
-1.03927142 & 0.13949668 & 1.56694829 & -0.66717355 \\\\
-0.97825977 & 0.09190721 & 1.59246789 & -0.70611533 \\\\
-1.10306273 & 0.02854757 & 1.58729045 & -0.51277529
\end{pmatrix}
$$

#### Feed Forward Neural Net

Finally we reach the feed forward network - the piece at the heart of most AI models. So much of AI is based on these things I think it's worth spending a bit of time explaining how they work.

The book [Deep Learning (2017, MIT), by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/contents/mlp.html) does a really good job of explaining this along with some common terms from deep learning so I will just quote it here.

> **Deep feedforward networks**, also often called **feedforward neural networks**, or **multilayer perceptrons (MLPs)**, are the quintessential deep learning models. The goal of a feedforward network is to approximate some function $f^*$. For example, for a classifier, $y = f^*(x)$ maps an input $x$ to a category $y$. A feedforward network defines a mapping $y = f(x; \theta)$ and learns the value of the parameters $\theta$ that result in the best function approximation.
>
> These models are called **feedforward** because information flows through the function being evaluated from $x$, through the intermediate computations used to define $f$, and finally to the output $y$. There are no feedback connections in which outputs of the model are fed back into itself. When feedforward neural networks are extended to include feedback connections, they are called **recurrent neural networks**, presented in chapter 10.
>
> **Feedforward networks** are of extreme importance to machine learning practitioners. They form the basis of many important commercial applications. For example, the convolutional networks used for object recognition from photos are a specialized kind of feedforward network. Feedforward networks are a conceptual stepping stone on the path to recurrent networks, which power many natural language applications.
>
> **Feedforward neural networks** are called **networks** because they are typically represented by composing together many different functions. The model is associated with a directed acyclic graph describing how the functions are composed together. For example, we might have three functions $f^{(1)}, f^{(2)},$ and $f^{(3)}$ connected in a chain, to form $f(x) = f^{(3)}(f^{(2)}(f^{(1)}(x)))$. These chain structures are the most commonly used structures of neural networks. In this case, $f^{(1)}$ is called the first layer of the network, $f^{(2)}$ is called the second layer, and so on. The overall length of the chain gives the depth of the model. It is from this terminology that the name "deep learning" arises. The final layer of a feedforward network is called the **output layer**. During neural network training, we drive $f(x)$ to match $f^*(x)$. The training data provides us with noisy, approximate examples of $f^*(x)$ evaluated at different training points. Each example $x$ is accompanied by a label $y \approx f^*(x)$. The training examples specify directly what the output layer must do at each point $x$; it must produce a value that is close to $y$. The behavior of the other layers is not directly specified by the training data. The learning algorithm must decide how to use those layers to produce the desired output, but the training data does not say what each individual layer should do. Instead, the learning algorithm must decide how to use these layers to best implement an approximation of $f^*$. Because the training data does not show the desired output for each of these layers, these layers are called **hidden layers**.
>
> Finally, these networks are called *neural* because they are loosely inspired by neuroscience. Each hidden layer of the network is typically vector-valued. The dimensionality of these hidden layers determines the **width** of the model. Each element of the vector may be interpreted as playing a role analogous to a neuron. Rather than thinking of the layer as representing a single vector-to-vector function, we can also think of the layer as consisting of many **units** that act in parallel, each representing a vector-to-scalar function. Each unit resembles a neuron in the sense that it receives input from many other units and computes its own activation value. The idea of using many layers of vector-valued representation is drawn from neuroscience. The choice of the functions $f^{(i)}(x)$ used to compute these representations is also loosely guided by neuroscientific observations about the functions that biological neurons compute. However, modern neural network research is guided by many mathematical and engineering disciplines, and the goal of neural networks is not to perfectly model the brain. It is best to think of feedforward networks as function approximation machines that are designed to achieve statistical generalization, occasionally drawing some insights from what we know about the brain, rather than as models of brain function.

You might be asking yourself what the difference between all this and a linear regression. The [Deep Learning book](https://www.deeplearningbook.org/contents/mlp.html) gets into this on page 165. I leave out this discussion and instead provide a simple picture below which displays some true function that we are trying to approximate and allows you to compare the two graphs:

![](images/2023-11-01-08-31-24.png)

I used [this code](#linear-regression-code) to generate the plot using tensor flow. What's cool about that plot is that it is a real machine learning problem and that is the real performance of the two models.

As the Deep Learning book points out, there isn't even a way to approximate something as simple as the XOR function with linear regression. Here is a [plot of what XOR](#plot-xor) looks like:

![](images/2023-11-01-09-01-39.png)





### Prepare and Embed Target Sequence for Decoder
### Decoder Stack Processing with Encoder Output
### Output Layer for Word Probabilities
### Loss Function and Back-Propagation

### Supplemental Information

#### Word Embeddings

The first question(s) I asked when I saw these matrices is, "Where do these values come from and what are they?" What followed soon after was, "Why do they exist?" Why they exist lends context to where do they come from and what are they.

*Why do they exist*

The article [Large language models, explained with a minimum of math and jargon](https://www.understandingai.org/p/large-language-models-explained-with) does a fantastic job of answering this in a non-mathematical way. The goal here is that we want to describe mathematically the relationship between some given word and other words because this will then allow us to more accurately predict what words should go in the answers our LLM produces. Words that are similar should have similar vector values. This concept was pioneered at Google in [a paper called Word2vec](https://en.wikipedia.org/wiki/Word2vec). You can see this visually at [this website](http://vectors.nlpl.eu/explore/embeddings/en/MOD_enwiki_upos_skipgram_300_2_2021/airplane_NOUN/). For example, I wanted to see what airplanes are likely to be related to airplanes:

![](images/2023-10-30-16-40-48.png)

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

#### Weight Matrices

The weight matrices $W_Q$, $W_K$, and $W_V$ are learned parameters in the self-attention mechanism of the transformer model. Here's a more detailed explanation:
1. **Initialization**: At the beginning of the training process, these matrices are usually initialized with small random values. This can be achieved through various initialization techniques, such as Xavier or He initialization.
2. **Training**: During the training process, the transformer model is fed with input data, and it tries to minimize the difference between its predictions and the actual output (this difference is often measured using a loss function like cross-entropy for classification tasks). As the model is trained using optimization algorithms like stochastic gradient descent (SGD) or its variants (e.g., Adam, RMSprop), these weight matrices get updated iteratively. The updates are done in a direction that reduces the loss.
3. **Role of Backpropagation**: The updating of these matrices is governed by backpropagation, a fundamental technique in training deep neural networks. Gradients are computed for the loss with respect to each element of these matrices, which indicates how much a small change in that element would affect the overall loss. These gradients are then used to adjust the matrices in the direction that minimizes the loss.
4. **Final Model**: After many iterations (epochs) of training, these matrices converge to values that allow the transformer model to effectively compute self-attention over the input data. In other words, through training, the model learns the best values for these matrices to perform the task at hand, whether it's language translation, text classification, or any other NLP task.

#### Softmax

Let's say we have a neural network that is trying to classify an input into one of three categories. The final layer of the network produces a vector of logits (raw prediction scores) for each category:

$$
x = [2.0, 1.0, 0.2]
$$

Here:
- The first value (2.0) corresponds to the prediction score for Category A.
- The second value (1.0) is for Category B.
- The third value (0.2) is for Category C.

To interpret these scores as probabilities, we use the softmax function.

Applying softmax:

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{3} e^{x_j}}
$$

For each component:

$$
\text{softmax}(x)_A = \frac{e^{2.0}}{e^{2.0} + e^{1.0} + e^{0.2}}
$$
$$
\text{softmax}(x)_B = \frac{e^{1.0}}{e^{2.0} + e^{1.0} + e^{0.2}}
$$
$$
\text{softmax}(x)_C = \frac{e^{0.2}}{e^{2.0} + e^{1.0} + e^{0.2}}
$$

Computing these, we might get something like:

$$
\text{softmax}(x) = [0.65, 0.24, 0.11]
$$

The results are probabilities:
- Category A: 65%
- Category B: 24%
- Category C: 11%

The values sum up to 1, and the largest logit (2.0 for Category A) corresponds to the largest probability (65%).

**Why do we use softmax for self attention?**

1. **Normalization**: The raw logits can be any set of values, positive or negative. Softmax ensures that we get a proper probability distribution where values are between 0 and 1 and sum up to 1.
2. **Emphasizing Differences**: Even subtle differences in logits can be emphasized. In our example, the difference between 2.0 and 1.0 becomes a difference between 65% and 24% in probabilities.
3. **Self-attention mechanism**: In the context of Transformers and self-attention, softmax is applied to the scores (results of Q and K dot products) to determine the weight or attention each input should get. By turning these scores into probabilities, the model decides which parts of the input sequence are most relevant for each position, effectively weighting the contribution of each input's value (from the V matrix) when producing the output.

#### Matrix Multiplication

Given:

$$
X + PE_{\text{input}} = 
\begin{pmatrix}
0.1 & 1.2 & -0.1 & 1.4 \\\\
-0.3 + \sin\left(\frac{1}{10000^0}\right) & 0.5 + \cos\left(\frac{1}{10000^{0.5}}\right) & 0.1 + \sin\left(\frac{1}{10000^2}\right) & -0.2 + \cos\left(\frac{1}{10000^{2.5}}\right) \\\\
0.4 + \sin\left(\frac{2}{10000^0}\right) & -0.3 + \cos\left(\frac{2}{10000^{0.5}}\right) & 0.2 + \sin\left(\frac{2}{10000^2}\right) & 0.1 + \cos\left(\frac{2}{10000^{2.5}}\right)
\end{pmatrix}
$$

$$
W_Q = 
\begin{pmatrix} 
1 & 0 & 0 & 1 \\\\ 
0 & 1 & 1 & 0 \\\\ 
0 & 1 & 0 & 1 \\\\ 
1 & 0 & 1 & 0 
\end{pmatrix} 
$$

1. First row of $Q$: Multiply each element of the first row of $(X + PE_{\text{input}})$ with the corresponding element of each column in $W_Q$, then sum them up.
  $$ Q[1,1] = (0.1 \times 1) + (1.2 \times 0) + (-0.1 \times 0) + (1.4 \times 1) = 0.1 + 0 + 0 + 1.4 = 1.5 $$
  $$ Q[1,2] = (0.1 \times 0) + (1.2 \times 1) + (-0.1 \times 1) + (1.4 \times 0) = 0 + 1.2 - 0.1 + 0 = 1.1 $$
  $$ Q[1,3] = (0.1 \times 0) + (1.2 \times 1) + (-0.1 \times 0) + (1.4 \times 1) = 0 + 1.2 + 0 + 0 = 1.2 $$
  $$ Q[1,4] = (0.1 \times 1) + (1.2 \times 0) + (-0.1 \times 1) + (1.4 \times 0) = 0.1 + 0 - 0.1 + 0 = 0 $$
2. Second row of $Q$
  $$ Q[2,1] = (-0.3 + \sin\left(\frac{1}{10000^0}\right)) \times 1 + (0.5 + \cos\left(\frac{1}{10000^{0.5}}\right)) \times 0 + (0.1 + \sin\left(\frac{1}{10000^2}\right)) \times 0 + (-0.2 + \cos\left(\frac{1}{10000^{2.5}}\right)) \times 1 $$
  $$ Q[2,2] = (-0.3 + \sin\left(\frac{1}{10000^0}\right)) \times 0 + (0.5 + \cos\left(\frac{1}{10000^{0.5}}\right)) \times 1 + (0.1 + \sin\left(\frac{1}{10000^2}\right)) \times 1 + (-0.2 + \cos\left(\frac{1}{10000^{2.5}}\right)) \times 0 $$
  $$ Q[2,3] = (-0.3 + \sin\left(\frac{1}{10000^0}\right)) \times 0 + (0.5 + \cos\left(\frac{1}{10000^{0.5}}\right)) \times 1 + (0.1 + \sin\left(\frac{1}{10000^2}\right)) \times 0 + (-0.2 + \cos\left(\frac{1}{10000^{2.5}}\right)) \times 1 $$
  $$ Q[2,4] = (-0.3 + \sin\left(\frac{1}{10000^0}\right)) \times 1 + (0.5 + \cos\left(\frac{1}{10000^{0.5}}\right)) \times 0 + (0.1 + \sin\left(\frac{1}{10000^2}\right)) \times 1 + (-0.2 + \cos\left(\frac{1}{10000^{2.5}}\right)) \times 0 $$
3. Third row of $Q$
  $$ Q[3,1] = (0.4 + \sin\left(\frac{2}{10000^0}\right)) \times 1 + (-0.3 + \cos\left(\frac{2}{10000^{0.5}}\right)) \times 0 + (0.2 + \sin\left(\frac{2}{10000^2}\right)) \times 0 + (0.1 + \cos\left(\frac{2}{10000^{2.5}}\right)) \times 1 $$
  $$ Q[3,2] = (0.4 + \sin\left(\frac{2}{10000^0}\right)) \times 0 + (-0.3 + \cos\left(\frac{2}{10000^{0.5}}\right)) \times 1 + (0.2 + \sin\left(\frac{2}{10000^2}\right)) \times 1 + (0.1 + \cos\left(\frac{2}{10000^{2.5}}\right)) \times 0 $$
  $$ Q[3,3] = (0.4 + \sin\left(\frac{2}{10000^0}\right)) \times 0 + (-0.3 + \cos\left(\frac{2}{10000^{0.5}}\right)) \times 1 + (0.2 + \sin\left(\frac{2}{10000^2}\right)) \times 0 + (0.1 + \cos\left(\frac{2}{10000^{2.5}}\right)) \times 1 $$
  $$ Q[3,4] = (0.4 + \sin\left(\frac{2}{10000^0}\right)) \times 1 + (-0.3 + \cos\left(\frac{2}{10000^{0.5}}\right)) \times 0 + (0.2 + \sin\left(\frac{2}{10000^2}\right)) \times 1 + (0.1 + \cos\left(\frac{2}{10000^{2.5}}\right)) \times 0 $$

Plug that into some Python:

```python
import numpy as np

X_PE_input = np.array([
    [0.1, 1.2, -0.1, 1.4],
    [-0.3 + np.sin(1), 0.5 + np.cos(1/10000**0.5), 0.1 + np.sin(1/10000**2), -0.2 + np.cos(1/10000**2.5)],
    [0.4 + np.sin(2), -0.3 + np.cos(2/10000**0.5), 0.2 + np.sin(2/10000**2), 0.1 + np.cos(2/10000**2.5)]
])

W_Q = np.array([
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

Q = X_PE_input.dot(W_Q)

print(Q)
```

that will get you:

```
[[1.5        1.1        2.6        0.        ]
 [1.34147098 1.59995001 2.29995    0.64147099]
 [2.40929743 0.89980003 1.79980001 1.50929745]]
```

Printed nicely out to four decimal places:

$$
\begin{pmatrix}
1.5 & 1.1 & 2.6 & 0 \\\\
1.3415 & 1.6 & 2.3 & 0.6415 \\\\
2.4093 & 0.8998 & 1.7998 & 1.5093
\end{pmatrix}
$$

You can get all three matrices with:

```python
import numpy as np

# Input matrix
X = np.array([
    [0.1, 1.2, -0.1, 1.4],
    [0.5415, 1.49995, 0.1001, 0.8],
    [1.3093, 0.6998, 0.2002, 1.1]
])

# Weight matrices
W_Q = np.array([
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

W_K = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
])

W_V = np.array([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1]
])

# Matrix multiplication
Q = np.dot(X, W_Q)
K = np.dot(X, W_K)
V = np.dot(X, W_V)

print("Q:")
print(Q)
print("\nK:")
print(K)
print("\nV:")
print(V)
```

Output

```
Q:
[[1.5     1.1     2.6     0.     ]
 [1.3415  1.60005 2.29995 0.6416 ]
 [2.4093  0.9     1.7998  1.5095 ]]

K:
[[1.1     1.5     0.      2.6    ]
 [1.60005 1.3415  0.6416  2.29995]
 [0.9     2.4093  1.5095  1.7998 ]]

V:
[[1.5     0.      1.1     2.6    ]
 [1.3415  0.6416  1.60005 2.29995]
 [2.4093  1.5095  0.9     1.7998 ]]
>
```

##### Calculating Y

Calculate for $Y$:

```python

import numpy as np

# Input matrix X
X = np.array([
    [0.0, 1.0, 0.0, 1.0],
    [0.6415, 1.39995, 0.3001, 1.1],
    [1.4093, 0.8998, -0.3998, 1.3]
])

# Weight matrices W_Q, W_K, W_V
W_Q = np.array([
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

W_K = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
])

W_V = np.array([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1]
])

# Calculate Q, K, V matrices
Q = np.dot(X, W_Q)
K = np.dot(X, W_K)
V = np.dot(X, W_V)

print("Q matrix:")
print(Q)

print("K matrix:")
print(K)

print("V matrix:")
print(V)

```

Output:

```
Q matrix:
[[1.      1.      2.      0.     ]
 [1.7415  1.70005 2.49995 0.9416 ]
 [2.7093  0.5     2.1998  1.0095 ]]
K matrix:
[[1.      1.      0.      2.     ]
 [1.70005 1.7415  0.9416  2.49995]
 [0.5     2.7093  1.0095  2.1998 ]]
V matrix:
[[1.      0.      1.      2.     ]
 [1.7415  0.9416  1.70005 2.49995]
 [2.7093  1.0095  0.5     2.1998 ]]
```

#### Calculate Attention Score

```python
import numpy as np
from scipy.special import softmax


def attention(Q, K, V):
    d_k = Q.shape[-1]
    matmul_qk = np.dot(Q, K.T)

    # Scale the matrix multiplication
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    # Apply softmax
    attention_weights = softmax(scaled_attention_logits, axis=-1)

    # Multiply by the Value matrix
    output = np.dot(attention_weights, V)

    return output


# Given matrices
Q = np.array([[1.5, 1.1, 2.6, 0.],
              [1.3415, 1.60005, 2.29995, 0.6416],
              [2.4093, 0.9, 1.7998, 1.5095]])

K = np.array([[1.1, 1.5, 0., 2.6],
              [1.60005, 1.3415, 0.6416, 2.29995],
              [0.9, 2.4093, 1.5095, 1.7998]])

V = np.array([[1.5, 0., 1.1, 2.6],
              [1.3415, 0.6416, 1.60005, 2.29995],
              [2.4093, 1.5095, 0.9, 1.7998]])

result = attention(Q, K, V)
print("Attention(Q, K, V) = ")
print(result)

```

Output:

```
Attention(Q, K, V) = 
[[2.11372594 1.21488963 1.06582258 1.96465889]
 [2.10729705 1.1957622  1.06288922 1.97442407]
 [1.82115196 0.90157546 1.21880742 2.13838393]]
```

#### Calculate Multi-Attention Head Output

```python
import numpy as np

# Given attention outputs
Attention1 = np.array([
    [2.11372594, 1.21488963, 1.06582258, 1.96465889],
    [2.10729705, 1.1957622, 1.06288922, 1.97442407],
    [1.82115196, 0.90157546, 1.21880742, 2.13838393]
])

Attention2 = np.array([
    [1.5, 0.8, 1.2, 2.0],
    [1.6, 0.9, 1.1, 2.1],
    [1.4, 0.7, 1.3, 1.9]
])

# Concatenate the attention outputs
concat_attention = np.concatenate([Attention1, Attention2], axis=1)

# Given W_O matrix
W_O = np.array([
    [0.37738326, 0.83274845, 0.37280978, 0.14584743],
    [0.28706851, 0.29072609, 0.69116998, 0.20106682],
    [0.26764653, 0.12058646, 0.82634382, 0.60818759],
    [0.44329703, 0.4425581,  0.89811744, 0.24551412],
    [0.9186323,  0.40029736, 0.17636762, 0.06896409],
    [0.41921272, 0.0495383,  0.77792527, 0.4354529 ],
    [0.14791365, 0.66822966, 0.48313699, 0.94127396],
    [0.11604641, 0.51794357, 0.62942357, 0.76420883]
])

# Multiply concatenated output with W_O
multihead_output = concat_attention.dot(W_O)

print(multihead_output)

```

Output:

```
[[4.42554033 5.589241   6.99844643 4.79288192]
 [4.55176485 5.6122494  7.09923364 4.82144704]
 [4.21254506 5.31988824 6.84520426 4.79017392]]
```

#### Calculate Layer Normalization

```python
import numpy as np

# Define matrix M
M = np.array([
    [4.42554033, 5.589241, 6.99844643, 4.79288192],
    [4.55176485, 5.6122494, 7.09923364, 4.82144704],
    [4.21254506, 5.31988824, 6.84520426, 4.79017392]
])

# Initialize epsilon for numerical stability
epsilon = 1e-6

# Calculate mean and variance for each row
mean = np.mean(M, axis=1, keepdims=True)
variance = np.var(M, axis=1, keepdims=True)

# Compute layer normalization
LN_M = (M - mean) / np.sqrt(variance + epsilon)

print(LN_M)

```

Output:

```
[[-1.03927142  0.13949668  1.56694829 -0.66717355]
 [-0.97825977  0.09190721  1.59246789 -0.70611533]
 [-1.10306273  0.02854757  1.58729045 -0.51277529]]
```

#### Linear Regression Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tensorflow as tf

# Generate data
X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = np.sin(X)

# Create and train the linear model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Predict with linear model
y_linear_pred = linear_model.predict(X)

# Create and train the neural network model
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,), activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X, y, epochs=1000, verbose=0)

# Predict with neural network model
y_nn_pred = nn_model.predict(X)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Linear model plot
axs[0].scatter(X, y, label='True Function', color='blue')
axs[0].plot(X, y_linear_pred, label='Linear Model', color='red')
axs[0].legend()
axs[0].set_title('Linear Model')

# Deep Feedforward Network plot
axs[1].scatter(X, y, label='True Function', color='blue')
axs[1].plot(X, y_nn_pred, label='Deep Network', color='green')
axs[1].legend()
axs[1].set_title('Deep Feedforward Network')

plt.show()

```

Output:

![](images/2023-11-01-08-31-24.png)

#### Plot XOR


```python
import matplotlib.pyplot as plt

def plot_xor():
    # Coordinates for the points
    x = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    
    # XOR values for the points
    colors = [0, 1, 1, 0]
    
    # Plot the points
    plt.scatter(x, y, c=colors, s=100, cmap="gray", edgecolors="black", linewidth=1.5)
    
    # Set axis labels and title
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Original $x$ space")
    
    # Adjust axis limits
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    # Show grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()

plot_xor()
```

Output:

![](images/2023-11-01-09-01-39.png)