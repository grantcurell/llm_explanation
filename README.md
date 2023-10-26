# LLM Mathematical Analysis

## Ovarall Process

Feed-forward neural networks are an integral component within Large Language Models (LLMs) like the Transformer architecture used in GPT. These feed-forward networks typically operate at the level of each individual token's representation and perform the same operation across different positions. They come into play in each layer of the Transformer architecture, along with the multi-head attention mechanism.

Here's a simplified breakdown of how feed-forward neural networks are used within a single layer of the Transformer model:

1. **Input Embedding**: Each token in the input sequence is converted into a fixed-size vector using an embedding layer.
  
2. **Multi-Head Attention**: These vectors are transformed using self-attention mechanisms to consider other tokens in the sequence.

3. **Feed-Forward Neural Network**: The output of the multi-head attention step for each token is then passed through a position-wise feed-forward neural network.

The feed-forward neural network in each Transformer layer usually consists of two main steps:

1. **Linear Transformation**: The vectors are first linearly transformed using a weight matrix and a bias term.
  
2. **Activation Function**: A non-linear activation function, like ReLU (Rectified Linear Unit) or GELU (Gaussian Error Linear Unit), is applied to introduce non-linearity.

3. **Another Linear Transformation**: A second linear transformation usually follows the activation.

Mathematically, if $\mathbf{x}$ is the input to the feed-forward network, and $W_1, b_1, W_2, b_2$ are the weights and biases, the operations could be expressed as:

$$
\mathbf{FFN}(\mathbf{x}) = W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{x} + b_1) + b_2
$$

This structure of using feed-forward neural networks within each layer is repeated multiple times to form a deep Transformer model. These stacked layers collectively enable the model to learn complex relationships in the data and perform various language tasks.

### Input Embedding

**TODO** EXPLAIN WHY IT WORKS THIS WAY

Certainly. Input embedding is a mapping from discrete tokens to continuous vectors in a high-dimensional space. In the simplest form, input embedding can be represented as a lookup table $W$ of dimensions $V \times D$, where $V$ is the vocabulary size and $D$ is the embedding dimension.

Suppose we have a vocabulary $\mathcal{V} = \{\text{"apple"}, \text{"banana"}, \text{"cherry"}\}$ with indices $\{1, 2, 3\}$, and we choose an embedding dimension $D = 4$.

Then the embedding table $W$ could look something like this (numbers are arbitrary):

$$
W = \begin{pmatrix}
0.2 & -0.1 & 0.4 & 0.8 \\
-0.5 & 0.9 & 0.3 & 0.1 \\
0.7 & -0.6 & -0.2 & 0.4
\end{pmatrix}
$$

Now, let's say we have an input sentence: "apple banana". The corresponding token indices could be $[1, 2]$.

The embedding for "apple" would be the vector corresponding to the first row of $W$:

$$
\text{Embedding}(\text{"apple"}) = [0.2, -0.1, 0.4, 0.8]
$$

And the embedding for "banana" would be the second row:

$$
\text{Embedding}(\text{"banana"}) = [-0.5, 0.9, 0.3, 0.1]
$$

To form the input representation for the entire sequence, you simply concatenate or stack these vectors:

$$
\text{Input Matrix} = \begin{pmatrix}
0.2 & -0.1 & 0.4 & 0.8 \\
-0.5 & 0.9 & 0.3 & 0.1
\end{pmatrix}
$$

Each row in this matrix serves as the input representation for a corresponding token in the sequence, which can then be passed through the rest of the model.

## Attention Head

Certainly. The Multi-Head Attention mechanism in a Transformer consists of multiple attention heads that perform scaled dot-product attention in parallel. Here, I'll simplify the process by discussing only one attention head.

Let's say our input matrix from the embedding layer is $X$ with shape $T \times D$, where $T$ is the sequence length and $D$ is the embedding dimension. For simplicity, $T = 2$ and $D = 4$ from the previous example:

$$
X = \begin{pmatrix}
0.2 & -0.1 & 0.4 & 0.8 \\
-0.5 & 0.9 & 0.3 & 0.1
\end{pmatrix}
$$

### Scaled Dot-Product Attention

The first step is to compute Query (Q), Key (K), and Value (V) matrices using learned weight matrices $W^Q, W^K,$ and $W^V$ respectively.

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

Let's assume:

$$
W^Q = W^K = W^V = \begin{pmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{pmatrix}
$$

Calculating $Q, K, V$:

$$
Q = K = V = X \begin{pmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{pmatrix} = X
$$

We keep it simple for illustration. In practice, these matrices are usually different.

### Attention Score

Next, we compute the attention scores by taking the dot product of $Q$ and $K^T$, followed by scaling down by $\sqrt{D_k}$, where $D_k$ is the key dimension:

$$
\text{Score} = \frac{Q K^T}{\sqrt{D_k}} = \frac{X X^T}{\sqrt{4}} = \frac{1}{2} \begin{pmatrix}
0.2 & -0.1 & 0.4 & 0.8 \\
-0.5 & 0.9 & 0.3 & 0.1
\end{pmatrix} \begin{pmatrix}
0.2 & -0.5 \\
-0.1 & 0.9 \\
0.4 & 0.3 \\
0.8 & 0.1
\end{pmatrix}
$$

Calculating, we get:

$$
\text{Score} = \frac{1}{2} \begin{pmatrix}
0.85 & 0.05 \\
0.05 & 1.02
\end{pmatrix}
$$

### Softmax and Output

The next step is to apply the softmax function to the scores along each row:

$$
\text{Softmax(Score)} = \begin{pmatrix}
\frac{e^{0.85/2}}{e^{0.85/2} + e^{0.05/2}} & \frac{e^{0.05/2}}{e^{0.85/2} + e^{0.05/2}} \\
\frac{e^{0.05/2}}{e^{1.02/2} + e^{0.05/2}} & \frac{e^{1.02/2}}{e^{1.02/2} + e^{0.05/2}}
\end{pmatrix}
$$

After calculating the softmax, we multiply the output by the Value matrix $V$ and sum along the rows to get the new representation $Z$:

$$
Z = \text{Softmax(Score)} \times V
$$

This resulting $Z$ will then be used as the input for the subsequent feed-forward neural networks within the Transformer layer. The multiple heads in Multi-Head Attention would perform this operation in parallel and their outputs would be concatenated and linearly transformed to produce the final output for the next layer.


## [Large language models, explained with a minimum of math and jargon](https://www.understandingai.org/p/large-language-models-explained-with)

### High Level

The models behind ChatGPT would then break that prompt into tokens. On average, a token is ⅘ of a word, so the above prompt and its 23 words might result in about 30 tokens. The GPT-3 model that gpt-3.5-turbo model is based on has 175 billion weights. The GPT-4 model, which is also available in ChatGPT, has an unknown number of weights.

Then, the model would set about generating a response that sounds right based on the immense volume of text that it consumed during its training. Importantly, it is not looking up anything about the query. It does not have any memory wherein it can search for “dataiku,” “value proposition,” “software,” or any other relevant terms. Instead, it sets about generating each token of output text, it performs the computation again, generating a token that has the highest probability of sounding right. 

### Word Vectors

Language models take a similar approach: each word vector1 represents a point in an imaginary “word space,” and words with more similar meanings are placed closer together. For example, the words closest to cat in vector space include dog, kitten, and pet. A key advantage of representing words with vectors of real numbers (as opposed to a string of letters, like “C-A-T”) is that numbers enable operations that letters don’t. 

Words are too complex to represent in only two dimensions, so language models use vector spaces with hundreds or even thousands of dimensions. The human mind can’t envision a space with that many dimensions, but computers are perfectly capable of reasoning about them and producing useful results.

![](2023-10-25-15-30-48.png)

The model’s input, shown at the bottom of the diagram, is the partial sentence “John wants his bank to cash the.” These words, represented as word2vec-style vectors, are fed into the first transformer. 

The transformer figures out that wants and cash are both verbs (both words can also be nouns). We’ve represented this added context as red text in parentheses, but in reality the model would store it by modifying the word vectors in ways that are difficult for humans to interpret. These new vectors, known as a hidden state, are passed to the next transformer in the stack.

Research suggests that the first few layers focus on understanding the syntax of the sentence and resolving ambiguities like we’ve shown above. Later layers (which we’re not showing to keep the diagram a manageable size) work to develop a high-level understanding of the passage as a whole.

### How Transformers Work

1. In the attention step, words “look around” for other words that have relevant context and share information with one another.
2. In the feed-forward step, each word “thinks about” information gathered in previous attention steps and tries to predict the next word.

You can think of the attention mechanism as a matchmaking service for words. Each word makes a checklist (called a query vector) describing the characteristics of words it is looking for. Each word also makes a checklist (called a key vector) describing its own characteristics. The network compares each key vector to each query vector (by computing a dot product) to find the words that are the best match. Once it finds a match, it transfers information from the word that produced the key vector to the word that produced the query vector.

For example, in the previous section we showed a hypothetical transformer figuring out that in the partial sentence “John wants his bank to cash the,” his refers to John. Here’s what that might look like under the hood. The query vector for his might effectively say “I’m seeking: a noun describing a male person.” The key vector for John might effectively say “I am: a noun describing a male person.” The network would detect that these two vectors match and move information about the vector for John into the vector for his.

Each attention layer has several “attention heads,” which means that this information-swapping process happens several times (in parallel) at each layer. Each attention head focuses on a different task:

- One attention head might match pronouns with nouns, as we discussed above.
- Another attention head might work on resolving the meaning of homonyms like bank.
- A third attention head might link together two-word phrases like “Joe Biden.”

And so forth.

Attention heads frequently operate in sequence, with the results of an attention operation in one layer becoming an input for an attention head in a subsequent layer. Indeed, each of the tasks we just listed above could easily require several attention heads rather than just one.

The largest version of GPT-3 has 96 layers with 96 attention heads each, so GPT-3 performs 9,216 attention operations each time it predicts a new word.

## [Transformers Explained Visually](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)

### What does Attention Do?

While processing a word, Attention enables the model to focus on other words in the input that are closely related to that word.



## My Notes

- Jo større modellen er, desto mer sannsynlig skjer feil.
- https://arxiv.org/pdf/2001.08361.pdf
- [Transformers Explained Visually Part 1: Overview of Functionality](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)
- [Transformers Explained Visually (Part 2): How it works, step-by-step](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34)