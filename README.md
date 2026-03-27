# TinyGPT

A GPT-style language model built from scratch in PyTorch — written as a learning project to understand how large language models actually work under the hood.

## What This Is

Most people interact with LLMs without knowing what's happening inside them. This project is an attempt to build one from first principles — no high-level abstractions, just PyTorch and math.

The model implements the core transformer architecture: tokenization, token embeddings, multi-head self-attention, feed-forward layers, residual connections, and a training loop with next-token prediction. It's trained on public domain mountaineering texts from Project Gutenberg.

## How It Works

**Tokenization** — Raw text is broken into characters, each mapped to an integer ID and then looked up in an embedding table to produce a vector.

**Multi-Head Self-Attention** — Each token computes Query, Key, and Value vectors. Dot products between Q and K produce relevance scores across all token pairs, which are normalized via softmax into attention weights. These weights blend the Value vectors into enriched, context-aware representations. Multiple attention heads run in parallel, each learning to notice different types of relationships.

**Feed-Forward Network** — After attention, each token's vector is independently passed through a two-layer network that expands into a larger working space, applies ReLU activation, and compresses back down — allowing the model to draw complex conclusions from the contextual information attention gathered.

**Residual Connections** — The input to each block is added back to its output, preserving the original token meaning while layering new information on top.

**Training** — The model is trained with next-token prediction: given a sequence of characters, predict what comes next. Cross-entropy loss measures how wrong the model was, backpropagation traces responsibility back through every weight matrix, and AdamW nudges each weight in the direction that reduces error.

## Architecture

```
Embedding dim:   64
Attention heads: 4
Transformer layers: 4
Block size:      64 tokens
Batch size:      16
```

## Training Data

Public domain mountaineering books from Project Gutenberg, including *Mountaineering in the Sierra Nevada* by Clarence King.

## Results

Loss drops from ~4.8 at initialization to ~2.2 after 1000 training steps. Generated text shows learned character-level patterns (common letter combinations, punctuation habits) consistent with the training corpus.

## What I Learned

- How attention produces contextual token representations through weighted vector blending
- Why Q, K, and V are separate transformations rather than raw embeddings
- How multi-head attention allows parallel specialization across different relationship types
- Why residual connections and layer normalization are critical for stable training
- How backpropagation and gradient descent actually update weights
- The tradeoffs between character-level and subword tokenization

## Stack

- Python 3.12
- PyTorch
