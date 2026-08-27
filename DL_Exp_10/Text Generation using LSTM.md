# Text Generation using LSTM

## AIM
To study and implement a text generation model using Long Short-Term Memory (LSTM) networks, and to understand how LSTM can learn the patterns and structure of a text corpus to generate new, coherent text sequences.

## THEORY

### Introduction
**Text Generation** is a Natural Language Processing (NLP) task in which a model learns the statistical structure of a given text corpus and uses that knowledge to generate new text that resembles the style, vocabulary, and grammar of the original data. It is a classic example of **sequence modeling**, where the goal is to predict the next element (character or word) in a sequence given the previous elements.

**LSTM (Long Short-Term Memory)** networks are particularly well-suited for text generation because they can capture long-range dependencies in sequential data — remembering relevant context from earlier in a sentence or paragraph — which is essential for generating grammatically correct and contextually meaningful text.

### Why LSTM for Text Generation?
Text is inherently sequential, and the meaning of a word or character often depends heavily on the words/characters that came before it. A simple RNN struggles to retain information over long sequences due to the vanishing gradient problem, making it difficult to model dependencies that span many words. LSTM overcomes this limitation using its gating mechanism (forget, input, and output gates) and a dedicated cell state, allowing it to selectively remember important context over long sequences — making it an effective choice for generating coherent and contextually relevant text.

### Approach: Character-Level vs Word-Level Text Generation
1. **Character-Level Generation**
   The model is trained to predict the next character in a sequence, given the previous characters. It learns spelling, punctuation, and basic sentence structure from scratch, and can even generate new/invented words.

2. **Word-Level Generation**
   The model is trained to predict the next word in a sequence, given the previous words. It typically requires a vocabulary of known words and tends to produce more grammatically coherent and semantically meaningful text, but requires a larger dataset and vocabulary handling (tokenization).

### Workflow of Text Generation using LSTM

1. **Data Collection and Preprocessing**
   - Collect a text corpus (e.g., a book, articles, or dialogues).
   - Clean the text (remove unwanted characters, normalize case, etc.).
   - Tokenize the text into characters or words.
   - Build a vocabulary mapping each unique character/word to an integer index (and vice versa).

2. **Sequence Preparation**
   - Convert the text into overlapping input-output sequence pairs using a **sliding window** approach.
   - For example, given a sequence length of `n`, the model is trained to predict the `(n+1)th` character/word based on the previous `n` characters/words.
   - Input sequences are converted into numerical form (integer encoding or one-hot encoding) and often normalized.

3. **Model Architecture**
   A typical text generation model consists of:
   - **Embedding Layer** (for word-level models): Converts word indices into dense vector representations.
   - **LSTM Layer(s)**: One or more stacked LSTM layers to learn temporal dependencies in the sequence.
   - **Dropout Layer**: Used to prevent overfitting by randomly deactivating neurons during training.
   - **Dense (Fully Connected) Output Layer**: With a **Softmax activation function**, producing a probability distribution over all possible next characters/words in the vocabulary.

4. **Training the Model**
   - The model is trained using **categorical cross-entropy loss**, comparing the predicted probability distribution with the actual next character/word.
   - An optimizer such as **Adam** or **RMSprop** is used to update the weights via backpropagation through time (BPTT).
   - The model is trained over multiple epochs until it can accurately predict the next element in a sequence.

5. **Text Generation (Inference)**
   - Provide a **seed sequence** (a starting string of text) to the trained model.
   - The model predicts the probability distribution for the next character/word.
   - A character/word is sampled from this distribution (using techniques like greedy sampling, temperature-based sampling, or top-k sampling).
   - The predicted character/word is appended to the seed sequence, and the oldest element is dropped to maintain a fixed input length (sliding window).
   - This process is repeated iteratively to generate a sequence of desired length.

### Role of Temperature in Sampling
**Temperature** is a hyperparameter used during sampling to control the randomness/creativity of the generated text:
- **Low temperature (< 1.0)**: Makes the model more confident and conservative, producing more predictable and repetitive text.
- **High temperature (> 1.0)**: Increases randomness, producing more diverse and creative (but potentially less coherent) text.
```
p_i = exp(log(p_i) / T) / Σ exp(log(p_j) / T)
```
Where `T` is the temperature and `p_i` is the predicted probability of class `i`.

### Applications of Text Generation
- Chatbots and conversational AI
- Auto-completion and predictive text (e.g., in messaging apps)
- Content creation (story generation, poetry, articles)
- Code generation
- Language translation (as a component of larger sequence-to-sequence systems)

### Advantages of LSTM-based Text Generation
- Captures long-term contextual dependencies, producing more coherent text.
- Can learn the style, tone, and structure of the training corpus.
- Flexible for both character-level and word-level generation tasks.

### Limitations
- Requires large amounts of training data to generate high-quality, coherent text.
- Sequential processing makes training and generation relatively slow.
- May struggle with maintaining long-range coherence over very long generated passages compared to newer architectures like Transformers.
- Generated text can sometimes be repetitive or grammatically inconsistent, especially with limited training data.

## CONCLUSION
The text generation model using LSTM was successfully studied and implemented. It was observed that LSTM networks, by learning the sequential patterns and long-term dependencies present in a text corpus, are able to generate new text that reflects the structure, style, and vocabulary of the training data. This demonstrates the effectiveness of LSTM-based architectures in sequence generation tasks and highlights their significance in various real-world Natural Language Processing applications.

---

### Screenshot

<img width="785" height="447" alt="image" src="https://github.com/user-attachments/assets/f184a97f-dd59-46cd-9649-ceb6b33553f0" />
