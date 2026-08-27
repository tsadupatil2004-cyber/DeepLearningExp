# Simple RNN

## AIM
To study and implement a Simple Recurrent Neural Network (RNN) for sequential data processing and to understand its architecture, working principle, and applications.

## THEORY

### Introduction
A **Recurrent Neural Network (RNN)** is a class of artificial neural networks specifically designed to handle **sequential data**, such as time series, text, speech, and video. Unlike traditional feedforward neural networks, which assume that inputs are independent of each other, RNNs have connections that form directed cycles, allowing information to persist across time steps. This makes RNNs particularly suitable for tasks where the order and context of data matter.

The key idea behind an RNN is the concept of **memory** — the network maintains a hidden state that captures information about previous inputs in the sequence, which is then used along with the current input to produce an output.

### Architecture of a Simple RNN
A simple RNN consists of the following components:

1. **Input Layer**
   Accepts sequential input data, one element (time step) at a time, e.g., a word in a sentence or a value in a time series.

2. **Hidden Layer (Recurrent Layer)**
   Maintains a hidden state that is updated at every time step based on the current input and the previous hidden state. This is what gives the network its "memory."

3. **Output Layer**
   Produces an output at each time step (or a single output after processing the entire sequence, depending on the task).

### Working Principle
At each time step **t**, the RNN takes two inputs:
- The current input `x(t)`
- The previous hidden state `h(t-1)`

It then computes the new hidden state and output using the following equations:

```
h(t) = f( Wxh * x(t) + Whh * h(t-1) + bh )
y(t) = g( Why * h(t) + by )
```

Where:
- `h(t)` = hidden state at time step t
- `x(t)` = input at time step t
- `Wxh` = weight matrix from input to hidden layer
- `Whh` = weight matrix from hidden to hidden layer (recurrent weights)
- `Why` = weight matrix from hidden to output layer
- `bh, by` = bias terms
- `f` = activation function (typically tanh or ReLU)
- `g` = output activation function (e.g., Softmax for classification)

This process is repeated for every time step in the sequence, and the same weights (`Wxh`, `Whh`, `Why`) are **shared across all time steps**, which significantly reduces the number of parameters compared to using a separate network for each time step.

### Unfolding the RNN
An RNN can be visualized as a chain of repeating units, where each unit corresponds to one time step. This "unfolded" representation shows how the hidden state flows from one time step to the next, carrying forward information learned from earlier inputs.

### Training an RNN — Backpropagation Through Time (BPTT)
RNNs are trained using a variant of backpropagation called **Backpropagation Through Time (BPTT)**. In BPTT:
1. The network is unfolded across all time steps.
2. Gradients of the loss function are calculated at each time step.
3. These gradients are propagated backward through the unfolded network to update the shared weights.

### Vanishing and Exploding Gradient Problem
A major limitation of simple RNNs is the **vanishing gradient problem**, where gradients become extremely small as they are propagated back through many time steps, making it difficult for the network to learn long-term dependencies. Conversely, gradients can also grow uncontrollably, causing the **exploding gradient problem**. These issues limit simple RNNs from effectively learning long sequences, which led to the development of advanced variants like **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)**.

### Applications of RNN
- Natural Language Processing (text generation, sentiment analysis, machine translation)
- Speech recognition
- Time series prediction (stock prices, weather forecasting)
- Music generation
- Video frame analysis

### Advantages of RNN
- Capable of processing sequences of arbitrary length.
- Shares weights across time steps, reducing the number of parameters.
- Maintains context/memory of previous inputs, useful for sequential dependencies.

### Limitations
- Struggles with long-term dependencies due to vanishing/exploding gradients.
- Sequential processing makes training slower compared to feedforward networks.
- Simple RNNs are largely replaced by LSTM/GRU in practical applications requiring long-range memory.

## CONCLUSION
The Simple Recurrent Neural Network was successfully studied and implemented. It was observed that RNNs are effective for processing sequential data by maintaining a hidden state that captures information from previous time steps, enabling the network to model temporal dependencies. However, simple RNNs face challenges with long-term dependencies due to the vanishing gradient problem, highlighting the need for advanced architectures such as LSTM and GRU for more complex sequential tasks.

---

### Screenshot

<img width="1523" height="462" alt="image" src="https://github.com/user-attachments/assets/6027d41b-c1a4-4971-bb7a-dbe1dd23adf8" />
