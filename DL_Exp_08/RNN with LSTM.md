# RNN with LSTM

## AIM
To study and implement a Recurrent Neural Network using Long Short-Term Memory (LSTM) units for sequential data processing, and to understand how LSTM overcomes the limitations of a simple RNN.

## THEORY

### Introduction
**Long Short-Term Memory (LSTM)** is a special kind of Recurrent Neural Network (RNN) architecture designed to effectively learn and remember long-term dependencies in sequential data. It was introduced by Hochreiter and Schmidhuber to address the **vanishing and exploding gradient problem** faced by simple RNNs, which makes it difficult for them to retain information over long sequences.

Unlike a simple RNN, which has a single hidden state that is repeatedly overwritten at every time step, an LSTM maintains a more sophisticated internal structure consisting of a **cell state** and multiple **gates** that regulate the flow of information — allowing the network to selectively remember or forget information over long periods.

### Why LSTM?
In a simple RNN, as sequences get longer, gradients calculated during backpropagation through time either shrink towards zero (vanishing gradient) or grow uncontrollably (exploding gradient). This prevents the network from learning dependencies that span many time steps. LSTM solves this problem through a **gating mechanism** that controls how information is added, retained, or removed from the network's memory, enabling it to capture long-range dependencies effectively.

### Architecture of LSTM
An LSTM unit consists of a **cell state** (Ct) and three gates that regulate information flow:

1. **Forget Gate (ft)**
   Decides what information should be discarded from the cell state.
   ```
   ft = σ( Wf . [h(t-1), x(t)] + bf )
   ```

2. **Input Gate (it)**
   Decides what new information should be added to the cell state.
   ```
   it = σ( Wi . [h(t-1), x(t)] + bi )
   C̃t = tanh( Wc . [h(t-1), x(t)] + bc )
   ```

3. **Cell State Update (Ct)**
   Combines the forget gate and input gate to update the cell state.
   ```
   Ct = ft * C(t-1) + it * C̃t
   ```

4. **Output Gate (ot)**
   Decides what part of the cell state should be output as the hidden state.
   ```
   ot = σ( Wo . [h(t-1), x(t)] + bo )
   h(t) = ot * tanh(Ct)
   ```

Where:
- `σ` = sigmoid activation function (outputs values between 0 and 1, acting as a gate)
- `tanh` = hyperbolic tangent activation function
- `x(t)` = input at time step t
- `h(t-1)` = hidden state from the previous time step
- `C(t-1)` = cell state from the previous time step
- `W, b` = weight matrices and bias terms for each gate

### Working Principle
1. The **forget gate** examines the previous hidden state and current input to decide what information from the previous cell state should be discarded.
2. The **input gate** determines what new information from the current input should be stored in the cell state.
3. The **cell state** is updated by combining the retained old information (via forget gate) and the new candidate information (via input gate), forming a continuous "memory conveyor belt" that runs through the entire sequence.
4. The **output gate** determines what part of the updated cell state should be exposed as the hidden state, which is passed to the next time step and/or used to generate the output.
5. This gating mechanism allows the LSTM to selectively retain important information over long sequences while discarding irrelevant information, effectively solving the long-term dependency problem.

### Training
Like simple RNNs, LSTMs are trained using **Backpropagation Through Time (BPTT)**. However, because of the gated architecture and the additive nature of the cell state update, gradients can flow through the network more effectively over long sequences without vanishing, allowing the model to learn long-range dependencies.

### Applications of LSTM
- Natural Language Processing (text generation, machine translation, sentiment analysis)
- Speech recognition and synthesis
- Time series forecasting (stock market prediction, sensor data analysis)
- Handwriting recognition
- Video captioning and activity recognition

### Advantages of LSTM
- Effectively captures long-term dependencies in sequential data.
- Overcomes the vanishing gradient problem present in simple RNNs.
- Gating mechanism provides fine control over what information to remember or forget.
- Performs well on a wide range of sequence modeling tasks.

### Limitations
- More computationally expensive than simple RNNs due to the increased number of parameters (multiple gates).
- Training can be slower because of the added complexity.
- Still processes sequences step-by-step, limiting parallelization compared to newer architectures like Transformers.

## CONCLUSION
The RNN with LSTM was successfully studied and implemented. It was observed that the LSTM architecture, through its use of forget, input, and output gates along with a dedicated cell state, effectively overcomes the vanishing gradient problem faced by simple RNNs and successfully captures long-term dependencies in sequential data. This makes LSTM a powerful and widely used architecture for tasks involving long sequences, such as language modeling, time series forecasting, and speech recognition.

---

### Screenshot

<img width="881" height="590" alt="image" src="https://github.com/user-attachments/assets/3d3320b5-8462-41c3-a645-7e64a6f3842a" />
