# RNN with GRU

## AIM
To study and implement a Recurrent Neural Network using Gated Recurrent Unit (GRU) for sequential data processing, and to understand how GRU addresses the limitations of a simple RNN with a simplified gating mechanism compared to LSTM.

## THEORY

### Introduction
**Gated Recurrent Unit (GRU)** is a variant of the Recurrent Neural Network (RNN) architecture, introduced by Cho et al. as a simpler alternative to LSTM. Like LSTM, GRU was designed to solve the **vanishing gradient problem** faced by simple RNNs and to effectively capture long-term dependencies in sequential data. However, GRU achieves this with a more streamlined architecture by combining and reducing the number of gates, resulting in fewer parameters and often faster training, while maintaining performance comparable to LSTM on many tasks.

Unlike LSTM, which maintains a separate cell state and hidden state, GRU merges these into a single hidden state, and it uses only two gates — the **update gate** and the **reset gate** — instead of the three gates used in LSTM.

### Why GRU?
Simple RNNs fail to capture long-term dependencies due to vanishing/exploding gradients. While LSTM solves this problem effectively, its architecture is relatively complex with three gates and two separate states (cell state and hidden state), leading to a higher computational cost. GRU simplifies this design by using fewer gates and a single state, reducing computational complexity while still being able to control the flow of information and retain relevant long-term dependencies.

### Architecture of GRU
A GRU unit consists of a single hidden state (ht) and two gates:

1. **Update Gate (zt)**
   Decides how much of the past information (from previous time steps) needs to be carried forward to the future, combining the roles of the forget and input gates in LSTM.
   ```
   zt = σ( Wz . [h(t-1), x(t)] + bz )
   ```

2. **Reset Gate (rt)**
   Decides how much of the past information to forget, controlling how the new input interacts with the previous memory.
   ```
   rt = σ( Wr . [h(t-1), x(t)] + br )
   ```

3. **Candidate Hidden State (h̃t)**
   Computes a candidate activation using the reset gate to control the influence of the previous hidden state.
   ```
   h̃t = tanh( W . [rt * h(t-1), x(t)] + b )
   ```

4. **Final Hidden State Update (ht)**
   Combines the previous hidden state and the candidate hidden state using the update gate.
   ```
   ht = (1 - zt) * h(t-1) + zt * h̃t
   ```

Where:
- `σ` = sigmoid activation function (outputs values between 0 and 1, acting as a gate)
- `tanh` = hyperbolic tangent activation function
- `x(t)` = input at time step t
- `h(t-1)` = hidden state from the previous time step
- `W, b` = weight matrices and bias terms for each gate

### Working Principle
1. The **reset gate** determines how much of the previous hidden state should be ignored when computing the new candidate hidden state, allowing the network to "forget" irrelevant past information when necessary.
2. The **candidate hidden state** is computed using the current input and the (possibly reset) previous hidden state, representing new information to potentially be added to memory.
3. The **update gate** decides the balance between retaining the previous hidden state and adopting the new candidate hidden state — acting as an interpolation between old and new information.
4. The final hidden state is a blend of the previous hidden state and the candidate hidden state, weighted by the update gate, which is then passed to the next time step and used to generate the output.
5. This mechanism allows GRU to selectively retain important long-term information while still being responsive to new input, similar to LSTM but with a simpler, more computationally efficient structure.

### Training
GRUs are trained using **Backpropagation Through Time (BPTT)**, similar to simple RNNs and LSTMs. The gating mechanism allows gradients to flow more effectively through time steps, mitigating the vanishing gradient problem and enabling the network to learn long-range dependencies.

### GRU vs LSTM
| Feature | LSTM | GRU |
|---|---|---|
| Number of Gates | 3 (Forget, Input, Output) | 2 (Update, Reset) |
| States Maintained | Cell state + Hidden state | Single Hidden state |
| Number of Parameters | More | Fewer |
| Training Speed | Slower | Faster |
| Performance | Slightly better on some complex/long-sequence tasks | Comparable, often similar accuracy |
| Computational Cost | Higher | Lower |

### Applications of GRU
- Natural Language Processing (text classification, machine translation, sentiment analysis)
- Speech recognition
- Time series forecasting
- Music generation
- Any sequence modeling task where computational efficiency is important

### Advantages of GRU
- Simpler architecture with fewer parameters compared to LSTM.
- Faster training and lower computational cost.
- Effectively mitigates the vanishing gradient problem.
- Performs comparably to LSTM on many sequence modeling tasks.

### Limitations
- May underperform compared to LSTM on tasks requiring very long-term dependencies or highly complex sequence patterns.
- Still processes sequences sequentially, limiting parallelization compared to newer architectures like Transformers.
- Choice between GRU and LSTM often requires empirical testing, as performance can vary by dataset and task.

## CONCLUSION
The RNN with GRU was successfully studied and implemented. It was observed that GRU, through its simplified gating mechanism consisting of update and reset gates, effectively addresses the vanishing gradient problem and captures long-term dependencies in sequential data, while being computationally more efficient than LSTM due to its reduced number of parameters and single hidden state. This makes GRU a practical and efficient choice for sequence modeling tasks, especially when training speed and computational resources are important considerations.

---

### screenshot

<img width="779" height="602" alt="image" src="https://github.com/user-attachments/assets/4a8a8b12-fb5e-4f13-a78e-8c8b42a76a38" />
