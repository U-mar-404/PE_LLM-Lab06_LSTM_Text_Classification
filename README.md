# Text Classification Using LSTM

This lab uses an **LSTM neural network** to classify IMDB movie reviews
as **positive or negative**. It also compares LSTM with SimpleRNN to
understand how they handle long text.

## What's Included

-   IMDB movie review dataset
-   Text padding and embedding
-   LSTM model built with TensorFlow/Keras
-   Training and evaluation
-   Accuracy/loss plots and confusion matrix
-   Predictions for new sentences
-   SimpleRNN vs LSTM experiments with sequence lengths of 200 and 400
-   Manual verification of LSTM equations against Keras

## Requirements

-   Python
-   TensorFlow
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn

Install the libraries:

``` bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## How to Run

1.  Download or clone this repository.
2.  Open `Lab06_LSTM_Text_Classification.ipynb` in Jupyter Notebook or
    Google Colab.
3.  Run the cells from top to bottom.
4.  The IMDB dataset downloads through Keras when needed.

## Lab Questions and Answers

### 1. Why does a SimpleRNN forget the beginning of a long review?

A SimpleRNN can lose old information as it reads more words, making it
hard to remember important words from the beginning.

### 2. What is the difference between the cell state and hidden state?

The **cell state** stores long-term information. The **hidden state**
carries information used for the current output.

### 3. What does the forget gate do? What if its value is 0 or 1?

The forget gate decides how much old information to keep. A value of
**0** means forget the old information; a value of **1** means keep it
all.

### 4. Why do gates use sigmoid but candidate memory uses tanh?

Sigmoid gives values between 0 and 1 to control information. Tanh gives
values between -1 and 1 to create candidate information.

### 5. Why does an LSTM have about 4 times as many parameters as SimpleRNN?

An LSTM has four sets of weights for the input gate, forget gate,
candidate memory, and output gate. SimpleRNN has one set.

### 6. Why is the LSTM cell-state update better for keeping information?

The update `C_t = f_t * C_(t-1) + i_t * candidate` lets useful old
information pass forward while adding new information. This can help the
model remember information for longer.

### 7. Compare SimpleRNN and LSTM for sequence lengths 200 and 400. Where did LSTM help, and what did it cost?

LSTM can remember useful information better in long reviews. However, it
has more parameters and usually takes more time and computing power to
train. Check the experiment table in the notebook for the actual
accuracy and time results.

### 8. Give a sentence where early words matter for sentiment. Did both models get it right?

Example: "I thought the movie would be boring, but it was amazing." The
early words differ from the final opinion. Whether both models classify
it correctly depends on the predictions from the notebook.

## Key Takeaway

LSTM uses three gates to decide what to forget, what new information to
store, and what information to output. This helps it keep useful
information across longer text sequences.
