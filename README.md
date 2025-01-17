# Recurrent Neural Networks (RNN) Lab

This repository contains the **Lab Work (TP N°4)** focused on **Recurrent Neural Networks (RNNs)** and their applications in text classification. The lab explores three types of deep learning models: **Dense Neural Networks**, **Convolutional Neural Networks (CNN)**, and **Long Short-Term Memory (LSTM)** networks. The goal is to classify movie reviews from the **IMDB dataset** as either "positive" or "negative" using these models.

---

## 🚀 Objectives

- **Apply Deep Learning Models**: Use Dense Neural Networks, CNNs, and LSTMs for text classification.
- **Text Preprocessing**: Clean and preprocess text data using techniques like tokenization, padding, and embedding.
- **Word Embeddings**: Utilize **GloVe** embeddings to convert text into numerical form.
- **Model Evaluation**: Compare the performance of the three models using metrics like accuracy and loss.

---

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow/Keras
- **Libraries**: NumPy, Pandas, NLTK, Matplotlib, Seaborn
- **Word Embeddings**: GloVe (Global Vectors for Word Representation)

---


## 🚀 Lab Overview

### 1. **Dataset**
   - **IMDB Dataset**: Contains 50,000 movie reviews labeled as "positive" or "negative".
   - **Tasks**:
     - Load and explore the dataset.
     - Check for class imbalance using visualizations.
     - Preprocess the text data (remove HTML tags, punctuation, etc.).

### 2. **Text Preprocessing**
   - **Tokenization**: Convert text into sequences of integers.
   - **Padding**: Ensure all sequences have the same length.
   - **Word Embeddings**: Use GloVe embeddings to represent words as vectors.

### 3. **Model Building**
   - **Dense Neural Network**: A simple feedforward neural network for baseline performance.
   - **Convolutional Neural Network (CNN)**: Apply 1D convolutions to capture local patterns in text.
   - **Long Short-Term Memory (LSTM)**: Use LSTM layers to capture sequential dependencies in text.

### 4. **Model Evaluation**
   - **Metrics**: Evaluate models using accuracy and loss.
   - **Visualization**: Plot training and validation accuracy/loss over epochs.
   - **Comparison**: Compare the performance of the three models.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SaifeddineBENZAIED/recurrent-neural-networks-lab.git
   cd recurrent-neural-networks-lab

2. Open the Jupyter Notebook and start exploring:
   ```bash
   jupyter notebook TP_RNN.ipynb

🔍 Key Features

Text Classification

- Dense Neural Network: Baseline model for text classification.

- CNN: Captures local patterns in text using 1D convolutions.

- LSTM: Captures sequential dependencies in text using recurrent layers.

Word Embeddings

- GloVe: Pre-trained word embeddings for converting text into numerical form.

Model Evaluation

- Accuracy and Loss: Metrics used to evaluate model performance.

- Visualization: Plots of training and validation accuracy/loss over epochs.

📫 Contact
For questions or feedback, feel free to reach out:

- Email: saif2001benz2036@gmail.com
