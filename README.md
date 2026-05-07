# XLM-RoBERTa BiGRU: Multilingual Sexual Predator Identification

A deep learning project for identifying sexual predators in online conversations using XLM-RoBERTa with Bidirectional GRU, supporting both English and Indonesian datasets.

## 📋 Overview

This project implements a binary classification model to detect sexual predators in conversational data. The model combines:
- **XLM-RoBERTa**: Multilingual transformer for text embeddings
- **BiGRU**: Bidirectional GRU for temporal context modeling
- **Attention Mechanism**: To focus on important messages in conversations
- **Multilingual Support**: Handles English and Indonesian conversations

## 📊 Dataset

**Source**: PAN-12 Sexual Predator Identification Challenge

- **Training Data**: 700 conversations (~10,882 messages)
  - Non-predators: 673 (96.1%)
  - Predators: 27 (3.9%)

- **Test Data**: 150 conversations (~1,900 messages)
  - Non-predators: 144 (96%)
  - Predators: 6 (4%)

### Data Files
```
code/
├── df_train.pth          # Original training data (English)
├── df_test.pth           # Original test data (English)
├── df_train_id.pth       # Indonesian translation of training data
├── df_test_id.pth        # Indonesian translation of test data
├── df_train_en.parquet   # Training data (parquet format)
├── df_test_en.parquet    # Test data (parquet format)
```

## 🏗️ Model Architecture

```
Input (Conversation)
    ↓
[Message Tokenization & Padding]
    ↓
XLM-RoBERTa (768-dim embeddings)
    ↓
BiGRU (hidden_size=128, bidirectional=True)
    ↓
Attention Mechanism
    ↓
Linear Classifier
    ↓
Sigmoid → Prediction
```

### Configuration
- **Model**: `xlm-roberta-base`
- **Max Sequence Length**: 64 tokens
- **Max Messages per Conversation**: 15
- **Batch Size**: 4
- **Epochs**: 5
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Loss Function**: BCEWithLogitsLoss (pos_weight=10.0)
- **Decision Threshold**: 0.3

## 📈 Results

### Test Set Performance
| Metric | Score |
|--------|-------|
| Accuracy | 0.9133 |
| Precision | 0.2941 |
| Recall | 0.8333 |
| F1-Score | 0.4348 |
| AUC-ROC | 0.9306 |

### Training History
- Best F1-Score: **0.4615** (Epoch 5)
- Final Loss: **0.4542**
- Training completed in ~22 minutes

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.12.9
torch
transformers
pandas
scikit-learn
tqdm
pyarrow
```

### Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers pandas scikit-learn tqdm pyarrow sentencepiece
```

### Running the Pipeline

#### 1. Preprocessing (if starting from XML)
```bash
jupyter notebook code/preprocessing.ipynb
```
This converts PAN-12 XML dataset to pandas DataFrames with predator labels.

#### 2. Training
```bash
jupyter notebook code/train.ipynb
```
Trains the XLM-RoBERTa BiGRU model on the dataset.
- Outputs: `best_model.pt`, `training_history.csv`

#### 3. Translation (Optional - for multilingual evaluation)
```bash
jupyter notebook code/translation.ipynb
```
Translates English conversations to Indonesian using NLLB-200 model.
- Requires: ~600MB GPU memory for model
- Duration: ~2-3 hours for full dataset

## 📁 Project Structure

```
xlm-bert_bigru/
├── README.md                           # This file
├── best_model.pt                       # Trained model weights (1.1GB)
├── code/
│   ├── preprocessing.ipynb             # Data preprocessing
│   ├── train.ipynb                     # Model training
│   ├── translation.ipynb               # Multilingual translation
│   ├── df_train.pth                    # Training data
│   ├── df_test.pth                     # Test data
│   ├── df_train_id.pth                 # Indonesian training data
│   ├── df_test_id.pth                  # Indonesian test data
│   ├── df_train_en.parquet             # Training parquet
│   ├── df_test_en.parquet              # Test parquet
│   └── training_history.csv            # Training metrics
└── pan12-sexual-predator-identification-test-and-training/
    ├── pan12-sexual-predator-identification-training-corpus-2012-05-01/
    └── pan12-sexual-predator-identification-test-corpus-2012-05-21/
```

## 🔬 Key Features

✅ **Multilingual Support**: XLM-RoBERTa handles 100+ languages  
✅ **Conversation-Level Classification**: Considers full conversation context  
✅ **Attention-Based Pooling**: Focuses on important messages  
✅ **Class Imbalance Handling**: Positive weight in BCEWithLogitsLoss  
✅ **MPS Support**: Optimized for Apple Silicon (M-chip) GPUs  
✅ **Checkpoint Mechanism**: Translation progress saved incrementally  

## 💡 Usage Example

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load model
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = torch.load("best_model.pt")
model.to(DEVICE)
model.eval()

# Prepare conversation
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
conversation_text = ["Hello", "How are you", "..."]  # List of messages

# Get prediction
with torch.no_grad():
    logits = model(input_ids, attention_mask, num_msgs)
    probability = torch.sigmoid(logits).item()
    prediction = 1 if probability >= 0.3 else 0
```

## ⚙️ Hyperparameter Tuning

Current hyperparameters achieved good F1-score. For further improvement:
- Increase `MAX_MSGS` for longer conversations
- Adjust `POS_WEIGHT` for different recall/precision trade-offs
- Modify `THRESHOLD` to optimize for your use case
- Try `num_beams > 1` for translation quality

## 🐛 Troubleshooting

**Out of Memory**: Reduce `BATCH_SIZE` or `MAX_LEN`  
**Slow Translation**: Use `facebook/nllb-200-distilled-600M` instead of full model  
**Low Precision**: Increase `POS_WEIGHT` to focus on predator class  

## 📝 Notes

- Model uses layer freezing (first 10 XLM-RoBERTa layers) for efficient fine-tuning
- Attention weights automatically handle variable-length conversations
- Decision threshold (0.3) is tuned for high recall (catching predators)

## 📚 References

- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base)
- [NLLB Translation](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [PAN-12 Dataset](https://pan.webis.de/clef12/pan12-web/index.html)

## 🤝 Contributing

Feel free to improve this project by:
- Optimizing hyperparameters
- Adding new languages
- Implementing ensemble methods
- Improving data preprocessing

---

**Last Updated**: May 2026  
**Status**: ✅ Training Complete
