# Fine-Tuning Phi-2 for Abstractive Summarization on XSum

Colab Link
**https://colab.research.google.com/drive/1mVxf5UX8cxmxvwRubKgmvND1HFrDU1U9?usp=sharing**


## Group Information
- **Group**: Group 9  
- **Course**: Deep Learning  
- **Institution**: Telkom University  

## Team Members

| Name | NIM |
|------|-----|
| **Fuji Aqbal Fadhlillah** | 1103223151 |
| **Ali Fatta Maulana** | 1103223228 |

---

## 📥 Pre-trained Model Availability

The fine-tuned model generated in this project can be saved locally after training.  
Due to repository size limitations, model weights are not directly included in this repository.  
Inference can be performed immediately after training or by loading the saved model directory.

---

## 📔 Table of Contents
- Purpose
- Project Overview
- Dataset Preparation
- Model Configuration
- Training Details
- Inference and Usage
- Results and Observations
- Repository Structure
- Installation & Setup
- Conclusion

---

## 🎯 Purpose

This project focuses on fine-tuning the **Phi-2 decoder-only Large Language Model (LLM)** for **abstractive text summarization** using the **XSum dataset**. The main objective is to train the model to generate concise, abstractive summaries of news articles using instruction-style prompting and causal language modeling.

---

## 🔍 Project Overview

### Abstractive Summarization

Abstractive summarization requires the model to understand the semantic meaning of a document and generate a new sentence that captures its key information, rather than extracting sentences directly from the text.

### Phi-2 Model

**Microsoft Phi-2** is a 2.7B-parameter decoder-only Transformer model known for strong reasoning and instruction-following capabilities despite its relatively small size compared to larger LLMs.

---

## 📊 Dataset Preparation

### Dataset Source

The **XSum (Extreme Summarization)** dataset contains BBC news articles paired with single-sentence abstractive summaries. To avoid compatibility issues with deprecated dataset scripts, the dataset is loaded from **Parquet files**.

### Instruction-Style Prompting

Each training sample is converted into an instruction-based prompt:

```
### Instruction:
Summarize the following article in one concise sentence.

### Article:
[Document text]

### Summary:
[Reference summary]
```

This format explicitly guides the model toward the summarization task.

---

## 🧠 Tokenization and Labeling

- Tokenization uses the tokenizer from `microsoft/phi-2`.
- The end-of-sequence token is reused as the padding token.
- All sequences are truncated and padded to a fixed maximum length.
- Labels are created by copying `input_ids` for causal language modeling.

---

## ⚙️ Model Configuration

The Phi-2 model is loaded using half-precision weights for memory efficiency. To accommodate limited GPU resources, a **partial fine-tuning strategy** is applied:

- All base model parameters are frozen.
- Only the `lm_head` parameters are updated during training.

This strategy significantly reduces memory usage while still allowing task adaptation.

---

## 🏋️ Training Details

- Training is performed on a small subset of the XSum dataset.
- Batch size: 1
- Number of epochs: 1
- Learning rate: 5e-4
- Optimization handled by Hugging Face `Trainer`.

This configuration enables successful training within constrained hardware environments such as Google Colab.

---

## 🚀 Inference and Usage

After training, the model can generate summaries using an instruction-based prompt. Text generation is controlled using parameters such as maximum new tokens, temperature, and nucleus sampling (`top-p`).

Inference can be performed either immediately after training or by loading a previously saved model.

---

## 📈 Results and Observations

During training, loss values may become zero or NaN in later steps due to numerical instability, limited data size, or learning rate sensitivity. Despite this, the end-to-end training and inference pipeline executes successfully, and the model is capable of generating abstractive summaries.

---

## 📁 Repository Structure

```
finetuning-phi-2-xsum/
│
├── Task_3_–_Fine_Tuning_Decoder_Only_LLM_(Phi_2)_for_Abstractive_Summarization_(XSum).ipynb
├── README.md
├── requirements.txt
└── final-phi2-xsum
    └── added_tokens.json
    └── config.json
    └── generation_config.json
    └── merges.txt
    └── model.safetensors.index.json
    └── special_tokens_map.json
    └──tokenizer.json
    └──tokenizer_config.json
    └── training_args.bin
    └── vocab.json
```

---

## 🛠️ Installation & Setup

### Requirements

- Python 3.8 or higher
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

Dependencies can be installed using:

```bash
pip install -r requirements.txt
```

---

## ✅ Conclusion

This project demonstrates that a decoder-only language model such as **Phi-2** can be adapted for abstractive summarization using instruction-style prompting and causal language modeling. By freezing most model parameters and training only the output head, fine-tuning becomes feasible even with limited computational resources.

