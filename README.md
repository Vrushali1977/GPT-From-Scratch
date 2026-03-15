# GPT From Scratch – Inspired by Andrej Karpathy

This repository contains an implementation of a **GPT-style Transformer model built from scratch in PyTorch**, inspired by **[Andrej Karpathy](https://github.com/karpathy/ng-video-lecture)**.  
The goal is **educational**: to understand how Large Language Models (LLMs), Transformers, and GPT architectures work internally.

---

## Table of Contents
- [Project Purpose](#project-purpose)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Training](#training)  
- [Text Generation](#text-generation)  
- [Repository Structure](#repository-structure)  
- [Learning Outcomes](#learning-outcomes)  
- [Credits](#credits)  
- [Disclaimer](#disclaimer)

---

## Project Purpose
This project is **not production-ready**. It is designed to:

- Implement the **Transformer architecture from scratch**  
- Understand **self-attention and multi-head attention mechanisms**  
- Learn **tokenization, embeddings, and positional encoding**  
- Explore **training loops, loss functions, and text generation**

---

## Features
- Character-level GPT implementation in **PyTorch**  
- Multi-head self-attention and Transformer decoder blocks  
- Custom tokenization and embeddings  
- Training pipeline for a **custom dataset**  
- Text generation after training  

---

## Dataset
The model is trained on a **custom dataset** scraped from **vk.com private messages** using a personal JavaScript script.

> ⚠️ **Privacy Notice:** The dataset and trained model weights are **not included** in this repository.

---

## Installation

```bash
# Clone repository
git clone https://github.com/Vrushali1977/GPT-From-Scratch.git
cd GPT-From-Scratch

# Optional: create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
Training
python train.py --data_file path_to_dataset.txt --epochs 10 --batch_size 64

--data_file: Path to your training dataset

--epochs: Number of training epochs

--batch_size: Batch size for training

Text Generation
python generate.py --prompt "Hello world" --length 200

--prompt: Initial text to seed generation

--length: Number of tokens to generate

Repository Structure
File	Description
GPT_model.py	Transformer / GPT architecture
train.py	Training loop and optimization
generate.py	Text generation script
CSV_manipulation.py	Data preprocessing scripts
requirements.txt	Python dependencies
README.md	Project documentation
Learning Outcomes

By completing this project, you will:

Gain hands-on experience with Transformers and GPT

Understand attention mechanisms and embeddings

Learn text generation and training process

Strengthen skills for AI/ML engineering roles

Credits

Inspired by Andrej Karpathy
.
Original lecture series: YouTube

Disclaimer

This project is for educational purposes only

No personal data or trained model weights are shared

Do not use this code for commercial purposes without permission
