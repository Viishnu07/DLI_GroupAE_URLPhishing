# DLI_GroupAE_URLPhishing

This repository contains our group project for the **Deep Learning for Intrusion Detection** module. The goal is to apply a deep learning model to detect **phishing URLs** using character- and domain-based features extracted from a real-world dataset.

The model is trained, tested, and evaluated in **Google Colab** and meets the assignment requirements including metrics, visualizations, and proper documentation.

---

## 👥 Group Members

| Name | Role |
|------|------|
|    | GUI Lead |
|  | Model Lead |
| Venus Ong Jin Wen | Data Lead |
| Viishnu Sree Ganesh | Evaluation Lead |
| Minhal Ali | Documentation Lead |

---

## 🎯 Project Objective

Detect phishing URLs using a **Multi-Layer Perceptron (MLP)** model trained on a public dataset with over **120,000+ samples**. The final evaluation uses metrics such as **F1-score**, **ROC-AUC**, and a **confusion matrix**.

---

## 📊 Evaluation Metrics

We use standard binary classification metrics:

- ✅ **Accuracy**
- ✅ **Precision**
- ✅ **Recall**
- ✅ **F1-score** (Target: ≥ 0.90)
- ✅ **ROC-AUC**
- ✅ Confusion Matrix

---

## 🛠️ Tools & Libraries

- [Google Colab](https://colab.research.google.com/)
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `tensorflow` / `keras`
- `gdown` (to download dataset from Drive, if needed)
- `streamlit` or `gradio` (optional GUI bonus)

---

## 📌 Dataset Information

- **Name**: Phishing URL Dataset
- **Size**: ~129,000 rows
- **Features**: Structural elements of URL, domain, path, etc.
- **Label**: `phishing` (0 = legitimate, 1 = phishing)

> 📎 Dataset Source: https://www.kaggle.com/datasets/michellevp/dataset-phishing-domain-detection-cybersecurity?resource=download 

---

## 🚀 Success Metric

Our final goal is to achieve:

```text
✅ F1-score ≥ 0.90 on the phishing detection task
