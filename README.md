# DLI_GroupAE_URLPhishing

This repository contains our group project for the **Deep Learning for Intrusion Detection** module. The goal is to apply a deep learning model to detect **phishing URLs** using character- and domain-based features extracted from a real-world dataset.

The model is trained, tested, and evaluated in **Google Colab** and meets the assignment requirements including metrics, visualizations, and proper documentation.

---

## 👥 Group Members

| Name | Role |
|------|------|
|    | GUI Lead |
| Loo Shu Hinn  | Model Lead |
| Venus Ong Jin Wen | Data Lead |
| Viishnu Sree Ganesh | Evaluation Lead |
| Minhal Ali | Documentation Lead |

---

## 🎯 Project Objective

Detect phishing URLs using a **Multi-Layer Perceptron (MLP)** model trained on a public dataset with over **88,647+ samples**. The final evaluation uses metrics such as **F1-score**, **ROC-AUC**, and a **confusion matrix**.

---

## 📊 Evaluation Metrics

We use standard binary classification metrics for phishing detection:

- ✅ **Accuracy**
- ✅ **Precision**
- ✅ **Recall**
- ✅ **F1-score** (🎯 Target: ≥ 0.90)
- ✅ **ROC-AUC**
- ✅ **Confusion Matrix**
- ✅ **ROC Curve**

---

## 🛠️ Tools & Libraries

Our project is implemented using the following tools and Python libraries:

- 🧠 Google Colab (free GPU/CPU)
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `tensorflow` / `keras`
- `gdown` (for dataset download, optional)
- `streamlit` or `gradio` (for GUI – bonus)

---

## 📌 Dataset Information

- **Name**: Phishing Website Dataset (Mendeley)
- **Source**: Nayak, G.S. (2023). _Detection of Phishing Websites Using Machine Learning and Deep Learning Techniques_. [Mendeley Data](https://data.mendeley.com/datasets/72ptz43s9v)
- **Size**: 88,647 rows, 112 columns
- **Features**: Numerical attributes representing URL structure, domain characteristics, and protocol flags
- **Label**: `phishing` —  
  - `0` = Legitimate  
  - `1` = Phishing

---

## 🚀 Success Metric

> 🎯 **Achieve F1-score ≥ 0.90** on the phishing detection task using a deep learning model (MLP).
>
> The model is trained in Google Colab and validated using metrics including F1-score, ROC-AUC, and confusion matrix.
>
> Example output:  
> `Achieved F1 = 0.91, target met ✅`
