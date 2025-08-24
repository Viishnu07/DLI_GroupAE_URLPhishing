# 🔐 DLI_GroupAE_URLPhishing  

This repository contains our group project. Our objective was to design and evaluate deep learning models for detecting **phishing URLs**, leveraging both character- and domain-based features from real-world datasets.  

The final system adopts a **Convolutional Neural Network (CNN)**, tuned with optimized thresholds for balanced **precision and recall**, achieving **F1 = 0.991** and **ROC-AUC = 0.9996** on the test set. These results demonstrate both robustness and efficiency, aligning with recent research such as **DEPHIDES (2024)**, which found CNNs highly effective for phishing URL detection:contentReference[oaicite:2]{index=2}.  

---

## 👥 Group Members  

| Name | Role |
|------|------|
| Loo Shu Hinn & Chan Yong Kang  | Model Lead |
| Venus Ong Jin Wen              | Data Lead |
| Viishnu Sree Ganesh            | Evaluation Lead |
| Minhal Ali                     | Documentation Lead |

---

## 🎯 Project Objective  

The objective of this project is to detect **phishing websites** using **deep learning architectures** trained on large-scale URL datasets. Phishing attacks remain one of the most common cybersecurity threats, often relying on deceptive URLs to steal sensitive information. Inspired by recent work such as **DEPHIDES (2024)**, which showed that CNNs are particularly effective for URL-based phishing detection, our project explores multiple models (MLP, CNN, RNN) and evaluates them against common success metrics.  

Our final chosen model — a **threshold-tuned CNN** — balances **precision and recall**, enabling accurate and efficient detection of phishing URLs in real-time conditions.  

---

## 📊 Evaluation Metrics  

We assess performance using:  

- ✅ **Accuracy**  
- ✅ **Precision**  
- ✅ **Recall**  
- ✅ **F1-score** (🎯 Target ≥ 0.90)  
- ✅ **ROC-AUC**  
- ✅ **Confusion Matrix**  
- ✅ **ROC Curve**  

**Final Tuned CNN Results (Threshold = 0.475):**  
- **Accuracy:** 0.992  
- **Precision:** 0.9923  
- **Recall:** 0.9896  
- **F1-score:** 0.991  
- **ROC-AUC:** 0.9996  
- **Inference speed:** 0.081 ms/sample (~12,400 samples/s)  

---

## 🛠️ Tools & Libraries  

- 🧠 Google Colab (GPU runtime)  
- `pandas`, `numpy`  
- `scikit-learn`  
- `matplotlib`, `seaborn`  
- `tensorflow` / `keras`  
- `streamlit` or `gradio` (for GUI – bonus)  

---

## 📌 Dataset Information  

- **Extended Reference (DEPHIDES, 2024):**  
  - ~5.1 million URLs (2.3M phishing from **PhishTank**, 2.8M legitimate from **Common Crawl**):contentReference[oaicite:3]{index=3}  
  - Balanced and diverse, designed to improve generalization and handle **zero-day phishing attacks**  
  - Vectorized using **character-based embeddings** (language-independent) for broader adaptability  

---

## 🚀 Success Metrics  

> 🎯 Achieve **F1 ≥ 0.90** with a deep learning model.  
>
> 🎯 Achieve **F1 Final CNN achieved F1 = 0.991**, demonstrating a balance of high precision and recall, with near-perfect ROC-AUC.  

---

📖 **Reference for background research:**  
O. K. Sahingoz, E. Buber, E. Kugu, *DEPHIDES: Deep Learning Based Phishing Detection System*, *IEEE Access*, 2024.  
DOI: [10.1109/ACCESS.2024.3352629](https://doi.org/10.1109/ACCESS.2024.3352629)  

---
