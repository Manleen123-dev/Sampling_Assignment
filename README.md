# Sampling Assignment – Handling Imbalanced Credit Card Data

> **Course Assignment | Clean, Structured, and Visualization-Rich Submission**

---

## 📌 Objective
To understand the importance of **sampling techniques** in handling **imbalanced datasets** and analyze how different sampling strategies affect the performance of multiple **machine learning models**.

---

## 📊 Dataset Overview
The dataset consists of credit card transactions with a **severely imbalanced target variable**, where fraudulent transactions form a very small minority.

### Class Distribution (Before Sampling)
![Class Distribution Before Sampling](results/class_distribution_before.png)

### Class Distribution After Random Over Sampling
![Class Distribution After ROS](results/class_distribution_after_ros.png)

### Class Distribution After SMOTE
![Class Distribution After SMOTE](results/class_distribution_after_smote.png)

---

## 🔁 Overall Workflow (Flow Diagram)

```mermaid
flowchart TD
    A[Load Dataset] --> B[Check Class Imbalance]
    B --> C[Apply Sampling Techniques]
    C --> D[Create 5 Balanced Samples]
    D --> E[Train 5 ML Models]
    E --> F[Evaluate Using Accuracy]
    F --> G[Generate Tables & Plots]
    G --> H[Analysis & Conclusion]
```

---

## ⚙️ Sampling Techniques Used

| Sampling ID | Technique |
|------------|----------|
| Sampling1 | Random Over Sampling (ROS) |
| Sampling2 | Random Under Sampling (RUS) |
| Sampling3 | SMOTE |
| Sampling4 | NearMiss |
| Sampling5 | Stratified Sampling |

---

## 🤖 Machine Learning Models

| Model ID | Model |
|--------|-------|
| M1 | Logistic Regression |
| M2 | Decision Tree |
| M3 | Random Forest |
| M4 | Support Vector Machine (SVM) |
| M5 | K-Nearest Neighbors (KNN) |

---

## 📈 Accuracy Comparison

### Accuracy Heatmap (Models vs Sampling Techniques)
![Accuracy Heatmap](results/accuracy_heatmap.png)

### Accuracy Matrix Table
![Accuracy Matrix Table](results/accuracy_matrix_table.png)

---

## 📊 Detailed Comparative Analysis

### Model-wise Accuracy Comparison
![Model-wise Accuracy Table](results/model_wise_accuracy_table.png)

### Accuracy Trend Across Sampling Techniques
![Accuracy Trend](results/accuracy_trend_across_sampling.png)

---

## 📦 Distribution Analysis

### Accuracy Distribution per Sampling Technique
![Accuracy Boxplot Sampling](results/accuracy_boxplot_sampling.png)

### Accuracy Distribution per Model
![Accuracy Boxplot Model](results/accuracy_boxplot_model.png)

---

## 📉 Mean Accuracy Analysis

### Mean Accuracy per Sampling Technique
![Sampling Mean Accuracy](results/sampling_mean_accuracy.png)

### Mean Accuracy per Model
![Model Mean Accuracy](results/model_mean_accuracy.png)

---

## 🏆 Best Sampling Technique per Model
The best-performing sampling technique for each model is stored in:

- `results/best_sampling_per_model.csv`

This directly answers the comparison requirement of the assignment.

---

## 🧠 Key Observations
- Oversampling techniques (ROS, SMOTE) generally improve performance.
- Under-sampling methods may cause information loss on small datasets.
- Tree-based models can overfit when datasets are very small.
- No single sampling technique is optimal for all models.

---

## 📁 Repository Structure

```
Sampling_Assignment/
├── data/
│   └── Creditcard_data.csv
├── src/
│   └── sampling_assignment.py
├── results/
│   ├── *.png
│   ├── *.csv
└── README.md
```

---

## ✅ Conclusion
This assignment demonstrates that **sampling is critical** when dealing with imbalanced datasets. Through structured experimentation, tables, charts, and visual analysis, the impact of different sampling techniques on model performance is clearly illustrated.

---

## ✍️ Author
**Manleen Kaur**
