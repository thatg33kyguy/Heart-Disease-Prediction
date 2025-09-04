```markdown
# Heart Disease Prediction using Machine Learning  
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/thatg33kyguy/Heart-Disease-Prediction)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thatg33kyguy/Heart-Disease-Prediction/blob/main/Heart.ipynb)  

This project develops a machine learning model to predict the likelihood of heart disease based on clinical attributes such as age, cholesterol, blood pressure, and exercise patterns.  
It demonstrates the complete ML pipeline: data preprocessing, exploratory analysis, model training, and performance evaluation.  

---

## Features  
- Preprocessed and cleaned the **UCI Heart Disease dataset**.  
- Conducted exploratory data analysis (EDA) with visualizations including histograms, scatter plots, and correlation heatmaps.  
- Applied one-hot encoding for categorical variables and standardized continuous features.  
- Implemented a Logistic Regression classifier for binary classification (disease vs. no disease).  
- Evaluated performance using accuracy, precision, recall, F1-score, and confusion matrix.  

---

## Tech Stack  
- **Languages/Tools:** Python, Jupyter Notebook  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  

---

## Workflow  
1. **Data Preprocessing** – categorical encoding and feature scaling.  
2. **EDA** – analysis of correlations between health metrics and heart disease.  
3. **Model Training** – Logistic Regression applied with a 70/30 train-test split.  
4. **Evaluation** – performance measured with classification metrics and confusion matrix.  

---

## Project Structure  
```

Heart-Disease-Prediction/
│── Heart.ipynb                # Main Jupyter Notebook
│── heart.csv                  # Dataset (UCI Heart Disease)
│── requirements.txt            # Dependencies
│── README.md                   # Documentation

```

---

## Results  
- **Training Accuracy:** 86.8%  
- **Testing Accuracy:** 86.8%  
- Key predictors identified include maximum heart rate, cholesterol, blood pressure, and exercise-induced angina.  

---

## Future Work  
- Compare performance with other algorithms such as Random Forest, SVM, and Gradient Boosting.  
- Apply hyperparameter tuning for improved accuracy.  
- Deploy as a web application using Flask or Streamlit.  
```
