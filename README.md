# The Influence of Weather and Road Conditions on Road Accident Severity

## Overview
The project involves thorough exploratory data analysis, data preparation including cleaning, outlier removal, data encoder, followed by the development of robust machine learning models (Random Forest, XGBoost, L-GBM, MLP). By systematically exploring the interplay between weather, road conditions, and accident severity, this project aims to address this gap by assessing the severity of road accidents across various classes, including fatal, hospitalization, medical treatment, minor injury, and property damage, attributing them to weather and road conditions. Additionally, it aims to provide valuable insights for enhancing road safety measures and reducing the toll of road accidents on public health and infrastructure.

### Note: This project includes many experiments in .py and .ipynb files to achieve the best outcome.

## Dataset 
https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads/resource/e88943c0-5968-4972-a15f-38e120d72ec0.<br>
The dataset consists of 380316 records with 53 columns, one of the columns, fatal, hospitalisation, medical treatment, minor injury, and property damage only is considered as a target, imbalanced classes, for prediction. Imbalanced classes are accounted for 1.53% fatal, 13.70% minor injury, 22.99% property damage only, 30.51% hospitalisation, and 31.27% medical treatment. Furthermore, this dataset covers attributes such as atmospheric conditions, lighting conditions, and road conditions. These attributes consist of which can be the main features for predicting severity classes. 

## Prerequisites
### **1. Required Software**
- **Python 3.x** ([Download Here](https://www.python.org/downloads/))
- **Scikit-learn** (For machine learning models, evaluation, and preprocessing)
- **imbalanced-learn (imblearn)** (For handling class imbalance with SMOTE, SMOTEENN, etc.)
- **XGBoost** (For gradient boosting classification)
- **LightGBM** (For efficient gradient boosting models)
- **Matplotlib & Seaborn** (For data visualization)
- **NumPy & Pandas** (For data manipulation and numerical operations)
- **Pickle** (For saving and loading trained models)
- **Custom Modules**:
  - `Evaluation.py` and `Preparation.py` (Ensure these are located in your project path)

## Key Features
- Data preparation: Outlier Removal, Rare Category Removal, Grouping Data, Encoder, and Class combination
- Sampling methods: SMOTE, Undersampling, SMOTEENN, and SMOTETomek
- Model evaluation
- Feature importance & SHAP analysis 

### Model used
- Random Forest
- XGBoost(eXtream Gradient Boosting)
- L-GBM(Light Gradient Boosting Machine)
- MLP(Multilayer Perceptron)

## Result
### Outlier Removal 
<div align="center"><img width="500" alt="image" src="https://github.com/user-attachments/assets/4ba34974-44c6-4f25-98eb-bd18e68e2974" />
</div>
<div align="center"><b>Figure 1: </b>Comparison of the data distribution before and after removing outliers. (a) shows the boxplot
with outliers. (b) shows the boxplot without outliers</div>

### Encoder
I transformed some columns using one-hot encoding for classes and label encoding for features.
<div align="center"><img width="250" alt="image" src="https://github.com/user-attachments/assets/f5e93e63-4aae-44e0-912d-711c11517763" />
</div>
<div align="center"><b>Figure 2: </b>Samples of classes after label encoding</div>

### Sampling Methods
To address the imbalance class issue, I did experiment with different sampling methods like SMOTE, Undersampling, SMOTEENN, and SMOTETomek to acheive best result.

### Model Evaluation
<div align="center"><img width="450" alt="image" src="https://github.com/user-attachments/assets/466f1fb4-2244-4d06-8481-0a8078484f9e" />
</div>
<div align="center"><b>Figure 2: </b>Classification report for models with SMOTE</div>
<div align="center"><img width="800" alt="image" src="https://github.com/user-attachments/assets/fc9f2804-2ee0-4471-8597-2e78a89c7efa" />
</div>
<div align="center"><b>Figure 3: </b>Confusion matrix for Random Forest, LGBM, and XGBoost models with SMOTE applied to
address class imbalance.</div>
<div align="center"><img width="350" alt="image" src="https://github.com/user-attachments/assets/cc2b328b-4628-4dd4-8bc6-06c1910c3427" />
</div>
<div align="center"><b>Figure 4: </b>Classification report for MLP</div>
<div align="center"><img width="600" alt="image" src="https://github.com/user-attachments/assets/975478d4-3da7-42bf-b010-99f61a2629fc" />
</div>
<div align="center"><b>Figure 5: </b>Comparison of training/validation loss and the confusion matrix for the best MLP model config-
uration with 1e-6 learning rate, 512 batch size, and 30 epochs</div>

### Feauture Importance & SHAP Analysis
<div align="center"><img width="500" alt="image" src="https://github.com/user-attachments/assets/6f7a0c5b-031f-4c1c-b3f7-4b1d4973efde" />
</div>
<div align="center"><b>Figure 6: </b>The Top 20 Feature Importance for L-GBM</div>
<div align="center"><img width="405" alt="image" src="https://github.com/user-attachments/assets/f6830bd3-4d0e-4c3a-8396-9a3ae4ab855a" />
</div>
<div align="center"><b>Figure 7: </b>SHAP values for Fatal Injury Class </div>

## My Contributions
- Processed grayscale RoIs and preprocessed WSI tiles
- Implemented grayscale U-Net model
- Built performance metric visualizations
- Developed custom dataset class and wrapper
- Created confusion matrix for evaluation
