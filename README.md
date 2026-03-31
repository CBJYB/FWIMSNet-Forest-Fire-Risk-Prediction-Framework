# Author:CBJ
# FWI-MSNet: Forest Fire Risk Prediction Framework

## Project Overview
This dataset needs to be obtained from a Chinese website, and the retrieved data information is in Chinese.
Data Source：https://www.nesdc.org.cn/sdo/list?searchKey=%E4%BC%9A%E5%90%8C%E7%AB%99
Dataset：
      1、会同站2005-2022年大气环境要素观测数据集：
        A Long-Term Monitoring Dataset of Meteorological Indicators at National Research Station of Huitong Forest Ecosystems(2005-2022)
      2、会同站2005-2022年水分要素观测数据集：
        A Long-Term Monitoring Dataset of Water Environment Indicators at National Research Station of Huitong Forest Ecosystems(2005-2022)
      3、会同站2005-2022年土壤要素观测数据集：
        A Long-Term Monitoring Dataset of Soil Indicators at National Research Station of Huitong Forest Ecosystems(2005-2022)
      
## Model Description:
1.  FWI-MSNet is a physics-guided deep learning framework for forest fire risk prediction. By coupling multiple fuel and climate factors, this model achieves high-precision, interpretable forest fire danger forecasting.
2.  Due to source data infringement issues, this project only provides demo data that has undergone data preprocessing and feature engineering to demonstrate the model.

## Model Usage Steps：
1.Data Preprocessing: Standardization, Data Augmentation(preprocess_data.py)
2.Construction of multi-scale input features（preprocess_data.py, models.py, train.py, predict.py)
3.Model Construction(models.py)：
    (1)Multi-scale Feature Extraction Module(1D-CNN)
    (2)GRU Transformer Collaborative Working Mechanism(GRU-Transformer)
    (3)Output Layer Construction
4.Model Training and Evaluation(train.py)
5.Model Prediction and Evaluation(predict.py)

## Specific operation steps:
1.preprocess_data.py
2.train.py
3.predict.py

## Module Description
- models: Deep Learning Model Definition（FWI-MSNet）
- utils: Utility Functions (Random Seed, FWI Level, Evaluation Metrics, etc.)
- train: Model Training Script
- predict: Model Prediction Script
- main: Data Visualization Script

## Paper
For specific research details, please refer to the paper:
If you use any content or information from this framework, please cite the paper: