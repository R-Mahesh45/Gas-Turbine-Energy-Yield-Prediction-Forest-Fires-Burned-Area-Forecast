# Gas Turbine Energy Yield Prediction & Forest Fires Burned Area Forecast  

This repository contains two predictive modeling projects leveraging Neural Networks.  
1. **Gas Turbine Energy Yield Prediction:** Predicting turbine energy yield (TEY) using ambient variables as features.  
2. **Forest Fires Burned Area Forecast:** Predicting burned areas of forest fires using neural networks.  

## Table of Contents  
- [Objective](#objective)  
- [Datasets](#datasets)  
- [Methodology](#methodology)  
- [Models](#models)  
- [Results](#results)  
- [Technologies Used](#technologies-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [License](#license)  

---

## Objective  
- **Gas Turbines:** Predict turbine energy yield (TEY) using ambient and gas turbine parameters.  
- **Forest Fires:** Predict the burned area of forest fires based on environmental and fire-related parameters.  

---

## Datasets  
- **Gas Turbines:** Contains 36,733 instances of 11 sensor measurements aggregated hourly. Key variables include ambient temperature, pressure, humidity, and turbine parameters.  
- **Forest Fires:** Environmental factors and fire-related data to forecast burned area.  

---

## Methodology  
1. Data Preprocessing: Standardized datasets and removed inconsistencies using `pandas` and `numpy`.  
2. Neural Network Modeling: Constructed models with multiple layers using `keras` and evaluated with RMSE and accuracy metrics.  
3. Performance Evaluation: Compared models using RMSE for Gas Turbines and Accuracy for Forest Fires.  

---

## Models  
### Gas Turbines:  
- **Model Architecture:**  
  ```python  
  model = Sequential()  
  model.add(Dense(10, input_dim=10, activation='relu'))  
  model.add(Dense(5, activation='relu'))  
  model.add(Dense(4, activation='relu'))  
  model.add(Dense(1, activation='sigmoid'))  
  ```  
- **Compilation & Evaluation:**  
  ```python  
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse'])  
  ```  
  - **RMSE:** 17989.7305  
  - **MSE (%):** 1798973.05%  

### Forest Fires:  
- **Model Architecture:**  
  ```python  
  model = Sequential()  
  model.add(Dense(20, input_dim=28, activation='relu'))  
  model.add(Dense(10, activation='relu'))  
  model.add(Dense(10, activation='relu'))  
  model.add(Dense(1, activation='sigmoid'))  
  ```  
- **Compilation & Evaluation:**  
  ```python  
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
  ```  
  - **Accuracy:** 99.03%  

---

## Results  
- **Gas Turbines:** Achieved RMSE of 17989.7305, demonstrating the model's ability to predict turbine energy yield.  
- **Forest Fires:** Achieved 99.03% accuracy in predicting burned areas of forest fires.  

---

## Technologies Used  
- **Programming Languages:** Python  
- **Libraries:**  
  - `keras` (Deep Learning)  
  - `numpy` (Numerical Computing)  
  - `pandas` (Data Manipulation)  
  - `scikit-learn` (Model Evaluation)  

---

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/R-Mahesh45/project-name.git  
   ```  
2. Install required libraries:  
   ```bash  
   pip install -r requirements.txt  
   ```  

---

## Usage  
1. Open the respective scripts for Gas Turbines or Forest Fires.  
2. Run the scripts to preprocess data, build the model, and evaluate performance.  
