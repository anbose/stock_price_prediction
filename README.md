# Advanced Time Series Forecasting for Stock Prices

## Description

This repository contains the code for a personal project exploring and comparing various time series forecasting techniques for stock price prediction. The project investigates classical methods, standard deep learning models, including the exploration of advanced neural network architectures.

The primary goal was to gain hands-on experience with diverse forecasting methodologies and evaluate their suitability for complex financial time series data. The original development and experimentation were primarily conducted on the Kaggle platform, and this repository serves as a consolidated codebase.

## Models Implemented

The following models were implemented and evaluated:

* **Classical Statistical Models:**
    * ARIMA (Autoregressive Integrated Moving Average)
* **Deep Learning Models:**
    * LSTM (Long Short-Term Memory)
    * LSTM with Attention Mechanism
    * GRU (Gated Recurrent Unit)
    * GRU with Attention Mechanism
* **Ongoing : Advanced/Exploratory Neural Networks:**
    * KAN (Kolmogorov-Arnold Networks)
 
## Dataset
The project utilizes historical stock price data for google, from 2004 to 2022.

* **Source:** dataset on Kaggle (link: https://www.kaggle.com/datasets/varpit94/google-stock-data)
* **Features:** [e.g. , Daily Open, High, Low, Closing prices with Volume]
* **Preprocessing:** Data preprocessing steps, including scaling (e.g., MinMaxScaler) and windowing for sequence models, are detailed within the notebooks/scripts.

## Technologies Used

* **Programming Language:** Python 3.x
* **Core Libraries:** Pandas, NumPy, Scikit-learn
* **Deep Learning Framework:** TensorFlow / Keras and PyTorch
* **Statistical Modeling:** `statsmodels`
* **Plotting:** Matplotlib, Seaborn
* **Libaries:** `pmdarima` (for ARIMA), `pykan` (for KAN)
* **Environment:** Jupyter Notebooks

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/anbose/stock_price_prediction.git
    cd stock_price_prediction
    ```

## Usage

The project code is organized into Jupyter Notebooks.

* **Notebooks:** Navigate the repository and open the `.ipynb` files using Jupyter Lab, Jupyter Notebook, VS Code, or Google Colab. Run the cells sequentially to reproduce the analysis, training, and evaluation for each model.

**Note:** The file paths for data loading might need adjustment if running outside the original Kaggle environment structure. In that case, it is advisable to download the dataset separately into the working directory.

## Future Work

* Further hyperparameter tuning for all models.
* Deployment of the best performing model as a simple API.
* Explore multivariate forecasting incorporating additional features.

## Acknowledgements

* Data sourced from [https://www.kaggle.com/datasets/varpit94/google-stock-data].

## Contact

* https://www.linkedin.com/in/anbose/
