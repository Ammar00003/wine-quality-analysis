# Wine Quality Analysis

A data science project analysing wine quality using Python, with automated visualisations and a predictive machine learning model.

## Project Overview
This project focuses on:
- **Wine Quality Prediction**: Using data from red and white wines, the goal is to predict the quality based on various physicochemical attributes.
- **Model Evaluation**: Several machine learning models are evaluated, including Random Forest and Polynomial Regression.
- **Data Visualisation**: The project includes automated plot generation to compare true vs predicted quality, and distributions of quality values.

## Key Features
- **Data Preprocessing**: Data cleaning, feature engineering, and splitting into training and test sets.
- **Machine Learning**: Implemented predictive models like Random Forest and Polynomial Regression to forecast wine quality.
- **Visualisations**: Automated plot generation:
  - True vs Predicted wine quality (scatter plots)
  - Distribution of wine quality (histogram plots)
- **Model Evaluation**: Performance metrics like R² Score, MAE, and RMSE to assess model quality.
- **Reproducibility**: Full dependency management via `requirements.txt` to ensure reproducibility.

## Technical Stack
- **Language**: Python
- **Libraries**: 
  - **Data Handling**: Pandas
  - **Visualisation**: Matplotlib, Seaborn
  - **Machine Learning**: Scikit-learn, XGBoost
- **Tools**: Git, GitHub, GitPod
  
## Sample Output
![Random Forest - Red Wine Quality Comparison](graphs/Random%20Forest%20-%20Red%20Wine%20quality%20comparison.png)  
*(Example visualisation from the analysis)*

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis (generates plots and model)
python analysis.py
```
##  License
This project is licensed under the [MIT License](LICENSE).
