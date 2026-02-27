# Drug Spending Analysis and Predictive Decision Support Using Medicaid Utilization Data

## Project Overview

Medicaid programs cover thousands of prescription drugs across different states, which leads to large and complex spending patterns. While standard reports show how much is spent, they often do not explain *why* certain drugs cost more or how spending might change in the future.

This project analyzes Medicaid State Drug Utilization Data to better understand drug-level spending, prescription behavior, and state-level variation. In addition, machine learning models are used to predict reimbursement amounts based on utilization characteristics. The overall aim is to support more informed and proactive Medicaid spending decisions.

The project focuses on:

* Identifying drugs that drive the majority of Medicaid spending
* Comparing prescription volume against total cost
* Exploring how utilization and spending differ across states
* Predicting drug spending using historical utilization data

## Business Problem

### Problem Statement

A small number of drugs account for a large share of Medicaid spending, but these cost drivers are not always easy to identify using traditional summaries. Decision-makers also lack predictive tools that can estimate future spending as utilization patterns change.

Without deeper analysis, it becomes difficult to prioritize high-impact drugs, control costs effectively, or plan future budgets.


## Project Objectives

This analysis is guided by the following questions:

1. Which drugs account for the highest total Medicaid reimbursement spending?
2. Which drugs are prescribed most frequently, and how do they contribute to overall costs?
3. How does drug utilization and reimbursement vary across states?
4. How accurately can Medicaid drug spending be predicted using drug identity and utilization features?
5. What insights from the analysis can support better Medicaid spending decisions?

## Data Source

The dataset used in this project is the **State Drug Utilization Data (SDUD)** obtained from Data.gov:

[https://catalog.data.gov/dataset/state-drug-utilization-data-2025](https://catalog.data.gov/dataset/state-drug-utilization-data-2025)

### Data Preparation Summary

* National-level aggregate rows were removed
* Drug names were standardized to reduce duplication
* Utilization and reimbursement variables were cleaned and consolidated

After preprocessing, the final dataset contains **603,682 records** at the **drug–state–quarter** level.


## Methodology Overview

### Exploratory Data Analysis

Exploratory analysis was used to understand how spending is distributed across drugs and states. The data shows a highly skewed distribution, where a small group of drugs accounts for most of the total reimbursement.

High-cost drugs were examined rather than removed, since they represent real and meaningful Medicaid expenditures. To focus on the most impactful drugs, a **cumulative spending threshold of 80–90%** was used instead of selecting an arbitrary number of drugs. This resulted in **683 unique drugs** being retained for deeper analysis.


## Modeling Approach

### Feature Engineering

The main features used for modeling include:

* Drug identity (standardized product name)
* Number of prescriptions
* Units reimbursed
* State
* Time indicators (year and quarter)

Reimbursement values were log-transformed to reduce skewness and improve model stability.

### Models Evaluated

Several models were trained and compared:

* Linear Regression (baseline)
* Random Forest Regressor
* XGBoost Regressor

Performance was evaluated using a train–test split and standard regression metrics.

### Best Model

**XGBoost** produced the strongest results overall. It handled nonlinear relationships well and provided more accurate predictions than the baseline and ensemble alternatives.


## Model Interpretability

To better understand the predictions, **SHAP values** were used to interpret the XGBoost model.

The explanations showed that:

* Prescription volume and units reimbursed are the strongest drivers of spending
* Drug identity has a significant effect on reimbursement amounts
* State-level differences also contribute to spending variation

This step helps ensure the model results are understandable and usable in a policy or budgeting context.

## Key Findings

* Medicaid drug spending is highly concentrated among a small subset of drugs
* High prescription volume does not always mean high total spending
* Chronic and specialty medications dominate total reimbursement
* Spending patterns differ across states, mainly due to utilization scale


## Business Recommendations

* Focus cost-control efforts on drugs within the top cumulative spending range
* Use utilization trends to anticipate future high-cost drugs
* Apply explainable models to support transparent decision-making
* Compare state-level patterns to identify potential policy improvements


## Conclusion

This project shows that combining exploratory analysis with predictive modeling can provide meaningful insights into Medicaid drug spending. By focusing on high-impact drugs and using interpretable machine learning models, Medicaid programs can move beyond static reporting toward more data-driven planning.

## Next Steps

* Extend the analysis to include additional years of data
* Explore time-series forecasting for future spending
* Develop a simple dashboard for ongoing monitoring
