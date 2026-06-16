# London Airbnb Market Analysis

## Overview

This project is an end-to-end data science analysis of the London Airbnb market using the publicly available London Airbnb Open Dataset (2022).  
The goal was to explore the market, understand host behavior, and build a price prediction model.


## Key Features

- Exploratory Data Analysis: uncover trends in neighbourhoods, room types, and pricing
- Price Prediction Models: compare Linear Regression and Random Forest
- Interactive Dashboard: explore listings with maps, filters, and charts using Streamlit
- Business Insights: identify what drives listing prices and host success

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Plotly
- Streamlit
- Jupyter Notebooks

## Model Performance

After training and evaluating two models, the Linear Regression model performed as follows:

Metric - Score       

MAE - £72.87   
RMSE - £117.40     
R² Score - 0.2252    

### What Does This Score Mean?

MAE £72.87- On average, the model's price predictions are off by about £73. For London, this is a reasonable starting point. |
R² 0.2252- The model explains about 22.5% of the variation in listing prices. The other ~77.5% is influenced by factors not captured in this dataset - like reviews, amenities, property size, or seasonality. |

### Is This a "Good" Score?

Honest answer: Not yet. But that's okay.

In real-world data science, R² scores of 0.20–0.40 are common for noisy, real-world datasets like this one. Factors like:
- Number of reviews and average rating
- Superhost status
- Bedrooms and bathrooms
- Amenities (WiFi, parking, pool)
- Distance to landmarks and transport

…are all missing from this dataset, which explains the modest score.

What this score does give me:
1. A clear direction for improvement - add more features
2. A baseline to compare future models against
3. Validation that I understand the full data science workflow - from cleaning to evaluation

## What's Next?

- Add more features (e.g., reviews, amenities, distance to landmarks)
- Try XGBoost or LightGBM
- Hyperparameter tuning
- Explore geospatial clustering (e.g., create neighborhood groups)


## Project Structure
london-airbnb-analysis/
├── data/
│ └── listings.csv
├── notebooks/
│ ├── 01_EDA.ipynb
│ └── 02_Model_Training.ipynb
├── app/
│ └── app.py
├── requirements.txt
└── README.md
