# gtc-ml-project1-hotel-bookings
Hotel Bookings Data Preprocessing Project
📌 Overview

This project focuses on cleaning and preprocessing the Hotel Bookings dataset to prepare it for machine learning models that predict last-minute booking cancellations.

It covers EDA, data cleaning, feature engineering, and dataset preparation, with an optional Streamlit app for interactive exploration.

📂 Project Structure
gtc_ml_project1_hotel_bookings/
│
├── gtc_ml_hotel_booking.ipynb   # Jupyter Notebook with all phases
├── hotel_bookings.csv                   # Raw dataset
├── README.md                            # Project documentation
├── data_quality_report.csv              # Data quality report
└── app.py                               # Optional Streamlit interactive app

🚀 Project Phases
🔹 Phase 1: Exploratory Data Analysis (EDA)

Load and inspect dataset structure

Generate summary statistics

Visualize missing values (heatmap)

Detect outliers (boxplots & IQR method)

🔹 Phase 2: Data Cleaning

Handle missing values (company, agent, country, children)

Remove duplicates

Cap extreme outliers (e.g., adr > 1000)

Fix data types (e.g., dates, integers)

🔹 Phase 3: Feature Engineering

Create new features:

total_guests = adults + children + babies

total_nights = weekend_nights + week_nights

is_family flag for family bookings

Encode categorical variables:

One-hot encoding (e.g., meal, market_segment)

Frequency encoding (e.g., country)

Remove data leakage columns

Split dataset into train/test sets

🛠️ How to Run

Open the notebook in Jupyter or Colab:

jupyter notebook hotel_bookings_preprocessing.ipynb


Run all cells sequentially.

(Optional) Run the Streamlit app for an interactive interface:

streamlit run app.py

📈 Outputs

X_train.csv, X_test.csv → Preprocessed feature sets

y_train.csv, y_test.csv → Target labels

Cleaned and transformed dataset ready for machine learning

✨ Future Work

Train ML models for cancellation prediction

Perform feature importance analysis

Build dashboards for business insights

👨‍💻 Author

Ahmed Mohamed
