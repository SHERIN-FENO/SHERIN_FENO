# Employee Departure Prediction using Machine Learning

## Introduction

Employee retention is a critical issue for organizations across various industries. This project aims to predict whether an employee is likely to leave the company by leveraging machine learning techniques. By analyzing employee demographics, work metrics, engagement, and performance reviews, the project seeks to provide actionable insights that help organizations proactively manage retention strategies.

## Objective

The primary objective is to develop a reliable machine learning model that accurately classifies employees as potential "leavers" or "stayers". This prediction model can aid in identifying at-risk employees early, allowing organizations to implement targeted interventions to improve retention.

## Dataset Overview

The dataset contains 300,000 employee records with features including:

- **Identifiers**: Unique record ID, Gender (binary)
- **Experience & Work Metrics**: YearsWorked, TrainingHours, WorkLifeBalance, NumOfProjects, JobInvolvement, TeamSize, MentorshipReceived
- **Engagement & Satisfaction**: SkillDevelopmentCourses, ProjectComplexity, WorkSatisfaction, JobEngagement
- **Health & Wellbeing**: PhysicalActivityScore, MentalWellbeingScore (1-9 scale)
- **Performance Reviews**: SelfReview (3-5 scale), SupervisorReview (2-5 scale)
- **Department & Turnover**: DepartmentCode (1-7) and Left (target variable: 1 = left, 0 = stayed)

## Project Workflow

### 1. Data Loading and Initial Inspection
- **Libraries Used:** `pandas`, `numpy`, `seaborn`, `matplotlib`, and others.
- **Process:** Load the CSV file, inspect data types, and view summary statistics.

### 2. Data Cleaning
- **String-to-Numeric Conversions:** Clean fields like `Distance`, `Salary`, and `PreviousSalary`.
- **Handling Missing Values:** Impute missing values using the median.
- **Removing Redundant Columns:** Drop unnecessary columns (e.g., `Unnamed: 0`, `RecordId`) to avoid model bias.

### 3. Exploratory Data Analysis (EDA)
- **Target Analysis:** Use count plots to inspect the balance of the target variable (`Left`).
- **Feature Distribution:** Visualize numerical features using histograms.
- **Correlation Analysis:** Generate a correlation matrix to identify relationships between features.

### 4. Feature Engineering
New features are created to capture nuanced aspects of employee performance and satisfaction, such as:
- **Engagement_Satisfaction_Ratio:** Ratio of job engagement to work satisfaction.
- **Performance_Satisfaction:** Combined measure using peer feedback and work satisfaction.
- **YearsWorked_Category:** Categorizes employees as New, Experienced, or Senior.
- **Salary_Range:** Segments employees into Low, Medium, and High salary groups.
- **Overall_Satisfaction:** Aggregates work satisfaction, job engagement, and work-life balance.
- **Health_Wellbeing_Score:** Sum of physical activity and mental wellbeing scores.
- **High_Job_Involvement, Overworked,** and **Skill_Development_Activity:** Additional indicators reflecting engagement and workload.

### 5. Data Preprocessing and Splitting
- **One-Hot Encoding:** Convert categorical variables into numerical form.
- **Scaling:** Apply `StandardScaler` to numerical features.
- **Train-Test Split:** Partition data into training (70%) and testing (30%) sets with stratification to maintain class balance.

### 6. Model Building, Evaluation, and Handling Class Imbalance
- **SMOTE:** Use SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
- **Models Developed:**
  - **Logistic Regression:** Baseline model with SMOTE.
  - **Decision Tree:** Both untuned and tuned via `RandomizedSearchCV`.
  - **Random Forest:** Evaluated with untuned and hyperparameter-tuned configurations.
- **Metrics:** Accuracy, precision, recall, F1-score, and AUC-ROC are used to evaluate model performance.

### 7. Feature Importance and Model Refinement
- **Feature Importance Analysis:** Identify the top 15 features contributing to predictions.
- **Retraining:** Re-train the best model (Random Forest) using only the top features, resulting in improved model accuracy and AUC-ROC.

## Results and Conclusions

- **Tuned Decision Tree:** Achieved an AUC-ROC of 0.92 with improved recall for predicting employee departures.
- **Tuned Random Forest:** Demonstrated strong performance with an AUC-ROC of 0.89, and retraining with the top features further improved results.
- **Recommendation:** Given its interpretability and high predictive performance, the tuned decision tree model is recommended for practical use, especially in environments where model transparency is essential.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/employee-departure-prediction.git
