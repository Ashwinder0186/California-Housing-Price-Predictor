# 🏠 California Housing Price Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![ML Pipeline](https://img.shields.io/badge/ML-Pipeline-purple.svg)

**A production-ready machine learning pipeline for predicting California housing prices using Random Forest Regression with automated preprocessing and model persistence**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Pipeline](#ml-pipeline) • [Results](#results)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Technical Details](#technical-details)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## 🎯 Overview

The **California Housing Price Predictor** is a comprehensive machine learning project that predicts median house values in California districts using various geographic and demographic features. Built with Scikit-Learn and featuring a complete ML pipeline, this project demonstrates best practices in data preprocessing, model training, and production deployment.

### Key Highlights
- 🎯 **Random Forest Regressor** for accurate predictions
- 🔄 **Automated ML Pipeline** with preprocessing and scaling
- 📊 **Stratified Sampling** for representative train-test splits
- 💾 **Model Persistence** using Joblib for fast inference
- 🛠️ **Production-Ready** code with clean architecture
- 📈 **Cross-Validation** for robust model evaluation

This project showcases:
- **Machine Learning Engineering**: End-to-end ML pipeline development
- **Data Preprocessing**: Handling missing values, scaling, encoding
- **Model Deployment**: Serialization and inference optimization
- **Software Engineering**: Clean, maintainable, production-ready code

---

## ✨ Features

### 🤖 Machine Learning Pipeline
- **Automated Preprocessing**: 
  - Missing value imputation with median strategy
  - Feature scaling using StandardScaler
  - One-hot encoding for categorical variables
- **Stratified Sampling**: Income-based stratification for representative splits
- **Model Persistence**: Trained models saved with Joblib
- **Batch Prediction**: Efficient inference on multiple samples

### 🔧 Technical Features
- **Column Transformer**: Separate pipelines for numerical and categorical features
- **Handle Unknown Categories**: Graceful handling of unseen categorical values
- **Reproducible Results**: Random state fixed for consistent outputs
- **Efficient Storage**: Pickle serialization for model and pipeline

### 📊 Data Processing
- **Feature Engineering**: Income category creation for stratified splits
- **Data Validation**: Automated checks for data quality
- **Flexible Input**: Accepts new data in CSV format
- **Output Generation**: Predictions saved to CSV with original features

---

## 🛠️ Tech Stack

### Core Libraries
- **Machine Learning**: Scikit-Learn 1.3+
- **Data Processing**: Pandas 2.0+, NumPy 1.24+
- **Model Persistence**: Joblib 1.3+
- **Language**: Python 3.8+

### ML Components
- **Preprocessing**: SimpleImputer, StandardScaler, OneHotEncoder
- **Model**: RandomForestRegressor
- **Pipeline**: Pipeline, ColumnTransformer
- **Evaluation**: RMSE, Cross-Validation
- **Sampling**: StratifiedShuffleSplit

---

## 🏗️ ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA (housing.csv)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Stratified Train-Test Split (80-20)             │
│           Based on Income Categories (5 strata)              │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌────────────────────┐          ┌────────────────────┐
│   Training Set     │          │    Test Set        │
│   (80% of data)    │          │  (saved to CSV)    │
└─────────┬──────────┘          └────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                        │
│  • Separate median_house_value (target)                     │
│  • Identify numerical features (8 features)                 │
│  • Identify categorical features (ocean_proximity)          │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌──────────────────────┐      ┌──────────────────────┐
│  Numerical Pipeline  │      │ Categorical Pipeline │
│  • Impute (median)   │      │  • OneHotEncode      │
│  • StandardScaler    │      │  • Handle unknown    │
└──────────┬───────────┘      └──────────┬───────────┘
           │                             │
           └──────────────┬──────────────┘
                          │
                          ▼
           ┌───────────────────────────┐
           │   Transformed Features    │
           │   (Ready for ML Model)    │
           └─────────────┬─────────────┘
                         │
                         ▼
           ┌───────────────────────────┐
           │  Random Forest Regressor  │
           │   • 100 estimators        │
           │   • Random state: 42      │
           └─────────────┬─────────────┘
                         │
                         ▼
           ┌───────────────────────────┐
           │   Trained Model + Pipeline│
           │   • model.pkl             │
           │   • pipeline.pkl          │
           └─────────────┬─────────────┘
                         │
                         ▼
           ┌───────────────────────────┐
           │      INFERENCE MODE       │
           │  • Load model & pipeline  │
           │  • Transform new data     │
           │  • Predict prices         │
           │  • Save to output.csv     │
           └───────────────────────────┘
```

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ashwinder0186/California-Housing-Price-Predictor.git
cd California-Housing-Price-Predictor
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
The California Housing dataset is included. If needed, download from:
```bash
# Dataset is already included as housing.csv
# Or download from scikit-learn
python -c "from sklearn.datasets import fetch_california_housing; fetch_california_housing()"
```

---

## 🚀 Usage

### Training Mode (First Run)

When `model.pkl` doesn't exist, the script automatically trains a new model:

```bash
python house_price_with_pipeline.py
```

**What happens:**
1. Loads `housing.csv`
2. Creates stratified train-test split
3. Saves test set to `input.csv`
4. Builds preprocessing pipeline
5. Trains Random Forest model
6. Saves `model.pkl` and `pipeline.pkl`

**Output:**
```
Model successfully trained
```

### Inference Mode (Subsequent Runs)

When models exist, the script performs batch prediction:

```bash
python house_price_with_pipeline.py
```

**What happens:**
1. Loads pre-trained model and pipeline
2. Reads `input.csv`
3. Applies preprocessing transformations
4. Makes predictions
5. Saves results to `output.csv`

**Output:**
```
Inference completed, results saved to output.csv
```

### Custom Predictions

To predict on your own data:

1. **Prepare your data** in CSV format with these columns:
   ```
   longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
   population, households, median_income, ocean_proximity
   ```

2. **Save as `input.csv`**

3. **Run inference:**
   ```bash
   python house_price_with_pipeline.py
   ```

4. **Check `output.csv`** for predictions

---

## 📁 Project Structure

```
California-Housing-Price-Predictor/
│
├── house_price_with_pipeline.py   # Main script (train & inference)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── housing.csv                     # Original dataset
├── input.csv                       # Test data (generated)
├── output.csv                      # Predictions (generated)
│
├── model.pkl                       # Trained Random Forest model
├── pipeline.pkl                    # Preprocessing pipeline
│
└── .gitignore                      # Git ignore file
```

---

## 📊 Model Performance

### Dataset Statistics
- **Total Samples**: 20,640 districts
- **Training Set**: 16,512 samples (80%)
- **Test Set**: 4,128 samples (20%)
- **Features**: 9 (8 numerical + 1 categorical)
- **Target**: Median house value (in $)

### Features Used

#### Numerical Features (8)
1. **longitude**: Longitude coordinate
2. **latitude**: Latitude coordinate
3. **housing_median_age**: Median age of houses in district
4. **total_rooms**: Total number of rooms
5. **total_bedrooms**: Total number of bedrooms
6. **population**: District population
7. **households**: Number of households
8. **median_income**: Median income (in $10,000s)

#### Categorical Features (1)
9. **ocean_proximity**: Distance to ocean
   - `<1H OCEAN`
   - `INLAND`
   - `ISLAND`
   - `NEAR BAY`
   - `NEAR OCEAN`

### Model: Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    # Default parameters provide good baseline performance
)
```

### Expected Performance
- **RMSE**: ~$50,000 - $55,000
- **R² Score**: ~0.80 - 0.85
- **Cross-Validation Score**: Consistent across folds

### Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score

# On test set
predictions = model.predict(X_test_prepared)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
```

---

## 🔍 Technical Details

### Stratified Sampling Strategy

```python
# Income-based stratification ensures representative splits
housing['income_cat'] = pd.cut(
    housing["median_income"], 
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
    labels=[1, 2, 3, 4, 5]
)
```

**Why Stratified Sampling?**
- Median income is a strong predictor
- Ensures all income categories represented proportionally
- Prevents sampling bias
- More reliable performance estimates

### Preprocessing Pipeline

#### Numerical Features Pipeline
```python
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
    ("scaler", StandardScaler())                     # Normalize features
])
```

#### Categorical Features Pipeline
```python
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # One-hot encode
])
```

#### Combined Pipeline
```python
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs), 
    ("cat", cat_pipeline, cat_attribs)
])
```

### Model Persistence

```python
# Save trained model and pipeline
joblib.dump(model, "model.pkl")
joblib.dump(pipeline, "pipeline.pkl")

# Load for inference
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")
```

**Benefits:**
- Fast loading (< 1 second)
- No retraining needed
- Consistent preprocessing
- Production-ready

---

## 📈 Sample Predictions

### Input Example
```csv
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity
-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,NEAR BAY
-122.22,37.86,21.0,7099.0,1106.0,2401.0,1138.0,8.3014,NEAR BAY
```

### Output Example
```csv
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity,median_house_value
-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,NEAR BAY,452600.0
-122.22,37.86,21.0,7099.0,1106.0,2401.0,1138.0,8.3014,NEAR BAY,358500.0
```

---

## 🔮 Future Enhancements

### Model Improvements
- [ ] **Hyperparameter Tuning**: Grid/Random Search for optimal parameters
- [ ] **Feature Engineering**: Create derived features (rooms per household, etc.)
- [ ] **Ensemble Methods**: Combine multiple models (Gradient Boosting, XGBoost)
- [ ] **Deep Learning**: Neural network for non-linear patterns
- [ ] **Feature Selection**: Identify most important features

### Technical Enhancements
- [ ] **API Deployment**: Flask/FastAPI REST API
- [ ] **Docker Container**: Containerize for easy deployment
- [ ] **CI/CD Pipeline**: Automated testing and deployment
- [ ] **Model Monitoring**: Track prediction drift and accuracy
- [ ] **A/B Testing**: Compare model versions in production

### User Interface
- [ ] **Web Application**: Streamlit/Gradio interface
- [ ] **Interactive Map**: Visualize predictions on California map
- [ ] **Batch Upload**: Process multiple CSV files
- [ ] **Model Explainability**: SHAP values for interpretability

### Data Enhancements
- [ ] **Real-Time Data**: Integrate with real estate APIs
- [ ] **More Features**: Add crime rates, school ratings, amenities
- [ ] **Temporal Analysis**: Track price trends over time
- [ ] **Regional Models**: Separate models for different regions

---

## 🧪 Model Comparison

The project uses Random Forest, but you can easily compare with other models:

```python
# In the training section, try different models:

# Linear Regression
model = LinearRegression()

# Decision Tree
model = DecisionTreeRegressor(random_state=42)

# Random Forest (current)
model = RandomForestRegressor(random_state=42)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=42)
```

---

## 🔬 Cross-Validation

To get robust performance estimates:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, 
    housing_prepared, 
    housing_labels,
    scoring="neg_mean_squared_error", 
    cv=10
)

rmse_scores = np.sqrt(-scores)
print(f"RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
```

---

## 🐛 Troubleshooting

### Common Issues

#### Missing Files
```bash
FileNotFoundError: housing.csv not found
```
**Solution**: Ensure `housing.csv` is in the project directory

#### Memory Errors
```bash
MemoryError: Unable to allocate array
```
**Solution**: Use smaller batch sizes or upgrade RAM

#### Import Errors
```bash
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: 
```bash
pip install -r requirements.txt
```

### Debugging Tips

Enable verbose output:
```python
# Add print statements
print(f"Training set shape: {housing.shape}")
print(f"Features: {num_attribs + cat_attribs}")
print(f"Model parameters: {model.get_params()}")
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests
- Update documentation

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Ashwinder Singh**
- **GitHub**: [@Ashwinder0186](https://github.com/Ashwinder0186)
- **LinkedIn**: [singh-ashwinder](https://linkedin.com/in/singh-ashwinder)
- **Email**: singhashwinder0186@gmail.com
- **Education**: MS in Computer Science, University of Texas at Arlington (GPA: 4.0/4.0)

### Background
Machine Learning Engineer with expertise in building production-ready ML pipelines and predictive models. Experience includes developing quantitative analytics solutions at Tata Consultancy Services for JPMorgan Chase financial systems. Passionate about applying machine learning to real-world problems.

**Core Competencies:**
- Machine Learning Pipeline Development
- Statistical Modeling & Data Analysis
- Python & Scikit-Learn
- Model Deployment & Production Systems

---

## 🙏 Acknowledgments

- **Dataset**: California Housing Dataset from Scikit-Learn
- **Libraries**: Scikit-Learn, Pandas, NumPy, Joblib
- **Inspiration**: Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow

---

## 📚 References

- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Pipeline & ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)

---

## 📞 Support

For questions, issues, or suggestions:
- **Email**: singhashwinder0186@gmail.com
- **Issues**: [Create an issue](https://github.com/Ashwinder0186/California-Housing-Price-Predictor/issues)
- **Discussions**: GitHub Discussions tab

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star!**

**🏠 Predicting California Housing Prices with Machine Learning**

Made with ❤️ and Python

</div>
