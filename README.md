## Author
**Name:** Sümeyye Bakırdal

**Email:** sumeyyebakirdal@gmail.com
# Data
The dataset used in this analysis is named side_effect_data 1.xlsx. It contains various features related to patients, drugs, and reported side effects.

# Methods
## Exploratory Data Analysis (EDA)
1. **Basic Analysis:** Used functions to check the shape, data types, and missing values in the dataset.
2. **Categorical Analysis:** Summarized categorical variables and visualized their distributions.
3. **Numerical Analysis:** Analyzed numerical variables and visualized distributions with histograms.
4. **Datetime Analysis:** Investigated datetime columns for insights.
## Feature Engineering
- Handled missing values using appropriate imputation techniques.
* Analyzed and managed outliers based on interquartile ranges.
+ Created new features, such as age and BMI categories, to enhance model performance.
## Standardization
Used `StandardScaler` to standardize numerical features for better model performance.

## Results
The initial analysis provides insights into the dataset, identifying key variables and relationships, which can inform further modeling.

## Conclusion
This project lays the groundwork for modeling the impact of various factors on the likelihood of experiencing drug side effects. Future work will include applying machine learning algorithms to predict side effects based on the processed data.

## Running the Project

To run this project, follow these steps:

1. **Clone the Repository**: First, clone the repository to your local machine using the command
   ```bash
   git clone https://github.com/sumeyyebakirdal/Pusula_Sumeyye_Bakirdal.git
2. **Navigate to the Project Directory**: Change to the project directory
   ```bash
   cd Pusula_Sumeyye_Bakirdal
3. **Install Required Packages**:Ensure you have the required Python packages installed. You can install them using pip
    ```bash
    pip install -r requirements.txt
4. **Run the Main Script**:Finally, run the main script
    ```bash
    python pusula_sumeyye_bakirdal.py


