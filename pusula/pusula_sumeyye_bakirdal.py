# Required Libraries and Functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10)
pd.set_option('display.max_rows', 30)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_excel("side_effect_data 1.xlsx")
##################################
# Exploratory Data Analysis
##################################

def check_df(dataframe, head=5):
    """ Basic analysis of the dataframe"""
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe())

check_df(df)

# Separation of variables into categorical and numerical
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """   
    Returns the names of categorical, numeric and categorical but cardinal variables in the data set.
    Note Categorical variables include categorical variables with numeric appearance.
    Parameters
    ------
        dataframe: dataframe
                Dataframe to get variable names
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optional
                class threshold for categorical but cardinal variables
    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                List of cardinal variables with categorical view
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset(“iris”)
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"]
    cat_cols = list(set(cat_cols + num_but_cat) - set(cat_but_car))
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O" and col not in num_but_cat]
    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Analysis of categorical variables
def cat_summary(dataframe, col_name, plot=False):
    print(f"Column: {col_name}")
    value_counts = dataframe[col_name].value_counts()
    print(pd.DataFrame({col_name: value_counts, "Ratio": 100 * value_counts / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title(f'Count Plot of {col_name}')
        plt.show()

def categorical_summary(dataframe, plot=False):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O" or dataframe[col].nunique() < 10]
    for col in cat_cols:
        cat_summary(dataframe, col, plot=plot)

categorical_summary(df, plot=True)

# DateTime analysis
def datetime_analysis(dataframe, datetime_cols):
    for col in datetime_cols:
        print(f"### {col} ###")
        print("Min:", dataframe[col].min())
        print("Max:", dataframe[col].max())
        print("Mean:", dataframe[col].mean())
        print("Missing Values:", dataframe[col].isnull().sum())
        print("Unique Values:", dataframe[col].nunique())
        plt.figure(figsize=(12, 6))
        dataframe[col].hist(bins=30)
        plt.title(f"{col} Histogram")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

datetime_cols = ['Dogum_Tarihi', 'Ilac_Baslangic_Tarihi', 'Ilac_Bitis_Tarihi', 'Yan_Etki_Bildirim_Tarihi']
datetime_analysis(df, datetime_cols)

# Analysis of numerical variables
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# Analysis of numerical variables by target
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Yan_Etki", col)

# Correlation matrix
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
plt.figure(figsize=[18, 13])
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="magma")
plt.title("Correlation Matrix", fontsize=20)
plt.show()

##################################
# Feature Engineering
##################################

# Analysis of missing values
def missing_values_table(dataframe):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print("Eksik değer tablosu:\n", missing_df)

missing_values_table(df)

# Filling in missing values
def fill_missing_values(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    imputer_num = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
    
    categorical_cols = df.select_dtypes(include=[object]).columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    
    return df

df = fill_missing_values(df)

# Outlier analysis
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    if dataframe[col_name].dtype in ['int64', 'float64']:
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit
    return None, None

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if low_limit is not None and up_limit is not None:
        dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
        dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

# Feature extraction
## Creating the age variable
df['YAS'] = (pd.to_datetime('today') - df['Dogum_Tarihi']).dt.days // 365

# Age categories
df.loc[(df["YAS"] >= 21) & (df["YAS"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["YAS"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI calculation
df['BMI'] = df['Kilo'] / (df['Boy'] / 100) ** 2

# BMI categories
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Age and BMI combination
df.loc[(df["BMI"] < 18.5) & ((df["YAS"] >= 21) & (df["YAS"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["YAS"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["YAS"] >= 21) & (df["YAS"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["YAS"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["YAS"] >= 21) & (df["YAS"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["YAS"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] >= 30) & ((df["YAS"] >= 21) & (df["YAS"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] >= 30) & (df["YAS"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# New side effect based on side effect reporting date
df['YENI_YAN_ETKI'] = df['Yan_Etki_Bildirim_Tarihi'].notnull().astype(int)

# BMI categories based on weight and height combination
df.loc[(df["YAS"] >= 21) & (df["YAS"] < 50) & (df["BMI"] < 18.5), "NEW_AGE_BMI_CAT"] = "lowmature"
df.loc[(df["YAS"] >= 50) & (df["BMI"] < 18.5), "NEW_AGE_BMI_CAT"] = "lowsenior"
df.loc[(df["YAS"] >= 21) & (df["YAS"] < 50) & (df["BMI"] >= 18.5) & (df["BMI"] < 25), "NEW_AGE_BMI_CAT"] = "normalmature"
df.loc[(df["YAS"] >= 50) & (df["BMI"] >= 18.5) & (df["BMI"] < 25), "NEW_AGE_BMI_CAT"] = "normalsenior"


# Viewing the final statuse
print(df.head())
df.shape
# Checking the types of variables
# Checking data types of categorical variables
print("Categorical Columns Types:")
print(df[cat_cols].dtypes)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
print("Binary Columns:", binary_cols)

for col in binary_cols:
    df = label_encoder(df, col)
# One-Hot Encoding Process
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Yan_Etki"]]
print("Updated Categorical Columns:", cat_cols)

# Adjusting categorical variables
for col in cat_cols:
    df[col] = df[col].astype('category')

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first,dtype=int)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)

# Checking data types after one-hot encoding
print("One-Hot Encoding Sonrası Veri Türleri:")
print(df.dtypes)

# Checking changes
print("Güncellenmiş Veri Türleri:")
print(df.dtypes)

# Viewing results
print("İlk 5 Satır:")
print(df.head(5))
print("Veri Setinin Boyutu:", df.shape)

##################################
# STANDARDIZATION
##################################

print(num_cols)

# Filtering to keep only numerical data types in num_cols 
num_cols = [col for col in num_cols if df[col].dtype in ['int64', 'float64']]
print("Sayısal Sütunlar:", num_cols)

# Standardization process
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Viewing results
print("Standartlaştırılmış Veri İlk 5 Satır:")
print(df.head())
print("Veri Setinin Boyutu:", df.shape)
