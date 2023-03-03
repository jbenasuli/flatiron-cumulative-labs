# Preprocessing with scikit-learn - Cumulative Lab

## Introduction
In this cumulative lab, you'll practice applying various preprocessing techniques with scikit-learn (`sklearn`) to the Ames Housing dataset in order to prepare the data for predictive modeling. The main emphasis here is on preprocessing (not EDA or modeling theory), so we will skip over most of the visualization and metrics steps that you would take in an actual modeling process.

## Objectives

You will be able to:

* Practice identifying which preprocessing technique to use
* Practice filtering down to relevant columns
* Practice applying `sklearn.impute` to fill in missing values
* Practice applying `sklearn.preprocessing`:
  * `OrdinalEncoder` for converting binary categories to 0 and 1 within a single column
  * `OneHotEncoder` for creating multiple "dummy" columns to represent multiple categories

## Your Task: Prepare the Ames Housing Dataset for Modeling

![house in Ames](images/ames_house.jpg)

<span>Photo by <a href="https://unsplash.com/@kjkempt17?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Kyle Kempt</a> on <a href="https://unsplash.com/s/photos/ames?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

### Requirements

#### 1. Drop Irrelevant Columns

For the purposes of this lab, we will only be using a subset of all of the features present in the Ames Housing dataset. In this step you will drop all irrelevant columns.

#### 2. Handle Missing Values

Often for reasons outside of a data scientist's control, datasets are missing some values. In this step you will assess the presence of NaN values in our subset of data, and use `MissingIndicator` and `SimpleImputer` from the `sklearn.impute` submodule to handle any missing values.

#### 3. Convert Categorical Features into Numbers

A built-in assumption of the scikit-learn library is that all data being fed into a machine learning model is already in a numeric format, otherwise you will get a `ValueError` when you try to fit a model. In this step you will use an `OrdinalEncoder` to replace data within individual non-numeric columns with 0s and 1s, and a `OneHotEncoder` to replace columns containing more than 2 categories with multiple "dummy" columns containing 0s and 1s.

At this point, a scikit-learn model should be able to run without errors!

#### 4. Preprocess Test Data

Apply Steps 1-3 to the test data in order to perform a final model evaluation.

## Lab Setup

### Getting the Data

In the cell below, we import the `pandas` library, open the CSV containing the Ames Housing data as a pandas `DataFrame`, and inspect its contents.


```python
import pandas as pd
df = pd.read_csv("data/ames.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>1456</td>
      <td>60</td>
      <td>RL</td>
      <td>62.0</td>
      <td>7917</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>175000</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>1457</td>
      <td>20</td>
      <td>RL</td>
      <td>85.0</td>
      <td>13175</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>210000</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>70</td>
      <td>RL</td>
      <td>66.0</td>
      <td>9042</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>2500</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>266500</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>1459</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>9717</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142125</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>1460</td>
      <td>20</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>147500</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 81 columns</p>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>



The prediction target for this analysis is the sale price of the home, so we separate the data into `X` and `y` accordingly:


```python
y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)
```

Next, we separate the data into a train set and a test set prior to performing any preprocessing steps:


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

(If you are working through this lab and you just want to start over with the original value for `X_train`, re-run the cell above.)


```python
print(f"X_train is a DataFrame with {X_train.shape[0]} rows and {X_train.shape[1]} columns")
print(f"y_train is a Series with {y_train.shape[0]} values")

# We always should have the same number of rows in X as values in y
assert X_train.shape[0] == y_train.shape[0]
```

    X_train is a DataFrame with 1095 rows and 80 columns
    y_train is a Series with 1095 values


#### Fitting a Model

For this lab we will be using a `LinearRegression` model from scikit-learn ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)).

Right now, we have not done any preprocessing, so we expect that trying to fit a model will fail:


```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-6-6d405156d44f> in <module>
          3 
          4 model = LinearRegression()
    ----> 5 model.fit(X_train, y_train)
    

    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_base.py in fit(self, X, y, sample_weight)
        503 
        504         n_jobs_ = self.n_jobs
    --> 505         X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc', 'coo'],
        506                                    y_numeric=True, multi_output=True)
        507 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
        430                 y = check_array(y, **check_y_params)
        431             else:
    --> 432                 X, y = check_X_y(X, y, **check_params)
        433             out = X, y
        434 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
        793         raise ValueError("y cannot be None")
        794 
    --> 795     X = check_array(X, accept_sparse=accept_sparse,
        796                     accept_large_sparse=accept_large_sparse,
        797                     dtype=dtype, order=order, copy=copy,


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        596                     array = array.astype(dtype, casting="unsafe", copy=False)
        597                 else:
    --> 598                     array = np.asarray(array, order=order, dtype=dtype)
        599             except ComplexWarning:
        600                 raise ValueError("Complex data not supported\n"


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/numpy/core/_asarray.py in asarray(a, dtype, order)
         83 
         84     """
    ---> 85     return array(a, dtype, copy=False, order=order)
         86 
         87 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/pandas/core/generic.py in __array__(self, dtype)
       1779 
       1780     def __array__(self, dtype=None) -> np.ndarray:
    -> 1781         return np.asarray(self._values, dtype=dtype)
       1782 
       1783     def __array_wrap__(self, result, context=None):


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/numpy/core/_asarray.py in asarray(a, dtype, order)
         83 
         84     """
    ---> 85     return array(a, dtype, copy=False, order=order)
         86 
         87 


    ValueError: could not convert string to float: 'RL'


As you can see, we got `ValueError: could not convert string to float: 'RL'`.

In order to fit a scikit-learn model, all values must be numeric, and the third column of our full dataset (`MSZoning`) contains values like `'RL'` and `'RH'`, which are strings. So this error was expected, but after some preprocessing, this model will work!

## 1. Drop Irrelevant Columns

For the purpose of this analysis, we'll only use the following columns, described by `relevant_columns`. You can find the full description of their values in the file `data/data_description.txt` included in this repository.

In the cell below, reassign `X_train` so that it only contains the columns in `relevant_columns`.

**Hint:** Even though we describe this as "dropping" irrelevant columns, it's easier if you invert the logic, so that we are only keeping relevant columns, rather than using the `.drop()` method. It is possible to use the `.drop()` method if you really want to, but first you would need to create a list of the column names that you don't want to keep.


```python

# Declare relevant columns
relevant_columns = [
    'LotFrontage',  # Linear feet of street connected to property
    'LotArea',      # Lot size in square feet
    'Street',       # Type of road access to property
    'OverallQual',  # Rates the overall material and finish of the house
    'OverallCond',  # Rates the overall condition of the house
    'YearBuilt',    # Original construction date
    'YearRemodAdd', # Remodel date (same as construction date if no remodeling or additions)
    'GrLivArea',    # Above grade (ground) living area square feet
    'FullBath',     # Full bathrooms above grade
    'BedroomAbvGr', # Bedrooms above grade (does NOT include basement bedrooms)
    'TotRmsAbvGrd', # Total rooms above grade (does not include bathrooms)
    'Fireplaces',   # Number of fireplaces
    'FireplaceQu',  # Fireplace quality
    'MoSold',       # Month Sold (MM)
    'YrSold'        # Year Sold (YYYY)
]

# Reassign X_train so that it only contains relevant columns
X_train = X_train.loc[:, relevant_columns]


# Visually inspect X_train
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>Fa</td>
      <td>1</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>10</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>626</th>
      <td>NaN</td>
      <td>12342</td>
      <td>Pave</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>TA</td>
      <td>8</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>Gd</td>
      <td>3</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>Pave</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>TA</td>
      <td>12</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>Pave</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>Pave</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>2009</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 15 columns</p>
</div>



Check that the new shape is correct:


```python

# X_train should have the same number of rows as before
assert X_train.shape[0] == 1095

# Now X_train should only have as many columns as relevant_columns
assert X_train.shape[1] == len(relevant_columns)
```

## 2. Handle Missing Values

In the cell below, we check to see if there are any NaNs in the selected subset of data:


```python
X_train.isna().sum()
```




    LotFrontage     200
    LotArea           0
    Street            0
    OverallQual       0
    OverallCond       0
    YearBuilt         0
    YearRemodAdd      0
    GrLivArea         0
    FullBath          0
    BedroomAbvGr      0
    TotRmsAbvGrd      0
    Fireplaces        0
    FireplaceQu     512
    MoSold            0
    YrSold            0
    dtype: int64



Ok, it looks like we have some NaNs in `LotFrontage` and `FireplaceQu`.

Before we proceed to fill in those values, we need to ask: **do these NaNs actually represent** ***missing*** **values, or is there some real value/category being represented by NaN?**

### Fireplace Quality

To start with, let's look at `FireplaceQu`, which means "Fireplace Quality". Why might we have NaN fireplace quality?

Well, some properties don't have fireplaces!

Let's confirm this guess with a little more analysis.

First, we know that there are 512 records with NaN fireplace quality. How many records are there with zero fireplaces?


```python
X_train[X_train["Fireplaces"] == 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>10</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>839</th>
      <td>70.0</td>
      <td>11767</td>
      <td>Pave</td>
      <td>5</td>
      <td>6</td>
      <td>1946</td>
      <td>1995</td>
      <td>1200</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>5</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>430</th>
      <td>21.0</td>
      <td>1680</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1971</td>
      <td>1971</td>
      <td>987</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>7</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>513</th>
      <td>71.0</td>
      <td>9187</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1983</td>
      <td>1983</td>
      <td>1080</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>40.0</td>
      <td>3951</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2009</td>
      <td>2009</td>
      <td>1224</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>330</th>
      <td>NaN</td>
      <td>10624</td>
      <td>Pave</td>
      <td>5</td>
      <td>4</td>
      <td>1964</td>
      <td>1964</td>
      <td>1728</td>
      <td>2</td>
      <td>6</td>
      <td>10</td>
      <td>0</td>
      <td>NaN</td>
      <td>11</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>63.0</td>
      <td>13072</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
      <td>1141</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>121</th>
      <td>50.0</td>
      <td>6060</td>
      <td>Pave</td>
      <td>4</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1123</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>Pave</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
<p>512 rows × 15 columns</p>
</div>



Ok, that's 512 rows, same as the number of NaN `FireplaceQu` records. To double-check, let's query for that combination of factors (zero fireplaces and `FireplaceQu` is NaN):


```python
X_train[
    (X_train["Fireplaces"] == 0) &
    (X_train["FireplaceQu"].isna())
]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>10</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>839</th>
      <td>70.0</td>
      <td>11767</td>
      <td>Pave</td>
      <td>5</td>
      <td>6</td>
      <td>1946</td>
      <td>1995</td>
      <td>1200</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>5</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>430</th>
      <td>21.0</td>
      <td>1680</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1971</td>
      <td>1971</td>
      <td>987</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>7</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>513</th>
      <td>71.0</td>
      <td>9187</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1983</td>
      <td>1983</td>
      <td>1080</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>40.0</td>
      <td>3951</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2009</td>
      <td>2009</td>
      <td>1224</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>330</th>
      <td>NaN</td>
      <td>10624</td>
      <td>Pave</td>
      <td>5</td>
      <td>4</td>
      <td>1964</td>
      <td>1964</td>
      <td>1728</td>
      <td>2</td>
      <td>6</td>
      <td>10</td>
      <td>0</td>
      <td>NaN</td>
      <td>11</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>63.0</td>
      <td>13072</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
      <td>1141</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>121</th>
      <td>50.0</td>
      <td>6060</td>
      <td>Pave</td>
      <td>4</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1123</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>Pave</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
<p>512 rows × 15 columns</p>
</div>



Looks good, still 512 records. So, NaN fireplace quality is not actually information that is missing from our dataset, it is a genuine category which means "fireplace quality is not applicable". This interpretation aligns with what we see in `data/data_description.txt`:

```
...
FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
...
```

So, let's replace those NaNs with the string "N/A" to indicate that this is a real category, not missing data:


```python
X_train["FireplaceQu"] = X_train["FireplaceQu"].fillna("N/A")
X_train["FireplaceQu"].value_counts()
```




    N/A    512
    Gd     286
    TA     236
    Fa      26
    Ex      19
    Po      16
    Name: FireplaceQu, dtype: int64



Eventually we will still need to perform some preprocessing to prepare the `FireplaceQu` column for modeling (because models require numeric inputs rather than inputs of type `object`), but we don't need to worry about filling in missing values.

### Lot Frontage

Now let's look at `LotFrontage` — it's possible that NaN is also a genuine category here, and it's possible that it's just missing data instead. Let's apply some domain understanding to understand whether it's possible that lot frontage can be N/A just like fireplace quality can be N/A.

Lot frontage is defined as the "Linear feet of street connected to property", i.e. how much of the property runs directly along a road. The amount of frontage required for a property depends on its zoning. Let's look at the zoning of all records with NaN for `LotFrontage`:


```python
df[df["LotFrontage"].isna()]["MSZoning"].value_counts()
```




    RL    229
    RM     19
    FV      8
    RH      3
    Name: MSZoning, dtype: int64



So, we have RL (residential low density), RM (residential medium density), FV (floating village residential), and RH (residential high density). Looking at the building codes from the City of Ames, it appears that all of these zones require at least 24 feet of frontage.

Nevertheless, we can't assume that all properties have frontage just because the zoning regulations require it. Maybe these properties predate the regulations, or they received some kind of variance permitting them to get around the requirement. **It's still not as clear here as it was with the fireplaces whether this is a genuine "not applicable" kind of NaN or a "missing information" kind of NaN.**

In a case like this, we can take a double approach:

1. Make a new column in the dataset that simply represents whether `LotFrontage` was originally NaN
2. Fill in the NaN values of `LotFrontage` with median frontage in preparation for modeling

### Missing Indicator for `LotFrontage`

First, we import `sklearn.impute.MissingIndicator` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html)). The goal of using a `MissingIndicator` is creating a new column to represent which values were NaN (or some other "missing" value) in the original dataset, in case NaN ends up being a meaningful indicator rather than a random missing bit of data.

A `MissingIndicator` is a scikit-learn transformer, meaning that we will use the standard steps for any scikit-learn transformer:

1. Identify data to be transformed (typically not every column is passed to every transformer)
2. Instantiate the transformer object
3. Fit the transformer object (on training data only)
4. Transform data using the transformer object
5. Add the transformed data to the other data that was not transformed


```python
from sklearn.impute import MissingIndicator

# (1) Identify data to be transformed
# We only want missing indicators for LotFrontage
frontage_train = X_train[["LotFrontage"]]

# (2) Instantiate the transformer object
missing_indicator = MissingIndicator()

# (3) Fit the transformer object on frontage_train
missing_indicator.fit(frontage_train)

# (4) Transform frontage_train and assign the result
# to frontage_missing_train
frontage_missing_train = missing_indicator.transform(frontage_train)

# Visually inspect frontage_missing_train
frontage_missing_train
```




    array([[False],
           [False],
           [False],
           ...,
           [False],
           [False],
           [False]])



The result of transforming `frontage_train` should be an array of arrays, each containing `True` or `False`. Make sure the `assert`s pass before moving on to the next step.


```python
import numpy as np

# frontage_missing_train should be a NumPy array
assert type(frontage_missing_train) == np.ndarray

# We should have the same number of rows as the full X_train
assert frontage_missing_train.shape[0] == X_train.shape[0]

# But we should only have 1 column
assert frontage_missing_train.shape[1] == 1
```

Now let's add this new information as a new column of `X_train`:


```python

# (5) add the transformed data to the other data
X_train["LotFrontage_Missing"] = frontage_missing_train
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>2008</td>
      <td>False</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>Fa</td>
      <td>1</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>N/A</td>
      <td>10</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>626</th>
      <td>NaN</td>
      <td>12342</td>
      <td>Pave</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>TA</td>
      <td>8</td>
      <td>2007</td>
      <td>True</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>Gd</td>
      <td>3</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>Pave</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>TA</td>
      <td>12</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>Pave</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>Pave</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>2009</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 16 columns</p>
</div>




```python

# Now we should have 1 extra column compared to
# our original subset
assert X_train.shape[1] == len(relevant_columns) + 1
```

### Imputing Missing Values for LotFrontage

Now that we have noted where missing values were originally present, let's use a `SimpleImputer` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)) to fill in those NaNs in the `LotFrontage` column.

The process is very similar to the `MissingIndicator` process, except that we want to replace the original `LotFrontage` column with the transformed version instead of just adding a new column on.

In the cell below, create and use a `SimpleImputer` with `strategy="median"` to transform the value of `frontage_train` (declared above).


```python

from sklearn.impute import SimpleImputer

# (1) frontage_train was created previously, so we don't
# need to extract the relevant data again

# (2) Instantiate a SimpleImputer with strategy="median"
imputer = SimpleImputer(strategy="median")

# (3) Fit the imputer on frontage_train
imputer.fit(frontage_train)

# (4) Transform frontage_train using the imputer and
# assign the result to frontage_imputed_train
frontage_imputed_train = imputer.transform(frontage_train)

# Visually inspect frontage_imputed_train
frontage_imputed_train
```




    array([[43.],
           [78.],
           [60.],
           ...,
           [60.],
           [55.],
           [53.]])



Now we can replace the original value of `LotFrontage` in `X_train` with the new value:


```python

# (5) Replace value of LotFrontage
X_train["LotFrontage"] = frontage_imputed_train

# Visually inspect X_train
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>2008</td>
      <td>False</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>Fa</td>
      <td>1</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>N/A</td>
      <td>10</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342</td>
      <td>Pave</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>TA</td>
      <td>8</td>
      <td>2007</td>
      <td>True</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>Gd</td>
      <td>3</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>Pave</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>TA</td>
      <td>12</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>Pave</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>Pave</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>2009</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 16 columns</p>
</div>



Now the shape of `X_train` should still be the same as before:


```python
assert X_train.shape == (1095, 16)
```

And now our `X_train` no longer contains any NaN values:


```python
X_train.isna().sum()
```




    LotFrontage            0
    LotArea                0
    Street                 0
    OverallQual            0
    OverallCond            0
    YearBuilt              0
    YearRemodAdd           0
    GrLivArea              0
    FullBath               0
    BedroomAbvGr           0
    TotRmsAbvGrd           0
    Fireplaces             0
    FireplaceQu            0
    MoSold                 0
    YrSold                 0
    LotFrontage_Missing    0
    dtype: int64



Great! Now we have completed Step 2.

## 3. Convert Categorical Features into Numbers

Despite dropping irrelevant columns and filling in those NaN values, if we feed the current `X_train` into our scikit-learn `LinearRegression` model, it will crash:


```python
model.fit(X_train, y_train)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-22-5a4f775ea40e> in <module>
          1 # __SOLUTION__
    ----> 2 model.fit(X_train, y_train)
    

    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_base.py in fit(self, X, y, sample_weight)
        503 
        504         n_jobs_ = self.n_jobs
    --> 505         X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc', 'coo'],
        506                                    y_numeric=True, multi_output=True)
        507 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
        430                 y = check_array(y, **check_y_params)
        431             else:
    --> 432                 X, y = check_X_y(X, y, **check_params)
        433             out = X, y
        434 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
        793         raise ValueError("y cannot be None")
        794 
    --> 795     X = check_array(X, accept_sparse=accept_sparse,
        796                     accept_large_sparse=accept_large_sparse,
        797                     dtype=dtype, order=order, copy=copy,


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        596                     array = array.astype(dtype, casting="unsafe", copy=False)
        597                 else:
    --> 598                     array = np.asarray(array, order=order, dtype=dtype)
        599             except ComplexWarning:
        600                 raise ValueError("Complex data not supported\n"


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/numpy/core/_asarray.py in asarray(a, dtype, order)
         83 
         84     """
    ---> 85     return array(a, dtype, copy=False, order=order)
         86 
         87 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/pandas/core/generic.py in __array__(self, dtype)
       1779 
       1780     def __array__(self, dtype=None) -> np.ndarray:
    -> 1781         return np.asarray(self._values, dtype=dtype)
       1782 
       1783     def __array_wrap__(self, result, context=None):


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/numpy/core/_asarray.py in asarray(a, dtype, order)
         83 
         84     """
    ---> 85     return array(a, dtype, copy=False, order=order)
         86 
         87 


    ValueError: could not convert string to float: 'Pave'


Now the first column to cause a problem is `Street`, which is documented like this:

```
...
Street: Type of road access to property

       Grvl	Gravel	
       Pave	Paved
...
```

Let's look at the full `X_train`:


```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1095 entries, 1023 to 1126
    Data columns (total 16 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   LotFrontage          1095 non-null   float64
     1   LotArea              1095 non-null   int64  
     2   Street               1095 non-null   object 
     3   OverallQual          1095 non-null   int64  
     4   OverallCond          1095 non-null   int64  
     5   YearBuilt            1095 non-null   int64  
     6   YearRemodAdd         1095 non-null   int64  
     7   GrLivArea            1095 non-null   int64  
     8   FullBath             1095 non-null   int64  
     9   BedroomAbvGr         1095 non-null   int64  
     10  TotRmsAbvGrd         1095 non-null   int64  
     11  Fireplaces           1095 non-null   int64  
     12  FireplaceQu          1095 non-null   object 
     13  MoSold               1095 non-null   int64  
     14  YrSold               1095 non-null   int64  
     15  LotFrontage_Missing  1095 non-null   bool   
    dtypes: bool(1), float64(1), int64(12), object(2)
    memory usage: 137.9+ KB


So, our model is crashing because some of the columns are non-numeric.

Anything that is already `float64` or `int64` will work with our model, but these features need to be converted:

* `Street` (currently type `object`)
* `FireplaceQu` (currently type `object`)
* `LotFrontage_Missing` (currently type `bool`)

There are two main approaches to converting these values, depending on whether there are 2 values (meaning the categorical variable can be converted into a single binary number) or more than 2 values (meaning we need to create extra columns to represent all categories).

(If there is only 1 value, this is not a useful feature for the purposes of predictive analysis because every single row contains the same information.)

In the cell below, we inspect the value counts of the specified features:


```python

print(X_train["Street"].value_counts())
print()
print(X_train["FireplaceQu"].value_counts())
print()
print(X_train["LotFrontage_Missing"].value_counts())
```

    Pave    1091
    Grvl       4
    Name: Street, dtype: int64
    
    N/A    512
    Gd     286
    TA     236
    Fa      26
    Ex      19
    Po      16
    Name: FireplaceQu, dtype: int64
    
    False    895
    True     200
    Name: LotFrontage_Missing, dtype: int64


So, it looks like `Street` and `LotFrontage_Missing` have only 2 categories and can be converted into binary in place, whereas `FireplaceQu` has 6 categories and will need to be expanded into multiple columns.

### Binary Categories

For binary categories, we will use an `OrdinalEncoder` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)) to convert the categories of `Street` and `LotFrontage_Missing` into binary values (0s and 1s).

Just like in Step 2 when we used the `MissingIndicator` and `SimpleImputer`, we will follow these steps:

1. Identify data to be transformed
2. Instantiate the transformer object
3. Fit the transformer object (on training data only)
4. Transform data using the transformer object
5. Add the transformed data to the other data that was not transformed

Let's start with transforming `Street`:


```python

# (0) import OrdinalEncoder from sklearn.preprocessing
from sklearn.preprocessing import OrdinalEncoder

# (1) Create a variable street_train that contains the
# relevant column from X_train
# (Use double brackets [[]] to get the appropriate shape)
street_train = X_train[["Street"]]

# (2) Instantiate an OrdinalEncoder
encoder_street = OrdinalEncoder()

# (3) Fit the encoder on street_train
encoder_street.fit(street_train)

# Inspect the categories of the fitted encoder
encoder_street.categories_[0]
```




    array(['Grvl', 'Pave'], dtype=object)



The `.categories_` attribute of `OrdinalEncoder` is only present once the `.fit` method has been called. (The trailing `_` indicates this convention.)

What this tells us is that when `encoder_street` is used to transform the street data into 1s and 0s, `0` will mean `'Grvl'` (gravel) in the original data, and `1` will mean `'Pave'` (paved) in the original data.

The eventual scikit-learn model only cares about the 1s and 0s, but this information can be useful for us to understand what our code is doing and help us debug when things go wrong.

Now let's transform `street_train` with the fitted encoder:


```python

# (4) Transform street_train using the encoder and
# assign the result to street_encoded_train
street_encoded_train = encoder_street.transform(street_train)

# Flatten for appropriate shape
street_encoded_train = street_encoded_train.flatten()

# Visually inspect street_encoded_train
street_encoded_train
```




    array([1., 1., 1., ..., 1., 1., 1.])



All of the values we see appear to be `1` right now, but that makes sense since there were only 4 properties with gravel (`0`) streets in `X_train`.

Now let's replace the original `Street` column with the encoded version:


```python

# (5) Replace value of Street
X_train["Street"] = street_encoded_train

# Visually inspect X_train
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>1.0</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>2008</td>
      <td>False</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>1.0</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>Fa</td>
      <td>1</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>1.0</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>N/A</td>
      <td>10</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342</td>
      <td>1.0</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>TA</td>
      <td>8</td>
      <td>2007</td>
      <td>True</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>1.0</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>1.0</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>Gd</td>
      <td>3</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>1.0</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>TA</td>
      <td>12</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>1.0</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>1.0</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>1.0</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>2009</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 16 columns</p>
</div>




```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1095 entries, 1023 to 1126
    Data columns (total 16 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   LotFrontage          1095 non-null   float64
     1   LotArea              1095 non-null   int64  
     2   Street               1095 non-null   float64
     3   OverallQual          1095 non-null   int64  
     4   OverallCond          1095 non-null   int64  
     5   YearBuilt            1095 non-null   int64  
     6   YearRemodAdd         1095 non-null   int64  
     7   GrLivArea            1095 non-null   int64  
     8   FullBath             1095 non-null   int64  
     9   BedroomAbvGr         1095 non-null   int64  
     10  TotRmsAbvGrd         1095 non-null   int64  
     11  Fireplaces           1095 non-null   int64  
     12  FireplaceQu          1095 non-null   object 
     13  MoSold               1095 non-null   int64  
     14  YrSold               1095 non-null   int64  
     15  LotFrontage_Missing  1095 non-null   bool   
    dtypes: bool(1), float64(2), int64(12), object(1)
    memory usage: 137.9+ KB


Perfect! Now `Street` should by type `int64` instead of `object`.

Now, repeat the same process with `LotFrontage_Missing`:


```python

# (1) We already have a variable frontage_missing_train
# from earlier, no additional step needed

# (2) Instantiate an OrdinalEncoder for missing frontage
encoder_frontage_missing = OrdinalEncoder()

# (3) Fit the encoder on frontage_missing_train
encoder_frontage_missing.fit(frontage_missing_train)

# Inspect the categories of the fitted encoder
encoder_frontage_missing.categories_[0]
```




    array([False,  True])




```python

# (4) Transform frontage_missing_train using the encoder and
# assign the result to frontage_missing_encoded_train
frontage_missing_encoded_train = encoder_frontage_missing.transform(frontage_missing_train)

# Flatten for appropriate shape
frontage_missing_encoded_train = frontage_missing_encoded_train.flatten()

# Visually inspect frontage_missing_encoded_train
frontage_missing_encoded_train
```




    array([0., 0., 0., ..., 0., 0., 0.])




```python

# (5) Replace value of LotFrontage_Missing
X_train["LotFrontage_Missing"] = frontage_missing_encoded_train

# Visually inspect X_train
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>1.0</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>2008</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>1.0</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>Fa</td>
      <td>1</td>
      <td>2006</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>1.0</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>N/A</td>
      <td>10</td>
      <td>2009</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342</td>
      <td>1.0</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>TA</td>
      <td>8</td>
      <td>2007</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>1.0</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2007</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>1.0</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>Gd</td>
      <td>3</td>
      <td>2007</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>1.0</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>TA</td>
      <td>12</td>
      <td>2009</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>1.0</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2006</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>1.0</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>2007</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>1.0</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>2009</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 16 columns</p>
</div>




```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1095 entries, 1023 to 1126
    Data columns (total 16 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   LotFrontage          1095 non-null   float64
     1   LotArea              1095 non-null   int64  
     2   Street               1095 non-null   float64
     3   OverallQual          1095 non-null   int64  
     4   OverallCond          1095 non-null   int64  
     5   YearBuilt            1095 non-null   int64  
     6   YearRemodAdd         1095 non-null   int64  
     7   GrLivArea            1095 non-null   int64  
     8   FullBath             1095 non-null   int64  
     9   BedroomAbvGr         1095 non-null   int64  
     10  TotRmsAbvGrd         1095 non-null   int64  
     11  Fireplaces           1095 non-null   int64  
     12  FireplaceQu          1095 non-null   object 
     13  MoSold               1095 non-null   int64  
     14  YrSold               1095 non-null   int64  
     15  LotFrontage_Missing  1095 non-null   float64
    dtypes: float64(3), int64(12), object(1)
    memory usage: 145.4+ KB


Great, now we only have 1 column remaining that isn't type `float64` or `int64`!

#### Note on Preprocessing Boolean Values
For binary values like `LotFrontage_Missing`, you might see a few different approaches to preprocessing. Python treats `True` and `1` as equal:


```python
print(True == 1)
print(False == 0)
```

    True
    True


This means that if your model is purely using Python, you actually might just be able to leave columns as type `bool` without any issues. You will likely see examples that do this. However if your model relies on C or Java "under the hood", this might cause problems.

There is also a technique using `pandas` rather than scikit-learn for this particular conversion of boolean values to integers:


```python
df_example = pd.DataFrame(frontage_missing_train, columns=["LotFrontage_Missing"])
df_example
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 1 columns</p>
</div>




```python
df_example["LotFrontage_Missing"] = df_example["LotFrontage_Missing"].astype(int)
df_example
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 1 columns</p>
</div>



This code is casting every value in the `LotFrontage_Missing` column to an integer, achieving the same result as the `OrdinalEncoder` example with less code.

The downside of using this approach is that it doesn't fit into a scikit-learn pipeline as neatly because it is using `pandas` to do the transformation instead of scikit-learn.

In the future, you will need to make your own determination of which strategy to use!

### Multiple Categories

Unlike `Street` and `LotFrontage_Missing`, `FireplaceQu` has more than two categories. Therefore the process for encoding it numerically is a bit more complicated, because we will need to create multiple "dummy" columns that are each representing one category.

To do this, we can use a `OneHotEncoder` from `sklearn.preprocessing` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)).

The first several steps are very similar to all of the other transformers we've used so far, although the process of combining the data with the original data differs.

In the cells below, complete steps `(0)`-`(4)` of preprocessing the `FireplaceQu` column using a `OneHotEncoder`:


```python

# (0) import OneHotEncoder from sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder

# (1) Create a variable fireplace_qu_train
# extracted from X_train
# (double brackets due to shape expected by OHE)
fireplace_qu_train = X_train[["FireplaceQu"]]

# (2) Instantiate a OneHotEncoder with categories="auto",
# sparse=False, and handle_unknown="ignore"
ohe = OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")

# (3) Fit the encoder on fireplace_qu_train
ohe.fit(fireplace_qu_train)

# Inspect the categories of the fitted encoder
ohe.categories_
```




    [array(['Ex', 'Fa', 'Gd', 'N/A', 'Po', 'TA'], dtype=object)]




```python

# (4) Transform fireplace_qu_train using the encoder and
# assign the result to fireplace_qu_encoded_train
fireplace_qu_encoded_train = ohe.transform(fireplace_qu_train)

# Visually inspect fireplace_qu_encoded_train
fireplace_qu_encoded_train
```




    array([[0., 0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           ...,
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1.]])



Notice that this time, unlike with `MissingIndicator`, `SimpleImputer`, or `OrdinalEncoder`, we have created multiple columns of data out of a single column. The code below converts this unlabeled NumPy array into a readable pandas dataframe in preparation for merging it back with the rest of `X_train`:


```python

# (5a) Make the transformed data into a dataframe
fireplace_qu_encoded_train = pd.DataFrame(
    # Pass in NumPy array
    fireplace_qu_encoded_train,
    # Set the column names to the categories found by OHE
    columns=ohe.categories_[0],
    # Set the index to match X_train's index
    index=X_train.index
)

# Visually inspect new dataframe
fireplace_qu_encoded_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ex</th>
      <th>Fa</th>
      <th>Gd</th>
      <th>N/A</th>
      <th>Po</th>
      <th>TA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>626</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>813</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 6 columns</p>
</div>



A couple notes on the code above:

* The main goal of converting this into a dataframe (rather than converting `X_train` into a NumPy array, which would also allow them to be combined) is **readability** — to help you and others understand what your code is doing, and to help you debug. Eventually when you write this code as a pipeline, it will be NumPy arrays "under the hood".
* We are using just the **raw categories** from `FireplaceQu` as our new dataframe columns, but you'll also see examples where a lambda function or list comprehension is used to create column names indicating the original column name, e.g. `FireplaceQu_Ex`, `FireplaceQu_Fa` rather than just `Ex`, `Fa`. This is a design decision based on readability — the scikit-learn model will work the same either way.
* It is very important that **the index of the new dataframe matches the index of the main `X_train` dataframe**. Because we used `train_test_split`, the index of `X_train` is shuffled, so it goes `1023`, `810`, `1384` etc. instead of `0`, `1`, `2`, etc. If you don't specify an index for the new dataframe, it will assign the first record to the index `0` rather than `1023`. If you are ever merging encoded data like this and a bunch of NaNs start appearing, make sure that the indexes are lined up correctly! You also may see examples where the index of `X_train` has been reset, rather than specifying the index of the new dataframe — either way works.

Next, we want to drop the original `FireplaceQu` column containing the categorical data:

(For previous transformations we didn't need to drop anything because we were replacing 1 column with 1 new column in place, but one-hot encoding works differently.)


```python

# (5b) Drop original FireplaceQu column
X_train.drop("FireplaceQu", axis=1, inplace=True)

# Visually inspect X_train
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>1.0</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>2008</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>1.0</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2006</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>1.0</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342</td>
      <td>1.0</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>1.0</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>2007</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>1.0</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>2007</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>1.0</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>12</td>
      <td>2009</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>1.0</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>1.0</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>6</td>
      <td>2007</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>1.0</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>6</td>
      <td>2009</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 15 columns</p>
</div>



Finally, we want to concatenate the new dataframe together with the original `X_train`:


```python

# (5c) Concatenate the new dataframe with current X_train
X_train = pd.concat([X_train, fireplace_qu_encoded_train], axis=1)

# Visually inspect X_train
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>...</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
      <th>Ex</th>
      <th>Fa</th>
      <th>Gd</th>
      <th>N/A</th>
      <th>Po</th>
      <th>TA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>1.0</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>5</td>
      <td>2008</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>1.0</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>1.0</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342</td>
      <td>1.0</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>1.0</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>4</td>
      <td>2007</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>1.0</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>2007</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>1.0</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>12</td>
      <td>2009</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>1.0</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>1.0</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>6</td>
      <td>2007</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>1.0</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>6</td>
      <td>2009</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 21 columns</p>
</div>




```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1095 entries, 1023 to 1126
    Data columns (total 21 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   LotFrontage          1095 non-null   float64
     1   LotArea              1095 non-null   int64  
     2   Street               1095 non-null   float64
     3   OverallQual          1095 non-null   int64  
     4   OverallCond          1095 non-null   int64  
     5   YearBuilt            1095 non-null   int64  
     6   YearRemodAdd         1095 non-null   int64  
     7   GrLivArea            1095 non-null   int64  
     8   FullBath             1095 non-null   int64  
     9   BedroomAbvGr         1095 non-null   int64  
     10  TotRmsAbvGrd         1095 non-null   int64  
     11  Fireplaces           1095 non-null   int64  
     12  MoSold               1095 non-null   int64  
     13  YrSold               1095 non-null   int64  
     14  LotFrontage_Missing  1095 non-null   float64
     15  Ex                   1095 non-null   float64
     16  Fa                   1095 non-null   float64
     17  Gd                   1095 non-null   float64
     18  N/A                  1095 non-null   float64
     19  Po                   1095 non-null   float64
     20  TA                   1095 non-null   float64
    dtypes: float64(9), int64(12)
    memory usage: 188.2 KB


Ok, everything is numeric now! We have completed the minimum necessary preprocessing to use these features in a scikit-learn model!


```python
model.fit(X_train, y_train)
```




    LinearRegression()



Great, no error this time.

Let's use cross validation to take a look at the model's performance:


```python
from sklearn.model_selection import cross_val_score

cross_val_score(model, X_train, y_train, cv=3)
```




    array([0.75131297, 0.66405511, 0.80347971])



Not terrible, we are explaining between 66% and 80% of the variance in the target with our current feature set. Let's say that this is our final model and move on to preparing the test data.

## 4. Preprocess Test Data

> Apply Steps 1-3 to the test data in order to perform a final model evaluation.

This part is done for you, and it should work automatically, assuming you didn't change the names of any of the transformer objects. Note that we are intentionally **not instantiating or fitting the transformers** here, because you always want to fit transformers on the training data only.

*Step 1: Drop Irrelevant Columns*


```python
X_test = X_test.loc[:, relevant_columns]
```

*Step 2: Handle Missing Values*


```python

# Replace FireplaceQu NaNs with "N/A"s
X_test["FireplaceQu"] = X_test["FireplaceQu"].fillna("N/A")

# Add missing indicator for lot frontage
frontage_test = X_test[["LotFrontage"]]
frontage_missing_test = missing_indicator.transform(frontage_test)
X_test["LotFrontage_Missing"] = frontage_missing_test

# Impute missing lot frontage values
frontage_imputed_test = imputer.transform(frontage_test)
X_test["LotFrontage"] = frontage_imputed_test

# Check that there are no more missing values
X_test.isna().sum()
```




    LotFrontage            0
    LotArea                0
    Street                 0
    OverallQual            0
    OverallCond            0
    YearBuilt              0
    YearRemodAdd           0
    GrLivArea              0
    FullBath               0
    BedroomAbvGr           0
    TotRmsAbvGrd           0
    Fireplaces             0
    FireplaceQu            0
    MoSold                 0
    YrSold                 0
    LotFrontage_Missing    0
    dtype: int64



*Step 3: Convert Categorical Features into Numbers*


```python

# Encode street type
street_test = X_test[["Street"]]
street_encoded_test = encoder_street.transform(street_test).flatten()
X_test["Street"] = street_encoded_test

# Encode frontage missing
frontage_missing_test = X_test[["LotFrontage_Missing"]]
frontage_missing_encoded_test = encoder_frontage_missing.transform(frontage_missing_test).flatten()
X_test["LotFrontage_Missing"] = frontage_missing_encoded_test

# One-hot encode fireplace quality
fireplace_qu_test = X_test[["FireplaceQu"]]
fireplace_qu_encoded_test = ohe.transform(fireplace_qu_test)
fireplace_qu_encoded_test = pd.DataFrame(
    fireplace_qu_encoded_test,
    columns=ohe.categories_[0],
    index=X_test.index
)
X_test.drop("FireplaceQu", axis=1, inplace=True)
X_test = pd.concat([X_test, fireplace_qu_encoded_test], axis=1)

# Visually inspect X_test
X_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>...</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
      <th>Ex</th>
      <th>Fa</th>
      <th>Gd</th>
      <th>N/A</th>
      <th>Po</th>
      <th>TA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>70.0</td>
      <td>8414</td>
      <td>1.0</td>
      <td>6</td>
      <td>8</td>
      <td>1963</td>
      <td>2003</td>
      <td>1068</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>98.0</td>
      <td>12256</td>
      <td>1.0</td>
      <td>8</td>
      <td>5</td>
      <td>1994</td>
      <td>1995</td>
      <td>2622</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>4</td>
      <td>2010</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>413</th>
      <td>56.0</td>
      <td>8960</td>
      <td>1.0</td>
      <td>5</td>
      <td>6</td>
      <td>1927</td>
      <td>1950</td>
      <td>1028</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>2010</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>522</th>
      <td>50.0</td>
      <td>5000</td>
      <td>1.0</td>
      <td>6</td>
      <td>7</td>
      <td>1947</td>
      <td>1950</td>
      <td>1664</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>10</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>89.0</td>
      <td>12898</td>
      <td>1.0</td>
      <td>9</td>
      <td>5</td>
      <td>2007</td>
      <td>2008</td>
      <td>1620</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>9</td>
      <td>2009</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>988</th>
      <td>70.0</td>
      <td>12046</td>
      <td>1.0</td>
      <td>6</td>
      <td>6</td>
      <td>1976</td>
      <td>1976</td>
      <td>2030</td>
      <td>2</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
      <td>6</td>
      <td>2007</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>243</th>
      <td>75.0</td>
      <td>10762</td>
      <td>1.0</td>
      <td>6</td>
      <td>6</td>
      <td>1980</td>
      <td>1980</td>
      <td>1217</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>4</td>
      <td>2009</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1342</th>
      <td>70.0</td>
      <td>9375</td>
      <td>1.0</td>
      <td>8</td>
      <td>5</td>
      <td>2002</td>
      <td>2002</td>
      <td>2169</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>70.0</td>
      <td>29959</td>
      <td>1.0</td>
      <td>7</td>
      <td>6</td>
      <td>1994</td>
      <td>1994</td>
      <td>1850</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>2009</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1418</th>
      <td>71.0</td>
      <td>9204</td>
      <td>1.0</td>
      <td>5</td>
      <td>5</td>
      <td>1963</td>
      <td>1963</td>
      <td>1144</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>365 rows × 21 columns</p>
</div>



Fit the model on the full training set, evaluate on test set:


```python
model.fit(X_train, y_train)
model.score(X_test, y_test)
```




    0.8016639002688328



Great, that worked! Now we have completed the full process of preprocessing the Ames Housing data in preparation for machine learning!

## Summary

In this cumulative lab, you used various techniques to prepare the Ames Housing data for modeling. You filtered down the full dataset to only relevant columns, filled in missing values, and converted categorical data into numeric data. Each time, you practiced the scikit-learn transformer workflow by instantiating the transformer, fitting on the relevant training data, transforming the training data, and transforming the test data at the end (without re-instantiating or re-fitting the transformer object).
