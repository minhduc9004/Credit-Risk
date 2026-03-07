Dataset is too large for GitHub.
Download Raw Data from:
https://drive.google.com/file/d/1KZ8tMBhNlpTZTmRVzqqGyHLQNMVdI00e/view?usp=sharing
## Data Preprocessing
Before modeling, the dataset was cleaned and standardized to ensure compatibility with traditional logistic regression and machine learning models, which require numerical inputs.
- Variables containing identifiers, text descriptions, or information with little predictive relevance were removed.
- Categorical variables stored as character values were transformed into appropriate numerical formats, including ordinal encoding and dummy variables where necessary.
- Binary categorical variables were converted into numerical indicators (0/1).
- Date variables were standardized and transformed into numerical representations based on the number of days relative to the loan issuance date.
- Employment-related and other textual numerical variables were converted into numeric values.
These preprocessing steps ensured that all variables were in a consistent numeric format suitable for statistical and machine learning models.

The original dataset contains approximately 2.26 million observations and 158 variables. Data preprocessing was conducted to address missing values and ensure consistency across different types of loan applications.
Missing values were first analyzed by calculating the proportion of missing observations for each variable. 
-For variables with a low missing rate (≤5%), observations containing missing values were removed.
-Variables with a high proportion of missing values were removed from the dataset, as they may reduce data quality and negatively affect model performance.
After preprocessing and data cleaning, the final dataset contains 2,013,321 observations and 102 variables, which are used for subsequent modeling and analysis.

Download Clean Data from:
https://drive.google.com/file/d/1SEpqWIyXYhcC58EbDmMpZVLa6Th2WkRM/view?usp=drive_link
