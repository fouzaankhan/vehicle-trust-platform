## EDA Observations — Session 2

- Total rows: 558837
- Total columns: 16
- Columns with missing values: [
 'make',
 'model',
 'trim',
 'body',
 'transmission',
 'vin',
 'condition',
 'odometer',
 'color',
 'interior',
 'mmr',
 'sellingprice',
 'saledate'
]
- Price range (min to max): 1 → 230000
- Most common make: Ford
- Anything that looks suspicious or unexpected: 
    - Selling prices range from 1 to 230000, which suggests the presence of unrealistic outliers or incorrect entries.
    - The transmission column has a very large number of missing values (65K+ rows), which may reduce its reliability as a feature.
    - Some target values (sellingprice) are missing, which is problematic because target rows usually need to be removed before training.
    - The dataset is heavily dominated by a few manufacturers like Ford and Chevrolet, which could introduce model bias toward common brands.
    - saledate is currently stored as a string instead of datetime format and will require preprocessing before time-based analysis.