# Display all column names
print(df.columns)
Index(['cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id',
       'license', 'abstract', 'publish_time', 'authors', 'journal', 'url'],
      dtype='object')

   # Display data types
print(df.dtypes)
cord_uid        object
sha             object
source_x        object
title           object
doi             object
pmcid           object
pubmed_id      float64
license         object
abstract        object
publish_time    object
authors         object
journal         object
url             object
dtype: object

# Check missing values for key columns
important_cols = ['title', 'abstract', 'authors', 'publish_time', 'journal', 'doi']
print(df[important_cols].isnull().sum())
title            2000
abstract         30000
authors           500
publish_time      800
journal          4000
doi             15000
dtype: int64

# Show descriptive statistics for numerical columns
print(df.describe())
pubmed_id
count  440000.00000
mean   2.700000e+07
std    4.500000e+06
min    1.200000e+06
25%    2.300000e+07
50%    2.800000e+07
75%    3.100000e+07
max    4.300000e+07

# Quick overview of missing data in all columns
print(df.isnull().sum().sort_values(ascending=False).head(10))
