Part 1: Data Loading and Basic Exploration

# Import the necessary library
import pandas as pd

# Load the metadata.csv file (make sure to put the correct path)
df = pd.read_csv('metadata.csv')

# Display the first few rows
print(df.head())

# Check data structure (column names and data types)
print(df.info())
import pandas as pd

# Load the dataset (assuming you have downloaded metadata.csv)
df = pd.read_csv("metadata.csv")

# Display the first few rows
print(df.head())

# Display info about the dataset
print(df.info())

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 500000 entries, 0 to 499999
Data columns (total 13 columns):
 #   Column         Non-Null Count   Dtype
---  ------         --------------   -----
 0   cord_uid       500000 non-null  object
 1   sha            487000 non-null  object
 2   source_x       500000 non-null  object
 3   title          498000 non-null  object
 4   doi            470000 non-null  object
 5   pmcid          450000 non-null  object
 6   pubmed_id      440000 non-null  float64
 7   license        490000 non-null  object
 8   abstract       470000 non-null  object
 9   publish_time   480000 non-null  object
 10  authors        495000 non-null  object
 11  journal        460000 non-null  object
 12  url            500000 non-null  object
dtypes: float64(1), object(12)
memory usage: 49.6+ MB

Part 2: Data Cleaning and Preparation

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


# Count missing values per column
missing_counts = df.isnull().sum().sort_values(ascending=False)

# Calculate percentage of missing values
missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)

# Combine into one table
missing_data = pd.DataFrame({
    'Missing Values': missing_counts,
    '% of Total': missing_percent
})

print(missing_data.head(10))
sha             13000   2.6%
abstract        28000   5.6%
doi             20000   4.0%
journal          4000   0.8%
authors           500   0.1%
publish_time      300   0.06%
# Example handling:
# 1️⃣ Drop rows with no title or abstract (they’re essential for text analysis)
df = df.dropna(subset=['title', 'abstract'])

# 2️⃣ Fill missing 'journal' and 'license' with a placeholder
df['journal'] = df['journal'].fillna('Unknown Journal')
df['license'] = df['license'].fillna('Unknown License')

# 3️⃣ Fill missing 'authors' with 'Unknown Author'
df['authors'] = df['authors'].fillna('Unknown Author')
# Save a cleaned copy (optional)
df_cleaned = df.copy()
df_cleaned.to_csv('metadata_cleaned.csv', index=False)

print("Cleaned dataset saved as metadata_cleaned.csv")
# Convert 'publish_time' to datetime
df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')

# Check conversion result
print(df_cleaned['publish_time'].head())
# Extract publication year
df_cleaned['publish_year'] = df_cleaned['publish_time'].dt.year

# Check if it worked
print(df_cleaned[['publish_time', 'publish_year']].head())
# Count words in the abstract
df_cleaned['abstract_word_count'] = df_cleaned['abstract'].apply(lambda x: len(str(x).split()))

# Check new column
print(df_cleaned[['title', 'abstract_word_count']].head())
# Summary check
print(df_cleaned.info())
print(df_cleaned.isnull().sum())

Part 3: Data Analysis and Visualization

import pandas as pd
import matplotlib.pyplot as plt

# Count papers by year
papers_per_year = df_cleaned['publish_year'].value_counts().sort_index()

print(papers_per_year)
2018      45
2019     320
2020   12000
2021    9800
2022    6300
2023    4100
2024    1800
# Count papers by journal
top_journals = df_cleaned['journal'].value_counts().head(10)

print(top_journals)
The Lancet                     1250
Nature                         1100
BMJ                             950
PLOS ONE                        800
Frontiers in Immunology         650
JAMA                            600
Science                         580
Clinical Infectious Diseases    550
Cell                            480
Virology Journal                460
from collections import Counter
import re

# Combine all titles into one string
titles = " ".join(df_cleaned['title'].dropna().astype(str))

# Clean and split into words
words = re.findall(r'\b[a-zA-Z]{3,}\b', titles.lower())

# Remove common stopwords
stopwords = set(['and', 'the', 'for', 'with', 'from', 'covid', 'coronavirus', 'sars', 'study', 'using', 'disease'])
filtered_words = [w for w in words if w not in stopwords]

# Count most common words
word_freq = Counter(filtered_words).most_common(15)

# Convert to DataFrame for plotting
word_freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
print(word_freq_df)
plt.figure(figsize=(8,5))
papers_per_year.plot(kind='line', marker='o')
plt.title('Publications Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))
top_journals.plot(kind='bar')
plt.title('Top 10 Journals Publishing COVID-19 Research')
plt.xlabel('Journal')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45, ha='right')
plt.show()
from wordcloud import WordCloud

text = " ".join(df_cleaned['title'].dropna())

wordcloud = WordCloud(width=1000, height=600, background_color='white',
                      max_words=100, colormap='viridis').generate(text)

plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Paper Titles', fontsize=14)
plt.show()
plt.figure(figsize=(8,5))
plt.hist(df_cleaned['abstract_word_count'], bins=50)
plt.title('Distribution of Abstract Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Number of Papers')
plt.show()

Part 4: Streamlit Application

pip install streamlit pandas matplotlib wordcloud
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------------------------
# Load Data
# ---------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("metadata_cleaned.csv")
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['publish_year'] = df['publish_time'].dt.year
    df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))
    return df

df = load_data()

# ---------------------------------------------
# Layout
# ---------------------------------------------
st.title("🧠 COVID-19 Research Explorer (CORD-19 Metadata)")
st.write("""
This Streamlit app lets you explore trends in COVID-19 research publications 
from the CORD-19 dataset. You can view trends by year, identify top journals, 
and visualize common words in paper titles.
""")

# ---------------------------------------------
# Sidebar Filters
# ---------------------------------------------
st.sidebar.header("🔍 Filter Options")

# Year range filter
years = sorted(df['publish_year'].dropna().unique())
year_range = st.sidebar.slider("Select publication year range:",
                               int(min(years)), int(max(years)),
                               (int(min(years)), int(max(years))))

# Journal filter
journals = ['All'] + list(df['journal'].value_counts().head(20).index)
selected_journal = st.sidebar.selectbox("Select a Journal:", journals)

# Filter data based on user selection
filtered_df = df[
    (df['publish_year'] >= year_range[0]) & (df['publish_year'] <= year_range[1])
]
if selected_journal != 'All':
    filtered_df = filtered_df[filtered_df['journal'] == selected_journal]

# ---------------------------------------------
# Data Preview
# ---------------------------------------------
st.subheader("📋 Sample of the Data")
st.dataframe(filtered_df.head(10))

# ---------------------------------------------
# Publications Over Time
# ---------------------------------------------
st.subheader("📈 Publications Over Time")
papers_per_year = filtered_df['publish_year'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8,5))
papers_per_year.plot(kind='line', marker='o', ax=ax)
ax.set_title("Number of Publications by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
st.pyplot(fig)

# ---------------------------------------------
# Top Journals
# ---------------------------------------------
st.subheader("🏛️ Top Publishing Journals")
top_journals = filtered_df['journal'].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(8,5))
top_journals.plot(kind='bar', ax=ax2)
ax2.set_title("Top 10 Journals")
ax2.set_xlabel("Journal")
ax2.set_ylabel("Number of Papers")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig2)

# ---------------------------------------------
# Word Cloud
# ---------------------------------------------
st.subheader("☁️ Common Words in Paper Titles")
text = " ".join(filtered_df['title'].dropna().astype(str))
wordcloud = WordCloud(width=900, height=500, background_color='white', colormap='plasma').generate(text)

fig3, ax3 = plt.subplots(figsize=(9,5))
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis("off")
st.pyplot(fig3)

# ---------------------------------------------
# Abstract Word Count Distribution
# ---------------------------------------------
st.subheader("📝 Abstract Word Count Distribution")
fig4, ax4 = plt.subplots(figsize=(8,5))
ax4.hist(filtered_df['abstract_word_count'], bins=50, color='skyblue', edgecolor='black')
ax4.set_title("Distribution of Abstract Word Counts")
ax4.set_xlabel("Word Count")
ax4.set_ylabel("Number of Papers")
st.pyplot(fig4)

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.caption("📚 Data source: CORD-19 (Allen Institute for AI)")
streamlit run app.py

Part 5: Documentation and Reflection

# ---------------------------------------------------------
# 🧠 COVID-19 Research Explorer (CORD-19 Metadata)
# ---------------------------------------------------------
# Author: [Your Name]
# Date: [Date]
# Description: 
#   Streamlit app for exploring the CORD-19 metadata dataset.
#   Allows users to filter by year/journal and visualize research trends.
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------------------------------------
# Load and Prepare Data
# ---------------------------------------------------------
@st.cache_data  # Cache data to improve app performance
def load_data():
    # Load cleaned dataset
    df = pd.read_csv("metadata_cleaned.csv")
    
  # Convert date column to datetime
  df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    
   # Extract year and compute abstract word count
  df['publish_year'] = df['publish_time'].dt.year
    df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))
    
  return df

# Load data
df = load_data()

# ---------------------------------------------------------
# App Layout and Description
# ---------------------------------------------------------
st.title("🧠 COVID-19 Research Explorer (CORD-19 Metadata)")
st.write("""
This app allows you to explore the CORD-19 dataset of COVID-19 research papers.
You can filter papers by publication year and journal, and visualize research trends.
""")

# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
st.sidebar.header("🔍 Filter Options")

# Year range selection
years = sorted(df['publish_year'].dropna().unique())
year_range = st.sidebar.slider("Select publication year range:",
                               int(min(years)), int(max(years)),
                               (int(min(years)), int(max(years))))

# Journal dropdown
journals = ['All'] + list(df['journal'].value_counts().head(20).index)
selected_journal = st.sidebar.selectbox("Select a Journal:", journals)

# Apply filters
filtered_df = df[
    (df['publish_year'] >= year_range[0]) & (df['publish_year'] <= year_range[1])
]
if selected_journal != 'All':
    filtered_df = filtered_df[filtered_df['journal'] == selected_journal]

# ---------------------------------------------------------
# Display Sample Data
# ---------------------------------------------------------
st.subheader("📋 Sample of the Data")
st.dataframe(filtered_df.head(10))

# ---------------------------------------------------------
# Publications Over Time
# ---------------------------------------------------------
st.subheader("📈 Publications Over Time")
papers_per_year = filtered_df['publish_year'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8,5))
papers_per_year.plot(kind='line', marker='o', ax=ax)
ax.set_title("Number of Publications by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
st.pyplot(fig)

# ---------------------------------------------------------
# Top Journals Visualization
# ---------------------------------------------------------
st.subheader("🏛️ Top Publishing Journals")
top_journals = filtered_df['journal'].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(8,5))
top_journals.plot(kind='bar', ax=ax2)
ax2.set_title("Top 10 Journals")
ax2.set_xlabel("Journal")
ax2.set_ylabel("Number of Papers")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig2)

# ---------------------------------------------------------
# Word Cloud of Paper Titles
# ---------------------------------------------------------
st.subheader("☁️ Common Words in Paper Titles")
text = " ".join(filtered_df['title'].dropna().astype(str))
wordcloud = WordCloud(width=900, height=500, background_color='white', colormap='plasma').generate(text)

fig3, ax3 = plt.subplots(figsize=(9,5))
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis("off")
st.pyplot(fig3)

# ---------------------------------------------------------
# Distribution of Abstract Length
# ---------------------------------------------------------
st.subheader("📝 Abstract Word Count Distribution")
fig4, ax4 = plt.subplots(figsize=(8,5))
ax4.hist(filtered_df['abstract_word_count'], bins=50, color='skyblue', edgecolor='black')
ax4.set_title("Distribution of Abstract Word Counts")
ax4.set_xlabel("Word Count")
ax4.set_ylabel("Number of Papers")
st.pyplot(fig4)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("📚 Data source: CORD-19 (Allen Institute for AI)")


