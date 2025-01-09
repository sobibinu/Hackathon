import pandas as pd
import os

# Define the path to the folder where the CSV files are stored
path = 'csv_folder/'

# Ensure the 'csv_folder' contains your 64 CSV files
csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]

# Read each CSV file into a list of DataFrames
df_list = [pd.read_csv(os.path.join(path, file)) for file in csv_files]

# Concatenate all DataFrames into one final DataFrame
final_df = pd.concat(df_list, ignore_index=True)

# Display the first few rows of the combined DataFrame
print(final_df.head())

# Step 1: Identify and Handle Missing Values

# Checking for missing values
print("Missing values per column before cleaning:")
print(final_df.isnull().sum())

# Handle missing values:
# - Fill missing numerical values with mean value (e.g., price)
final_df['price'].fillna(final_df['price'].mean(), inplace=True)

# - Fill missing categorical values with mode (e.g., location)
final_df['location'].fillna(final_df['location'].mode()[0], inplace=True)

# - Drop rows where 'property_id' is missing (essential field)
final_df.dropna(subset=['property_id'], inplace=True)

# Step 2: Handle Duplicates
print("\nDuplicates before dropping:")
print(final_df.duplicated().sum())  # Checking for duplicates

# Drop duplicates in the final dataset
final_df.drop_duplicates(inplace=True)

print("\nDuplicates removed")

# Step 3: Clean the 'photo_urls' column
# Extract the number of photos (assuming photo_urls column contains comma-separated URLs)

final_df['photo_count'] = final_df['photo_urls'].apply(
    lambda x: len(x.split(',')) if isinstance(x, str) else 0
)

# Preview of cleaned data (first 5 rows)
print("\nPreview of cleaned data:")
print(final_df.head())

# Step 1: Create the 'photo_count' feature (already done in Task 2, so assuming it's available in final_df)
# final_df['photo_count'] is already created from photo_urls

# Step 2: Load the 'property_interactions' dataset
interactions_df = pd.read_csv('property_interactions.csv')  # Assuming this is the file path for interactions data

# Preview the structure of property_interactions (for debugging/confirmation)
print(interactions_df.head())

# Step 3: Merge property data with interactions data (using property_id as the key)
final_df = final_df.merge(interactions_df[['property_id', 'interaction_count']], 
                           on='property_id', how='left')

# Step 4: Calculate total interactions for each property
# Assuming 'interaction_count' exists in the property_interactions dataset
final_df['total_interactions'] = final_df['interaction_count'].fillna(0)

# Step 5: Preview the DataFrame with new features
print(final_df[['property_id', 'photo_count', 'total_interactions']].head())

# Summary statistics for numeric columns
print(final_df.describe())
# Properties count per location
print(final_df['location'].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns

# Price distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(final_df['price'], kde=True, bins=20)
plt.title('Price Distribution of Properties')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
# Correlation heatmap for photo_count and price
corr_matrix = final_df[['price', 'photo_count', 'total_interactions']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
# Sorting properties by price to get the most expensive ones
top_expensive_properties = final_df.sort_values(by='price', ascending=False).head(10)
print(top_expensive_properties[['property_id', 'location', 'price']])
# Plotting total interactions vs. photo_count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='photo_count', y='total_interactions', data=final_df)
plt.title('Total Interactions vs. Photo Count')
plt.xlabel('Photo Count')
plt.ylabel('Total Interactions')
plt.show()
# Correlation between photo_count and total_interactions
photo_interaction_corr = final_df[['photo_count', 'total_interactions']].corr()
print(photo_interaction_corr)
# Boxplot of price vs location
plt.figure(figsize=(12, 8))
sns.boxplot(x='location', y='price', data=final_df)
plt.title('Price Distribution Across Locations')
plt.xticks(rotation=45)
plt.show()

