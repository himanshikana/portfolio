import pandas as pd
import zipfile

def load_and_clean_data(zip_path, csv_filename):
    try:
        # Open ZIP and load CSV
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(csv_filename) as f:
                df = pd.read_csv(f)

        print("✅ Data loaded successfully!")
        print("Raw data shape:", df.shape)

        # Select columns to keep
        cols = ['title', 'year', 'date_published', 'metascore',
                'reviews_from_users', 'reviews_from_critics',
                'usa_gross_income', 'worlwide_gross_income']

        df = df[[c for c in cols if c in df.columns]]
        df = df.drop_duplicates()

        # Convert to numeric with errors coerced to NaN
        for col in ['year', 'metascore', 'reviews_from_users', 'reviews_from_critics',
                    'usa_gross_income', 'worlwide_gross_income']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where 'year' or 'title' is missing
        df = df.dropna(subset=['year', 'title'])
        df['year'] = df['year'].astype(int)

        # Fill NaNs in income columns with 0
        df['usa_gross_income'] = df['usa_gross_income'].fillna(0)
        df['worlwide_gross_income'] = df['worlwide_gross_income'].fillna(0)

        # Strip spaces in text columns
        df['title'] = df['title'].str.strip()
        if 'date_published' in df.columns:
            df['date_published'] = df['date_published'].astype(str).str.strip()

        df = df.reset_index(drop=True)

        print("✅ Data cleaning completed!")
        print("Cleaned data shape:", df.shape)
        print(df.info())
        print("\nHere is a preview of the cleaned data:\n")
        print(df.head())

        return df  # ✅ Properly indented inside the try block

    except Exception as e:
        print("❌ Error:", e)
        return None  # ✅ Properly indented under except block


if __name__ == "__main__":
    zip_path = r"C:\Users\HIMANSHI SHAH\OneDrive\Desktop\movie_project\IMDb movies.csv.zip"
    csv_filename = "IMDb movies.csv"

    df = load_and_clean_data(zip_path, csv_filename)

    if df is not None:
        output_path = r"C:\Users\HIMANSHI SHAH\OneDrive\Desktop\movie_project\cleaned_imdb_movies.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Cleaned CSV saved at:\n{output_path}")

        import pandas as pd

# Basic statistics
print(df.describe())

# Missing values count
print("\nMissing values:\n", df.isnull().sum())

# Data types
print("\nData types:\n", df.dtypes)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='year', order=sorted(df['year'].unique()))
plt.xticks(rotation=90)
plt.title("Number of Movies Released Per Year")
plt.xlabel("Year")
plt.ylabel("Movie Count")
plt.tight_layout()
plt.show()
top_metascore = df.sort_values(by='metascore', ascending=False).head(10)
print(top_metascore[['title', 'metascore']])
top_user_reviews = df.sort_values(by='reviews_from_users', ascending=False).head(10)
print(top_user_reviews[['title', 'reviews_from_users']])
top_critic_reviews = df.sort_values(by='reviews_from_critics', ascending=False).head(10)
print(top_critic_reviews[['title', 'reviews_from_critics']])
top_gross_world = df.sort_values(by='worlwide_gross_income', ascending=False).head(10)
print(top_gross_world[['title', 'worlwide_gross_income']])
plt.figure(figsize=(10, 5))
df.groupby('year')['worlwide_gross_income'].sum().plot(kind='line')
plt.title("Worldwide Gross Income Over Years")
plt.ylabel("Income (in billions?)")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.show()
sns.scatterplot(data=df, x='reviews_from_users', y='worlwide_gross_income')
plt.title("User Reviews vs Worldwide Gross Income")
plt.show()
sns.scatterplot(data=df, x='metascore', y='worlwide_gross_income')
plt.title("Metascore vs Worldwide Gross Income")
plt.show()
# Step 4: Derive Key Insights

def print_insights(df):
    print("🎯 KEY INSIGHTS FROM THE DATA\n")

    # 1. Year range
    min_year = df['year'].min()
    max_year = df['year'].max()
    print(f"📅 Movies in dataset range from {min_year} to {max_year}.\n")

    # 2. Most productive movie year
    top_year = df['year'].value_counts().idxmax()
    count_top_year = df['year'].value_counts().max()
    print(f"🎬 The year with the most movies released: {top_year} ({count_top_year} movies).\n")

    # 3. Top grossing movie
    top_gross_movie = df.sort_values(by='worlwide_gross_income', ascending=False).iloc[0]
    print(f"💰 Highest grossing movie worldwide: \"{top_gross_movie['title']}\" (${top_gross_movie['worlwide_gross_income']:.2f}).\n")

    # 4. Most critically acclaimed movie
    top_critic_movie = df[df['metascore'].notnull()].sort_values(by='metascore', ascending=False).iloc[0]
    print(f"⭐ Highest Metascore movie: \"{top_critic_movie['title']}\" (Metascore: {top_critic_movie['metascore']}).\n")

    # 5. Correlation between user reviews and revenue
    correlation = df[['reviews_from_users', 'worlwide_gross_income']].corr().iloc[0, 1]
    print(f"🔗 Correlation between user reviews and worldwide income: {correlation:.2f}.\n")

    # 6. Missing values summary
    missing = df.isnull().sum()
    print("🧼 Missing Data (after cleaning):")
    print(missing[missing > 0], "\n")

    print("✅ Insight generation complete.\n")

# Call the function
print_insights(df)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def advanced_analysis(df):
    print("\n🔎 Step 5: Advanced Analysis\n")

    # 1. Average user reviews per year
    avg_reviews_year = df.groupby('year')['reviews_from_users'].mean().dropna()
    print("Average User Reviews by Year (sample):")
    print(avg_reviews_year.head(), "\n")

    # 2. Average metascore per year
    avg_metascore_year = df.groupby('year')['metascore'].mean().dropna()
    print("Average Metascore by Year (sample):")
    print(avg_metascore_year.head(), "\n")

    # 3. Total worldwide gross income per decade
    df['decade'] = (df['year'] // 10) * 10
    income_by_decade = df.groupby('decade')['worlwide_gross_income'].sum()
    print("Total Worldwide Gross Income by Decade:")
    print(income_by_decade, "\n")

    # 4. Simple Linear Regression to predict worldwide gross income
    print("📈 Predicting Worldwide Gross Income using Metascore and User Reviews")

    # Select features and drop rows with missing values in those columns
    features = ['metascore', 'reviews_from_users', 'reviews_from_critics']
    data_model = df[features + ['worlwide_gross_income']].dropna()

    X = data_model[features]
    y = data_model['worlwide_gross_income']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Model evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.3f}\n")

    print("Model Coefficients:")
    for feat, coef in zip(features, model.coef_):
        print(f"  - {feat}: {coef:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")

    print("\n✅ Advanced analysis complete.\n")

# Call this function after your cleaning & visualization steps:
advanced_analysis(df)

# i reedited this code because my file was being save din git master









