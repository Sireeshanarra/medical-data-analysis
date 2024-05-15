import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Add an overweight column to the data
def add_overweight_column(df):
    df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad for cholesterol and gluc
def normalize_data(df):
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

# Draw the Categorical Plot
def draw_cat_plot(df):
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat['value'] = df_cat['value'].astype(int)  # Ensure 'value' column is of integer type
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig
    plt.show()  # Display the plot
    return fig

# Draw the Heat Map
def draw_heat_map(df):
    df_heat = df[(df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975)) &
                 (df['ap_lo'] <= df['ap_hi'])]
    corr = df_heat.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='coolwarm', mask=mask, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
    plt.show()  # Display the plot
    return fig

# Import the data from medical_examination.csv
def import_data(file_path):
    return pd.read_csv(file_path)

# Call the function with the file path
df = import_data("C:/Python34/medical_examination.csv")

# Main function
def main():

    # Task 2: Add overweight column
    add_overweight_column(df)

    # Task 3: Normalize data
    normalize_data(df)

    # Task 4: Draw Categorical Plot
    draw_cat_plot(df)

    # Task 5: Draw Heat Map
    draw_heat_map(df)

# Call the main function
if __name__ == "__main__":
    main()
 
