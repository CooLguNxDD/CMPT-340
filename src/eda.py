import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Reference https://www.kaggle.com/raniaioan/starter-skin-cancer-mnist-ham10000-6a5a3b01-0
def plotScatterMatrix(df, plotSize, textSize, filePath):
    # keep only numerical columns
    df = df.select_dtypes(include =[np.number]) 
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    # keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]] 
    columnNames = list(df)
    # reduce the number of columns for matrix inversion of kernel density plots
    if len(columnNames) > 10: 
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.savefig(filePath)
    plt.show()


def main():
    # Load data
    df = pd.read_csv('../data/HAM10000_metadata.csv')
    rgb_df = pd.read_csv('../data/hmnist_28_28_RGB.csv')

    # Missing values
    ax = sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
    fig = ax.get_figure()
    plt.title('Missing Values Heatmap')
    fig.savefig('../plots/missing_values_heatmap.png')

    # Data visulization
    # Age Distribution by Gender
    ax = sns.boxplot(x='sex', y='age', data=df)
    fig = ax.get_figure()    
    plt.title('Age Distribution by Gender')
    fig.savefig('../plots/age_distribution_by_gender.png')

    # Cases by Diagnostic Category
    count_by_category = df.groupby(['dx']).size().reset_index(name='count')
    fig = plt.figure()
    plt.bar(count_by_category['dx'], count_by_category['count'], align='center')
    plt.title('Cases by Diagnostic Category')
    plt.xlabel('Diagnostic Category')
    plt.ylabel('Cases')
    fig.savefig('../plots/cases_by_diagnostic_category.png')

    # Cases by Localization
    count_by_localization = df.groupby(['localization']).size().reset_index(name='count')
    fig = plt.figure()
    plt.bar(count_by_localization['localization'], count_by_localization['count'], align='center')
    plt.title('Cases by Localization')
    plt.xlabel('Localization')
    plt.xticks(rotation='vertical')
    plt.ylabel('Cases')
    fig.savefig('../plots/cases_by_localization.png')

    # Cases by Diagnostic Category and Confirmation Type
    count_by_dx_dxtype = df.groupby(['dx','dx_type']).size().reset_index(name='count')
    count_by_dx_dxtype = count_by_dx_dxtype.pivot_table(values='count', index='dx', columns='dx_type')
    barplot = count_by_dx_dxtype.plot.bar()
    fig = barplot.get_figure()
    plt.title('Cases by Diagnostic Category and Confirmation Type')
    plt.xlabel('Diagnostic Category')
    plt.xticks(rotation='horizontal')
    plt.ylabel('Cases')
    fig.savefig("../plots/cases_by_diagnostic_category_and_confirmation_type.png")

    # Cases by Age Group and Gender
    bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89]
    labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]
    df['age_bin'] = pd.cut(x=df['age'], bins=bins, labels=labels, right=False)
    count_by_age_sex = df.groupby(['age_bin','sex']).size().reset_index(name='count')
    count_by_age_sex = count_by_age_sex.pivot_table(values='count', index='age_bin', columns='sex')
    barplot = count_by_age_sex.plot.bar()
    fig = barplot.get_figure()
    plt.title('Cases by Age Group and Gender')
    plt.xlabel('Age Group')
    plt.xticks(rotation='horizontal')
    plt.ylabel('Cases')
    fig.savefig("../plots/cases_by_age_group_and_gender.png")

    # Cases by Age Group and Diagnostic Category
    diagnostic_categories = df['dx'].unique()
    count_by_age_dx = df.groupby(['age_bin','dx']).size().reset_index(name='count')
    colors = ['blue', 'red', 'yellow', 'green', 'black', 'brown', 'purple']

    for i in range(len(diagnostic_categories)):
        plt.plot('age_bin', 'count', data=count_by_age_dx[count_by_age_dx['dx']==diagnostic_categories[i]], c=colors[i])

    plt.title('Cases by Age Group and Diagnostic Category')
    plt.ylabel('Cases')
    plt.legend(diagnostic_categories)
    fig.savefig("../plots/cases_by_age_group_and_diagnostic_category.png")

    # Scatter matrix
    rgb_df = pd.read_csv('../data/hmnist_28_28_RGB.csv')
    rgb_df.head()

    plotScatterMatrix(rgb_df, 20, 10, "../plots/scatter_matrix.jpg")


if __name__ == "__main__":
    main()
