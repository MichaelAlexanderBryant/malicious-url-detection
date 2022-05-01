import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

#plot colors: [grey, blue, orange, green, pink, brown, purple, yellow, red]
light = ['#8c8c8c', '#88bde6', '#fbb258', '#90cd97', '#f6aac8', '#bfa454', '#bc99c7', '#eddd46', '#f07d6e']
medium = ['#4d4d4d', '#5da6da', '#faa43a', '#60bd68', '#f17cb1', '#b2912f', '#b276b2', '#dece3f', '#f15954']
dark_bright = ['#000000', '#265dab', '#df5c24', '#059749', '#e5126d', '#9d732a', '#7a3a96', '#c7b52e', '#cb2026']

#grid/rule lines: light grey
grid = ['#e0e0e0']

#plot parameters
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.top"] = True

df_no_sampling = pd.read_csv('../output/modeling/no_sampling/df_scores.csv')
df_upsampling = pd.read_csv('../output/modeling/upsampling/df_scores.csv')
df_downsampling = pd.read_csv('../output/modeling/downsampling/df_scores.csv')

df = pd.concat([df_no_sampling, df_upsampling, df_downsampling],
               axis=0).reset_index().drop('index', axis=1)


for idx, val in enumerate(df.columns[2:]):
    sorted_index_descent = df.groupby(['Model']).max().sort_values(by= val,ascending=False).index
    
    df = df.sort_values(by=['Model', 'Sampling',val], ascending=True)
    
    plt.figure(figsize=(10, 10))
    #ax = sns.scatterplot(data = df, y = 'Model', x = val, hue = 'Sampling', style='Sampling', s=225, linewidth=0, alpha=0.75, markers=('o','s','^'), palette=[medium[1],medium[8],medium[3]])
    #ax = sns.stripplot(y = df['Model'], x = df[val], hue = df['Sampling'], s=15, jitter=False, order=sorted_index_descent, alpha=0.75, palette=[medium[1],medium[8],medium[3]])
    ax = sns.catplot(data = df, y = 'Model', x = val, hue = 'Sampling', kind='bar')
    ax.set(xlim=(0,1))
    # ax.set_ylabel('')
    # ax.set_xlabel(val)
    # ax.set_xlim([-.1,1.1])
    # ax.spines["bottom"].set_color(dark_bright[0])
    # ax.spines["left"].set_color(dark_bright[0])
    # ax.spines["top"].set_color(dark_bright[0])
    # ax.spines["right"].set_color(dark_bright[0])
    # ax.xaxis.label.set_color(dark_bright[0])
    # ax.yaxis.label.set_color(dark_bright[0])
    # ax.title.set_color(dark_bright[0])
    # ax.tick_params(axis='x', colors = light[0], labelcolor=dark_bright[0])
    # ax.tick_params(axis='y', colors = light[0], labelcolor=dark_bright[0])
    # ax.yaxis.set_ticks_position('none')
    # ax.grid(color = 'w', axis='x')
    # ax.grid(color = grid[0], axis='y')
    # ax.legend(loc="upper left")    
    # ax.set_axisbelow(True)
    plt.savefig('../output/modeling/model_comparison/catplot_model_{}.jpg'.format(val), bbox_inches='tight')
    plt.show()