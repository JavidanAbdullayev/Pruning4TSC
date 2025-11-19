import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon

def plot_1v1_perf(res_df, p_value, x_y_lim=0, acc_base=1, co_col='ens_co', base_col='base_ens', xlabel='base perf.', ylabel='CoTrain perf.', legend_base='Base', legend_co='CoTrain', title='Title', file_name=None):
    # Define the points for the diagonal line
    x_line = [x_y_lim, acc_base]
    y_line = [x_y_lim, acc_base]

    # Define points for scatter plot
    x_scatter = res_df[base_col].tolist() 
    y_scatter = res_df[co_col].tolist() 

    x_above = np.array([x for x, y in zip(x_scatter, y_scatter) if y > x])
    y_above = np.array([y for x, y in zip(x_scatter, y_scatter) if y > x])

    x_same = np.array([x for x, y in zip(x_scatter, y_scatter) if y == x])
    y_same = np.array([y for x, y in zip(x_scatter, y_scatter) if y == x])

    x_below = np.array([x for x, y in zip(x_scatter, y_scatter) if y < x])
    y_below = np.array([y for x, y in zip(x_scatter, y_scatter) if y < x])

    # Plot the diagonal line
    plt.plot(x_line, y_line,  color='blue')
    num_wins = res_df[res_df[co_col] > res_df[base_col]].shape[0]
    num_ties = res_df[res_df[co_col] == res_df[base_col]].shape[0]
    num_losses = res_df[res_df[co_col] < res_df[base_col]].shape[0]

    # Add p-value to the legend
    plt.scatter([], [], label=f'p-value: {p_value}', color='none')  # Invisible point for legend entry

    # Plot the scatter points
    plt.scatter(x_below, y_below, label=f'{legend_base} Wins - ' + str(num_losses), color='green')
    plt.scatter(x_same, y_same, label='Equal - ' + str(num_ties), color='orange')
    plt.scatter(x_above, y_above, label=f'{legend_co} Wins - ' + str(num_wins), color='red')

    # # Set axis limits
    plt.xlim(x_y_lim, acc_base)
    plt.ylim(x_y_lim, acc_base)

    plt.xlabel(xlabel, fontsize=14)  
    plt.ylabel(ylabel, fontsize=14)  
    plt.title(title, fontsize=16)    


    # Add a legend
    plt.legend()
    if file_name:
        plt.savefig(f'{file_name}.pdf', bbox_inches='tight', pad_inches=0)

    plt.show()
    
    
    
    
df = pd.read_csv(os.getcwd() + '/results.csv')


# Perform the Wilcoxon signed-rank test
stat, p_value = wilcoxon(df['base_ens_acc'].tolist(), df['finetune_ens_acc'])
print('p_value: ', p_value)

plot_1v1_perf(df, 
              np.round(p_value, 5), 
              x_y_lim=0.1, 
              acc_base=1.01, 
              co_col='finetune_ens_acc', 
              base_col='base_ens_acc', 
              xlabel='LITETime', 
              ylabel='Pruned + Finetuned LITE Ens.', 
              legend_base='Base', 
              legend_co='Pruned', 
              title='LITETime vs Pruned + Finetuned LITE Ens.', 
              file_name='fig_res')