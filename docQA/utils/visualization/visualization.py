import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_fitting(
        data: dict,
        model_name: str,
        metric: str = 'loss',
        x_label: str = '',
        y_label: str = '',
        storage_path: str = ''
):
    dataframe = pd.DataFrame(data)
    plt.ioff()
    fig, ax = plt.subplots()
    sns.lineplot(data=dataframe, ax=ax)
    plt.title(model_name)
    ax.set(xlabel=x_label, ylabel=y_label)
    fig.savefig(f'{storage_path}/train_history/'+f'{model_name}_training_{metric}.png')
    plt.close()
