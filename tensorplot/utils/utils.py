import seaborn as sns

def draw_line(data_df, x, y, hue):
    return sns.lineplot(data=data_df, x=x, y=y, hue=hue)