import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# Label Distribution
def plot_label_distribution(df, label_col="label", title="Label Distribution"):
    plt.figure(figsize=(10,5))
    sns.countplot(x=label_col, 
                data=df,
                order=df[label_col].value_counts().index, 
                palette="viridis"
                )
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# WordCloud per label
def plot_wordcloud(text_series, title="WordCloud"):
    text=" ".join(text_series.astype(str).values)
    wc=WordCloud(width=800,height=400, background_color="white").generate(text)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()

# Tweet length Distribution
def plot_length_distribution(df, text_col="text"):
    df["char_len"]=df[text_col].astype(str).apply(len)
    df["word_len"]=df[text_col].apply(lambda x: len(x.split()))

    fig,ax=plt.subplots(1,2, figsize=(14,6))
    sns.histplot(df["char_len"], bins=30, ax=ax[0], color="skyblue")
    ax[0].set_title("Tweet Character Length Distribution")
    sns.histplot(df["word_len"], bins=30, ax=ax[1], color="orange")
    ax[1].set_title("Tweet Word Count Distribution")
    plt.show()

# Confusion Matrix of ML Models

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    cm=confusion_matrix(y_true,y_pred,labels=labels)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    fig,ax=plt.subplots(figsize=(10,8))
    disp.plot(ax=ax,cmap="Blues",values_format="d",xticks_rotation=45)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()