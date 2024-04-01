import os
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import json

figure(figsize=(20, 6), dpi=1000)

tokenizer = AutoTokenizer.from_pretrained("pretrained_models/bart-large-cnn")
# cmap = sns.diverging_palette(200,20,sep=20,as_cmap=True)

data = load_from_disk("dataset/ami_map")
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
# plt.subplots(1, 3, fig_kw={})
cmap = "RdBu_r"
for i, item in enumerate(data):

    his_ids = item["history_ids"]
    res_ids = item["response_ids"]
    map = item["grad_attn_map"]

    ratio = int(len(his_ids) / len(res_ids))
    if ratio > 100:
        continue
    fig, ax = plt.subplots(1, 3, figsize=(80, 10))
    plt.subplot(131)
    
    x_ticks = [tokenizer.convert_ids_to_tokens(it).replace("Ġ", "") for it in his_ids]
    y_ticks = [tokenizer.convert_ids_to_tokens(it).replace("Ġ", "") for it in res_ids]
    # x_ticks = tokenizer.decode(his_ids).split()
    # y_ticks = tokenizer.decode(res_ids).split()
    ax = sns.heatmap(map, cmap=cmap, xticklabels=x_ticks, yticklabels=y_ticks, label="Attn*Grad")
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=6)
    plt.title("Attn*Grad", fontsize=12)

    plt.subplot(132)
    map = item["attn_map"]
    
    x_ticks = [tokenizer.convert_ids_to_tokens(it).replace("Ġ", "") for it in his_ids]
    y_ticks = [tokenizer.convert_ids_to_tokens(it).replace("Ġ", "") for it in res_ids]
    # x_ticks = tokenizer.decode(his_ids).split()
    # y_ticks = tokenizer.decode(res_ids).split()
    ax = sns.heatmap(map, cmap=cmap, xticklabels=x_ticks, yticklabels=y_ticks, label="Attn")
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=180)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=6)
    plt.title("Attn", fontsize=12)

    plt.subplot(133)
    map = item["grad_map"]
    
    x_ticks = [tokenizer.convert_ids_to_tokens(it).replace("Ġ", "") for it in his_ids]
    y_ticks = [tokenizer.convert_ids_to_tokens(it).replace("Ġ", "") for it in res_ids]
    # x_ticks = tokenizer.decode(his_ids).split()
    # y_ticks = tokenizer.decode(res_ids).split()
    ax = sns.heatmap(map, cmap=cmap, xticklabels=x_ticks, yticklabels=y_ticks, label="Grad")
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=-90)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=6)
    plt.title("Grad", fontsize=12)


    plt.yticks(fontproperties = 'Times New Roman', size = 6)
    plt.xticks(fontproperties = 'Times New Roman', size = 6)
    fig.tight_layout()
    if not os.path.exists("./plot_results"):
        os.mkdir("./plot_results")
    plt.savefig("./plot_results/{}.jpg".format(i), dpi=400)
    plt.close()

    if i > 5000:
        exit()

