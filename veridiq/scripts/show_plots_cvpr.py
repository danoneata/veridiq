import matplotlib.pyplot as plt
import streamlit as st
import matplotlib as mpl
import numpy as np
import seaborn as sns


sns.set(style="whitegrid", context="talk", font="Arial")

RESULTS_DFE_TO_DFE = """
AV-HuBERT (A)     & 65.81
Auto-AVSR         & 63.53
Wav2Vec2 XLS-R 2B & 58.71
AV-HuBERT (V)     & 72.06
Auto-AVSR (VSR)   & 64.26
FSFM              & 71.72
CLIP VIT-L/14     & 73.90
Video-MAE-large   & 54.48
"""

RESULTS_FAVC_TO_DFE = """
AV-HuBERT (A)     & 49.05
Auto-AVSR (ASR)   & 49.41
Wav2Vec2 XLS-R 2B & 62.30
AV-HuBERT (V)     & 63.68
Auto-AVSR (VSR)   & 48.70
FSFM              & 71.80
CLIP VIT-L/14     & 55.54
Video-MAE-large   & 45.64
"""

RESULTS_FAVC_TO_FAVC = """
AV-HuBERT (A)     & 100.00
Auto-AVSR (ASR)   & 99.66     
Wav2Vec2          & 100.00
AV-HuBERT (V)     & 100.00
Auto-AVSR (VSR)   & 97.83     
FSFM              & 97.13     
CLIP VIT-L/14     & 99.77     
Video-MAE-large   & 99.96     
"""

RESULTS_DET_LOC = """
AV-HuBERT (A) (random)   & 98.8  & 54.4
AV-HuBERT (A)            & 99.95 & 92.00
Auto-AVSR (ASR)          & 96.38 & 93.72
Wav2Vec2 XLS-R 2B        & 99.97 & 96.81
AV-HuBERT (V) (random)   & 50.2  & 48.4
AV-HuBERT (V)            & 93.67 & 94.53
Auto-AVSR (VSR)          & 58.98 & 65.14
FSFM                     & 95.25 & 52.68
CLIP VIT-L/14            & 96.54 & 80.17
Video-MAE-large          & 99.80 & 78.54
AV-HuBERT                & 99.88 & 96.80
Auto-AVSR                & 91.58 & 90.20
"""
# AV-HuBERT (random init.) & 51.32 & 51.45

MAIN_TABLE = """
AV-HuBERT (A) (random) & 99.8 & 97.8 & 98.8 & 92.0 & 50.9 & 51.9 & 53.4 & 41.9 & 46.4 & 63.5
AV-HuBERT (A)          & 100. & 100  & 100  & 99.0 & 50.0 & 57.2 & 65.8 & 49.1 & 48.3 & 67.3
Auto-AVSR (ASR)        & 99.7 & 76.0 & 96.4 & 50.3 & 52.9 & 49.6 & 63.5 & 49.4 & 47.5 & 54.3
Wav2Vec2 XLS-R 2B      & 100  & 99.9 & 100  & 96.6 & 51.3 & 56.3 & 58.7 & 62.3 & 58.6 & 70.8
AV-HuBERT (V) (random) & 83.8 & 52.5 & 50.2 & 49.3 & 60.4 & 53.8 & 56.8 & 60.6 & 54.0 & 55.1
AV-HuBERT (V)          & 100  & 95.5 & 93.7 & 64.1 & 98.3 & 90.5 & 72.1 & 63.7 & 67.7 & 80.0
Auto-AVSR (VSR)        & 97.8 & 77.5 & 59.0 & 51.3 & 83.3 & 70.1 & 64.3 & 48.7 & 56.1 & 64.5
FSFM                   & 97.1 & 40.9 & 95.3 & 52.7 & 84.3 & 36.8 & 71.7 & 71.8 & 43.5 & 55.0
CLIP VIT-L/14          & 99.8 & 95.2 & 96.5 & 71.1 & 60.3 & 53.3 & 73.9 & 55.6 & 43.5 & 63.2
Video-MAE-large        & 100  & 70.4 & 99.8 & 60.0 & 71.3 & 47.2 & 54.5 & 45.6 & 39.3 & 55.6
AV-HuBERT              & 100  & 99.5 & 99.9 & 94.5 & 78.5 & 84.4 & 70.4 & 58.2 & 54.3 & 78.2
Auto-AVSR              & 94.7 & 68.3 & 91.6 & 53.2 & 59.6 & 54.6 & 61.2 & 43.0 & 49.2 & 54.7
"""

MODEL_SHORT = {
    "AV-HuBERT (A) (random)": "AV-H (A) rand",
    "AV-HuBERT (A)": "AV-H (A)",
    "Auto-AVSR (ASR)": "AVSR (A)",
    "Wav2Vec2 XLS-R 2B": "W2V2",
    "AV-HuBERT (V) (random)": "AV-H (V) rand",
    "AV-HuBERT (V)": "AV-H (V)",
    "Auto-AVSR (VSR)": "AVSR (V)",
    "FSFM": "FSFM",
    "CLIP VIT-L/14": "CLIP",
    "Video-MAE-large": "Video-MAE",
    "AV-HuBERT (random init.)": "AV-H (rand)",
    "AV-HuBERT": "AV-H",
    "Auto-AVSR": "AVSR",
}

MODEL_MODALITY = {
    "AV-HuBERT (A) (random)": "A",
    "AV-HuBERT (A)": "A",
    "Auto-AVSR (ASR)": "A",
    "Wav2Vec2 XLS-R 2B": "A",
    "AV-HuBERT (V) (random)": "V",
    "AV-HuBERT (V)": "V",
    "Auto-AVSR (VSR)": "V",
    "FSFM": "V",
    "CLIP VIT-L/14": "V",
    "Video-MAE-large": "V",
    "AV-HuBERT (random init.)": "AV",
    "AV-HuBERT": "AV",
    "Auto-AVSR": "AV",
}


def format_results(results_str):
    def parse_line(line):
        model, *scores = line.split("&")
        scores = [float(s.strip()) for s in scores]
        return model.strip(), *scores

    return [parse_line(line) for line in results_str.strip().split("\n")]


def make_bar_plot():
    results_dfe_to_dfe = format_results(RESULTS_DFE_TO_DFE)
    results_favc_to_dfe = format_results(RESULTS_FAVC_TO_DFE)
    results_favc_to_favc = format_results(RESULTS_FAVC_TO_FAVC)

    models = [m for m, _ in results_dfe_to_dfe]
    id_scores = [s for _, s in results_dfe_to_dfe]
    ood_scores = [s for _, s in results_favc_to_dfe]
    favc_scores = [s for _, s in results_favc_to_favc]

    x = range(len(models))
    width = 0.25

    MODALITY_COLOR_IDS = {"A": 0, "V": 1, "AV": 2}
    COLORPALLETES = [
        sns.color_palette("muted"),
        sns.color_palette(),
        sns.color_palette("dark"),
    ]

    colors_ood = [COLORPALLETES[0][MODALITY_COLOR_IDS[MODEL_MODALITY[m]]] for m in models]
    colors_id = [COLORPALLETES[1][MODALITY_COLOR_IDS[MODEL_MODALITY[m]]] for m in models]
    colors_favc = [COLORPALLETES[2][MODALITY_COLOR_IDS[MODEL_MODALITY[m]]] for m in models]

    fig, ax = plt.subplots(figsize=(10, 3))
    bars1 = ax.bar([i - width for i in x], ood_scores, width, color=colors_ood)
    bars2 = ax.bar([i for i in x], id_scores, width, color=colors_id)
    bars3 = ax.bar([i + width for i in x], favc_scores, width, color=colors_favc)

    ax.legend(
        handles=[bars1[0], bars2[0], bars3[0]],
        labels=["FAVC → DFE", "DFE → DFE", "FAVC → FAVC"],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )

    ax.set_ylabel("AUC (%)")
    model_names = [MODEL_SHORT.get(m, m) for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=0, ha="center")

    for bar in list(bars1) + list(bars2) + list(bars3):
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig("output/plots/dfe-2024.pdf", bbox_inches="tight")
    return fig


def make_loc_plot():
    results_det_loc = format_results(RESULTS_DET_LOC)
    models, results_det, results_loc = zip(*results_det_loc)

    width = 0.4
    x = range(len(models))

    # colorpalettes = {
    #     "A": sns.color_palette("muted"),
    #     "V": sns.color_palette("colorblind"),
    #     "AV": sns.color_palette("dark"),
    # }
    # colors1 = [colorpalettes[MODEL_MODALITY[m]][0] for m in models]
    # colors2 = [colorpalettes[MODEL_MODALITY[m]][1] for m in models]

    colorpalette = sns.color_palette("Paired")
    def get_index(model_name):
        indices = ["A", "V", "AV"]
        return 2 * indices.index(MODEL_MODALITY[model_name])

    colors1 = [colorpalette[get_index(m)] for m in models]
    colors2 = [colorpalette[get_index(m) + 1] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar([i - width for i in x], results_det, width, color=colors1)
    bars2 = ax.bar([i for i in x], results_loc, width, color=colors2, hatch="/")

    ax.legend(
        handles=[bars1[0], bars2[0]],
        labels=["Classification", "Localization"],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )

    ax.set_ylabel("AUC (%)")
    model_names = [MODEL_SHORT.get(m, m) for m in models]
    xx = [x1 - width / 2 for x1 in x]
    ax.set_xticks(xx)
    # ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_xticklabels(model_names, rotation=90, ha="center")

    # for bar in list(bars1) + list(bars2):
    #     h = bar.get_height()
    #     ax.annotate(
    #         f"{h:.1f}",
    #         xy=(bar.get_x() + bar.get_width() / 2, h),
    #         xytext=(0, 3),
    #         textcoords="offset points",
    #         ha="center",
    #         va="bottom",
    #         fontsize=10,
    #     )

    fig.tight_layout()
    # fig.savefig("output/plots/det-vs-loc.pdf", bbox_inches="tight", transparent=True)
    fig.savefig("output/plots/det-vs-loc.png", bbox_inches="tight", transparent=True)
    return fig

def make_loc_plot_2():
    """AI4TRUST report"""
    results_det_loc = format_results(RESULTS_DET_LOC)
    models, results_det, _ = zip(*results_det_loc)
    results_det_ood = format_results(MAIN_TABLE)
    results_det_ood_all = list(zip(*results_det_ood))
    results_det_ood_all = [
        results_det_ood_all[2],
        results_det_ood_all[6],
        results_det_ood_all[9],
    ]
    results_det_ood_all = np.array(results_det_ood_all).T
    results_det_ood = np.mean(results_det_ood_all, axis=1)

    width = 0.4
    x = range(len(models))

    # colorpalettes = {
    #     "A": sns.color_palette("muted"),
    #     "V": sns.color_palette("colorblind"),
    #     "AV": sns.color_palette("dark"),
    # }
    # colors1 = [colorpalettes[MODEL_MODALITY[m]][0] for m in models]
    # colors2 = [colorpalettes[MODEL_MODALITY[m]][1] for m in models]

    colorpalette = sns.color_palette("Paired")
    def get_index(model_name):
        indices = ["A", "V", "AV"]
        return 2 * indices.index(MODEL_MODALITY[model_name])

    colors1 = [colorpalette[get_index(m)] for m in models]
    colors2 = [colorpalette[get_index(m) + 1] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar([i - width for i in x], results_det, width, color=colors1)
    bars2 = ax.bar([i for i in x], results_det_ood, width, color=colors2)

    ax.legend(
        handles=[bars1[0], bars2[0]],
        labels=["In-domain", "Out-of-domain"],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )

    ax.set_ylabel("AUC (%)")
    model_names = [MODEL_SHORT.get(m, m) for m in models]
    xx = [x1 - width / 2 for x1 in x]
    ax.set_xticks(xx)
    # ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_xticklabels(model_names, rotation=90, ha="center")

    # for bar in list(bars1) + list(bars2):
    #     h = bar.get_height()
    #     ax.annotate(
    #         f"{h:.1f}",
    #         xy=(bar.get_x() + bar.get_width() / 2, h),
    #         xytext=(0, 3),
    #         textcoords="offset points",
    #         ha="center",
    #         va="bottom",
    #         fontsize=10,
    #     )

    fig.tight_layout()
    # fig.savefig("output/plots/id-vs-ood.pdf", bbox_inches="tight", transparent=True)
    fig.savefig("output/plots/id-vs-ood.png", bbox_inches="tight", transparent=True)
    return fig



if __name__ == "__main__":
    # fig = make_bar_plot()
    # st.pyplot(fig)
    fig = make_loc_plot()
    st.pyplot(fig)
    fig = make_loc_plot_2()
    st.pyplot(fig)
