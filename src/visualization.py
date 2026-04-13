import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("reports/figures", exist_ok=True)

def plot_target_distribution(df, target, log=False, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    data = np.log1p(df[target]) if log else df[target]
    label = f"log({target}+1)" if log else target
    axes[0].hist(data, bins=50, color="#2196F3", edgecolor="white")
    axes[0].set_title(f"Distribución de {label}")
    axes[0].set_xlabel(label)
    axes[1].boxplot(data.dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#90CAF9"))
    axes[1].set_title(f"Boxplot de {label}")
    plt.tight_layout()
    if save:
        plt.savefig(f"reports/figures/dist_{target}.png", dpi=150)
    plt.show()

def plot_correlation_heatmap(df, features, save=True):
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Mapa de Correlaciones", fontsize=14, pad=15)
    plt.tight_layout()
    if save:
        plt.savefig("reports/figures/correlation_heatmap.png", dpi=150)
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15, title="", save=True):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices][::-1], color="#42A5F5")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Importancia")
    ax.set_title(f"Top {top_n} Features — {title}", fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(f"reports/figures/feature_importance_{title}.png", dpi=150)
    plt.show()

def plot_confusion_matrix(y_test, y_pred, labels=["No Viral", "Viral"], title="", save=True):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión — {title}")
    plt.tight_layout()
    if save:
        plt.savefig(f"reports/figures/confusion_matrix_{title}.png", dpi=150)
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, title="", save=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, color="#1565C0", s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Predicho")
    ax.set_title(f"Real vs Predicho — {title}")
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"reports/figures/actual_vs_predicted_{title}.png", dpi=150)
    plt.show()