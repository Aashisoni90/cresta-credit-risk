import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def add_box(ax, text, center_x, center_y, width, height, facecolor, edgecolor, fontcolor='white', fontsize=12):
    box = patches.FancyBboxPatch(
        (center_x - width/2, center_y - height/2),
        width, height,
        boxstyle="round,pad=0.2",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(center_x, center_y, text, ha='center', va='center', 
            color=fontcolor, fontsize=fontsize, fontweight='bold', wrap=True)

def add_arrow(ax, x1, y1, x2, y2):
    ax.annotate("",
                xy=(x2, y2), xycoords='data',
                xytext=(x1, y1), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="gray", lw=2, shrinkA=0, shrinkB=0))

def main():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    c_blue = '#1f77b4'
    c_orange = '#ff7f0e'
    c_red = '#d62728'
    c_green = '#2ca02c'
    c_purple = '#9467bd'
    c_teal = '#17becf'

    w, h = 3.5, 1.2
    
    add_box(ax, "1. Raw Data Ingestion\n(Synthetic N=60,000, 10 Years)", 2.5, 7, w, h, c_blue, 'darkblue', fontsize=10)
    add_box(ax, "2. Quality Gates\n• Accounting Identity\n• Sign Sanity\n• Coverage Threshold", 7.5, 7, w, h, c_orange, 'darkorange', fontsize=10)
    add_box(ax, "3. Circularity Ablation\n• Shift Target ($Y_{t+1}$)\n• Drop Deterministic Features", 7.5, 4.5, w, h, c_red, 'darkred', fontsize=10)
    add_box(ax, "4. Feature Engineering\n(Financials, Market, ESG)\n• Size • Profitability • Leverage\n• Velocity • ESG Score", 2.5, 4.5, w, h, c_green, 'darkgreen', fontsize=10)
    add_box(ax, "5. XGBoost Modeling Engine\n• Chronological Split\n• scale_pos_weight\n• 5-Classifier Benchmark", 2.5, 2, w, h, c_purple, 'indigo', fontsize=10)
    add_box(ax, "6. Validation & Explainability\n• Out-of-Time Eval (PR/ROC)\n• Bootstrapped ΔAUC\n• SHAP & PDP Analysis", 7.5, 2, w, h, c_teal, 'darkcyan', fontsize=10)
    
    add_arrow(ax, 4.25, 7, 5.75, 7) # 1 -> 2
    add_arrow(ax, 7.5, 6.4, 7.5, 5.1) # 2 -> 3
    add_arrow(ax, 5.75, 4.5, 4.25, 4.5) # 3 -> 4
    add_arrow(ax, 2.5, 3.9, 2.5, 2.6) # 4 -> 5
    add_arrow(ax, 4.25, 2, 5.75, 2) # 5 -> 6
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    plt.title("CRESTA Pipeline Architecture", fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/cresta_architecture.png', dpi=300, bbox_inches='tight')
    print("Generated architecture diagram at reports/figures/cresta_architecture.png")

if __name__ == "__main__":
    main()
