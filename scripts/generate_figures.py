"""
ALSE V3.4b - Publication-Quality Figure Generation
Generates all figures for paper submission
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Create figures directory
figures_dir = Path('/Users/sambhavdixit/Desktop/ALCE/figures')
figures_dir.mkdir(exist_ok=True)

# Color scheme
ALSE_COLOR = '#2E86AB'  # Blue
BPE_COLOR = '#A23B72'   # Purple
NEUTRAL_COLOR = '#F18F01'  # Orange

# ============================================================================
# FIGURE 1: BPB Comparison Across Vocab Sizes (Main Result)
# ============================================================================
def generate_bpb_comparison():
    """Bar chart comparing BPB across vocab sizes"""
    vocab_sizes = ['128', '512', '2048']
    alse_bpb = [1.3347, 2.1383, 2.1286]
    bpe_bpb = [3.5497, 2.8145, 2.7461]

    x = np.arange(len(vocab_sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width/2, alse_bpb, width, label='ALSE', color=ALSE_COLOR, alpha=0.9)
    bars2 = ax.bar(x + width/2, bpe_bpb, width, label='BPE', color=BPE_COLOR, alpha=0.9)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    # Add improvement percentages
    improvements = [(bpe_bpb[i] - alse_bpb[i]) / bpe_bpb[i] * 100 for i in range(len(vocab_sizes))]
    for i, imp in enumerate(improvements):
        ax.text(i, max(alse_bpb[i], bpe_bpb[i]) + 0.15,
               f'-{imp:.0f}%', ha='center', fontsize=9,
               fontweight='bold', color='green')

    ax.set_xlabel('Vocabulary Size', fontweight='bold')
    ax.set_ylabel('Bits Per Byte (BPB) ↓', fontweight='bold')
    ax.set_title('BPB Comparison: ALSE vs BPE Across Vocab Sizes', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(vocab_sizes)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(bpe_bpb) * 1.15)

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig1_bpb_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig1_bpb_comparison.png")

# ============================================================================
# FIGURE 2: Scaling Curves
# ============================================================================
def generate_scaling_curves():
    """Line plot showing BPB scaling with vocab size"""
    vocab_sizes = [128, 512, 2048]
    alse_bpb = [1.3347, 2.1383, 2.1286]
    bpe_bpb = [3.5497, 2.8145, 2.7461]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(vocab_sizes, alse_bpb, marker='o', linewidth=2.5,
           markersize=10, label='ALSE', color=ALSE_COLOR)
    ax.plot(vocab_sizes, bpe_bpb, marker='s', linewidth=2.5,
           markersize=10, label='BPE', color=BPE_COLOR)

    # Add value annotations
    for i, (v, a, b) in enumerate(zip(vocab_sizes, alse_bpb, bpe_bpb)):
        ax.annotate(f'{a:.3f}', (v, a), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=9)
        ax.annotate(f'{b:.3f}', (v, b), textcoords="offset points",
                   xytext=(0,-15), ha='center', fontsize=9)

    ax.set_xlabel('Vocabulary Size', fontweight='bold')
    ax.set_ylabel('Bits Per Byte (BPB) ↓', fontweight='bold')
    ax.set_title('Scaling Analysis: BPB vs Vocabulary Size', fontweight='bold', pad=15)
    ax.set_xscale('log', base=2)
    ax.set_xticks(vocab_sizes)
    ax.set_xticklabels(['128', '512', '2048'])
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig2_scaling_curves.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig2_scaling_curves.png")

# ============================================================================
# FIGURE 3: PATH C - Large-Scale LM Parity (CRITICAL RESULT)
# ============================================================================
def generate_lm_parity():
    """Bar chart for 50M parameter LM comparison"""
    models = ['BPE LM\n(50M params)', 'ALSE LM\n(50M params)']
    bpb_values = [2.7748, 0.8275]

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [BPE_COLOR, ALSE_COLOR]
    bars = ax.bar(models, bpb_values, color=colors, alpha=0.9, width=0.6)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add improvement annotation
    improvement = (bpb_values[0] - bpb_values[1]) / bpb_values[0] * 100
    ax.text(0.5, max(bpb_values) * 0.8,
           f'ALSE: 70% Better',
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_ylabel('Bits Per Byte (BPB) ↓', fontweight='bold')
    ax.set_title('PATH C: Large-Scale LM Parity (50M Parameters)',
                fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(bpb_values) * 1.2)

    # Add annotation box
    textstr = 'Same architecture\nSame training data\nSame parameter count'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig3_lm_parity.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig3_lm_parity.png")

# ============================================================================
# FIGURE 4: PATH B - Distillation Comparison
# ============================================================================
def generate_distillation_comparison():
    """Bar chart for distillation results"""
    students = ['Student A\n(Byte-level)', 'Student B\n(ALSE-tokenized)']
    bpb_values = [3.3413, 1.3347]

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [NEUTRAL_COLOR, ALSE_COLOR]
    bars = ax.bar(students, bpb_values, color=colors, alpha=0.9, width=0.6)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add improvement annotation
    improvement = (bpb_values[0] - bpb_values[1]) / bpb_values[0] * 100
    ax.text(0.5, max(bpb_values) * 0.7,
           f'60% Better\n(No Tokenizer Mismatch)',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_ylabel('Bits Per Byte (BPB) ↓', fontweight='bold')
    ax.set_title('PATH B: Distillation Results (BPE Teacher)',
                fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(bpb_values) * 1.2)

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig4_distillation.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig4_distillation.png")

# ============================================================================
# FIGURE 5: Vocab Usage Efficiency
# ============================================================================
def generate_vocab_usage():
    """Line plot showing vocab usage efficiency"""
    vocab_sizes = [128, 512, 2048]
    alse_usage = [67.2, 43.2, 7.4]
    bpe_usage = [78.9, 86.7, 85.5]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(vocab_sizes, alse_usage, marker='o', linewidth=2.5,
           markersize=10, label='ALSE', color=ALSE_COLOR)
    ax.plot(vocab_sizes, bpe_usage, marker='s', linewidth=2.5,
           markersize=10, label='BPE', color=BPE_COLOR)

    # Add value annotations
    for i, (v, a, b) in enumerate(zip(vocab_sizes, alse_usage, bpe_usage)):
        ax.annotate(f'{a:.1f}%', (v, a), textcoords="offset points",
                   xytext=(0,-15), ha='center', fontsize=9)
        ax.annotate(f'{b:.1f}%', (v, b), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=9)

    ax.set_xlabel('Vocabulary Size', fontweight='bold')
    ax.set_ylabel('Vocabulary Usage (%)', fontweight='bold')
    ax.set_title('Vocabulary Usage Efficiency', fontweight='bold', pad=15)
    ax.set_xscale('log', base=2)
    ax.set_xticks(vocab_sizes)
    ax.set_xticklabels(['128', '512', '2048'])
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    # Add note about 2048
    ax.text(2048, 7.4, '← Needs more training', fontsize=8,
           va='center', ha='left', style='italic')

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig5_vocab_usage.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig5_vocab_usage.png")

# ============================================================================
# FIGURE 6: Comprehensive Comparison Matrix
# ============================================================================
def generate_comparison_matrix():
    """Heatmap showing all key metrics"""
    metrics = ['BPB (128)', 'BPB (512)', 'BPB (2048)', 'LM Parity', 'Distillation']

    # Normalize to percentage improvement (ALSE vs BPE)
    improvements = [
        62.4,  # BPB at 128
        24.0,  # BPB at 512
        22.5,  # BPB at 2048
        70.2,  # LM Parity
        60.1   # Distillation
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create data matrix
    data = np.array(improvements).reshape(-1, 1)

    im = ax.imshow(data, cmap='Greens', aspect='auto', vmin=0, vmax=80)

    # Set ticks
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xticks([0])
    ax.set_xticklabels(['ALSE Improvement (%)'])

    # Add text annotations
    for i in range(len(metrics)):
        text = ax.text(0, i, f'{improvements[i]:.1f}%',
                      ha="center", va="center", color="black",
                      fontweight='bold', fontsize=12)

    ax.set_title('ALSE Performance Improvements Over BPE',
                fontweight='bold', pad=15)

    # Color bar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Improvement (%)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig6_comparison_matrix.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig6_comparison_matrix.png")

# ============================================================================
# FIGURE 7: Compression Ratio Analysis
# ============================================================================
def generate_compression_analysis():
    """Bar chart showing bytes per token"""
    systems = ['BPE-128', 'ALSE-128', 'BPE-512', 'ALSE-512', 'BPE-2048', 'ALSE-2048']
    bytes_per_token = [1.00, 3.58, 2.21, 3.58, 3.17, 3.38]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [BPE_COLOR, ALSE_COLOR] * 3
    bars = ax.bar(systems, bytes_per_token, color=colors, alpha=0.9)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('System', fontweight='bold')
    ax.set_ylabel('Bytes Per Token (Higher = More Compression)', fontweight='bold')
    ax.set_title('Token Compression Efficiency', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BPE_COLOR, label='BPE'),
                      Patch(facecolor=ALSE_COLOR, label='ALSE')]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig7_compression.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig7_compression.png")

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("ALSE V3.4b - Generating Publication-Quality Figures")
    print("="*70 + "\n")

    print("Output directory:", figures_dir)
    print()

    generate_bpb_comparison()
    generate_scaling_curves()
    generate_lm_parity()
    generate_distillation_comparison()
    generate_vocab_usage()
    generate_comparison_matrix()
    generate_compression_analysis()

    print("\n" + "="*70)
    print("✓ All figures generated successfully!")
    print(f"✓ Location: {figures_dir}")
    print("="*70 + "\n")

    # List all generated files
    print("Generated files:")
    for fig_file in sorted(figures_dir.glob('*.png')):
        print(f"  - {fig_file.name}")
    print()
