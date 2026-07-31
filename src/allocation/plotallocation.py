import numpy as np
import matplotlib.pyplot as plt

# Data from Seed 0
feature_labels = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 
                  'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8']
phi = np.array([0.000, 0.305, 0.000, 0.269, 0.142, 0.107, 0.000, 0.177])
kappa = np.array([0, 3, 0, 2, 2, 1, 0, 2])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Plot 1: Feature Importance (phi)
bars1 = ax1.bar(feature_labels, phi, color='steelblue', edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Feature Importance (φ)', fontsize=12, fontweight='bold')
ax1.set_title('AHQE Feature Importance (Seed 0)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 0.4)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='Importance Threshold')
ax1.legend()

# Add value labels on bars
for bar, val in zip(bars1, phi):
    if val > 0.01:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Depth Allocation (kappa)
colors = ['red' if k == 0 else 'forestgreen' for k in kappa]
bars2 = ax2.bar(feature_labels, kappa, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Depth Allocation (κ)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Feature Groups', fontsize=12, fontweight='bold')
ax2.set_title('AHQE Depth Allocation (Seed 0)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 4)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars2, kappa):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='forestgreen', label='Allocated Depth'),
                   Patch(facecolor='red', label='Zero Allocation')]
ax2.legend(handles=legend_elements, loc='upper right')

# Add budget text
ax2.text(0.02, 0.95, f'Budget used: {sum(kappa)}/10', 
         transform=ax2.transAxes, fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='black'))

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('outputs/allocation.png', dpi=300, bbox_inches='tight')
print("Saved allocation.png to outputs/")

# Show plot
plt.show()