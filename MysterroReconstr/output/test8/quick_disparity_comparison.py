#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Disparity Comparison - Parallel View
==========================================

Quick script to generate a single row of parallel disparity maps for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def read_pfm(filepath):
    """Read PFM file"""
    with open(filepath, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        if header != 'Pf':
            raise ValueError('Not a PFM file')
        
        dims = f.readline().decode('utf-8').strip().split()
        width, height = int(dims[0]), int(dims[1])
        
        scale = float(f.readline().decode('utf-8').strip())
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape((height, width))
        
        if scale < 0:
            data = np.flipud(data)
        
        return data

def read_png_disparity(filepath):
    """Read PNG disparity map"""
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32)

def normalize_disparity(disparity):
    """Normalize disparity for visualization"""
    if disparity is None:
        return None
        
    valid_mask = disparity > 0
    if not np.any(valid_mask):
        return np.zeros_like(disparity)
    
    min_val = np.min(disparity[valid_mask])
    max_val = np.max(disparity[valid_mask])
    
    normalized = np.zeros_like(disparity)
    normalized[valid_mask] = (disparity[valid_mask] - min_val) / (max_val - min_val)
    return normalized

def create_parallel_comparison():
    """Create parallel comparison of disparity maps"""
    
    # Define files to compare
    files_to_compare = [
        ('Ground Truth', 'disp0.pfm'),
        ('ELAS (SGM Fill)', 'disparity_ELAS_left_original_sgm_fill.png'),
        ('ELAS (No Fill)', 'disparity_ELAS_left_original_no_fill.png'),
        ('ADCE (Balanced)', 'disparity_ADCE_extended_range_Balanced_fill_on.png'),
        ('ADCE (Original)', 'disparity_ADCE_original_fill_on.png')
    ]
    
    # Load disparity maps
    disparity_maps = {}
    for name, filename in files_to_compare:
        filepath = Path(filename)
        if filepath.exists():
            try:
                if filename.endswith('.pfm'):
                    disparity_maps[name] = read_pfm(filepath)
                else:
                    disparity_maps[name] = read_png_disparity(filepath)
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
        else:
            print(f"✗ File not found: {filename}")
    
    if not disparity_maps:
        print("No disparity maps loaded!")
        return
    
    # Create parallel comparison plot
    n_maps = len(disparity_maps)
    fig, axes = plt.subplots(1, n_maps, figsize=(20, 6))
    
    if n_maps == 1:
        axes = [axes]
    
    # Custom colormap for disparity visualization
    colors = ['black', 'blue', 'cyan', 'green', 'yellow', 'red', 'white']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('disparity_cmap', colors, N=256)
    
    # Plot each disparity map
    for i, (name, disparity) in enumerate(disparity_maps.items()):
        # Normalize for visualization
        normalized = normalize_disparity(disparity)
        
        # Plot
        im = axes[i].imshow(normalized, cmap=cmap, vmin=0, vmax=1)
        axes[i].set_title(f'{name}\n{disparity.shape[1]}×{disparity.shape[0]}', 
                        fontsize=12, fontweight='bold')
        axes[i].axis('off')
        
        # Add statistics
        valid_mask = disparity > 0
        if np.any(valid_mask):
            coverage = np.sum(valid_mask) / disparity.size * 100
            mean_disp = np.mean(disparity[valid_mask])
            axes[i].text(0.02, 0.98, f'Coverage: {coverage:.1f}%\nMean: {mean_disp:.1f}', 
                       transform=axes[i].transAxes, fontsize=10,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, shrink=0.8, aspect=20)
    cbar.set_label('Normalized Disparity', fontsize=12)
    
    # Add main title
    fig.suptitle('Disparity Map Comparison: Ground Truth vs ELAS vs ADCE', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save and show
    plt.savefig('disparity_parallel_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: disparity_parallel_comparison.png")
    
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("QUICK DISPARITY COMPARISON")
    print("="*60)
    create_parallel_comparison()
    print("="*60)
    print("COMPLETED!")
    print("="*60) 