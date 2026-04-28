"""
Genotype Comparison & Activity Analysis Figures
================================================

All genotype-related figures, statistical test visualisations,
dataset overview, between-organoid comparisons, activity analysis,
and ROI peak frame generation.

Split from group_analysis.py for maintainability.

Imported by group_analysis.run_analysis().

Status: Refactoring
"""

import os
import logging
import traceback
from collections import OrderedDict
from typing import List, Optional, Dict, Any, Tuple
import re
from pathlib import Path

#Scientific stack
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.sparse import load_npz
from scipy.ndimage import percentile_filter
from sklearn.preprocessing import StandardScaler

#Visualisation stack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

# Import data structures and helpers from the main module
try:
    from .group_analysis import (
        DatasetMetrics, FEATURE_NAMES, _abbrev,
        _get_neuron_rates, _get_neuron_amplitudes, _recording_metric,
        _extract_organoid_id, _extract_genotype, _extract_line_id,
        _zscore_within_dataset,
        _pairwise_correlations, _synchrony_index,
        _flag_suspicious_neurons,
        _measure_transient_amplitudes,
        build_feature_matrix,
    )
except ImportError:
    from group_analysis import (
        DatasetMetrics, FEATURE_NAMES, _abbrev,
        _get_neuron_rates, _get_neuron_amplitudes, _recording_metric,
        _extract_organoid_id, _extract_genotype, _extract_line_id,
        _zscore_within_dataset,
        _pairwise_correlations, _synchrony_index,
        _flag_suspicious_neurons,
        _measure_transient_amplitudes,
        build_feature_matrix,
    )

# Import shared figure utils from overview module
try:
    from .figures_overview import _fmt_p, _sig_stars, _draw_sig_bracket
except ImportError:
    from figures_overview import _fmt_p, _sig_stars, _draw_sig_bracket

#Global Logger
logger = logging.getLogger(__name__)


# Refactored Constants
DEFAULT_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
CTRL_COLOR = '#4472C4'
MUT_COLOR = '#ED7D31'

# =============================================================================
# ORGANOID-LEVEL FIGURES
# =============================================================================

def _fig_spike_rate_by_organoid(datasets, organoid_ids, unique_organoids,
                                 org_colors, output_dir):
    """Spike rate by organoid dot plot with per-day grouping."""

    def _extract_day_rec(name):
        date_match = re.search(r'(\d{6})', name)
        rec_match = re.search(r'(R\d+)', name)
        day = date_match.group(1) if date_match else 'unknown'
        rec = rec_match.group(1) if rec_match else ''
        return day, rec

    rng = np.random.default_rng(42)
    org_day_rates = OrderedDict()
    for oid in unique_organoids:
        org_day_rates[oid] = OrderedDict()

    for ds, oid in zip(datasets, organoid_ids):
        rates = _get_neuron_rates(ds)
        if len(rates) == 0:
            continue
        rec_mean = float(np.mean(rates))
        day, rec = _extract_day_rec(ds.name)
        if day not in org_day_rates[oid]:
            org_day_rates[oid][day] = []
        org_day_rates[oid][day].append((rec, rec_mean))

    total_days = sum(len(days) for days in org_day_rates.values())
    if total_days == 0:
        logger.warning("  No spike rate data — skipping spike rate by organoid figure")
        return

    n_org = len(unique_organoids)
    fig_width = max(18, total_days * 1.1 + n_org * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x_pos = 0
    x_ticks = []
    x_labels = []
    all_plotted_vals = []

    for oid in unique_organoids:
        days = org_day_rates[oid]
        for day_idx, (day, recordings) in enumerate(sorted(days.items())):
            if not recordings:
                continue
            day_vals = np.array([r[1] for r in recordings])
            all_plotted_vals.extend(day_vals.tolist())
            jitter = rng.uniform(-0.15, 0.15, len(day_vals))
            ax.scatter(x_pos + jitter, day_vals, c=org_colors[oid], s=55,
                      alpha=0.75, edgecolor='white', linewidth=0.6, zorder=5)
            if len(day_vals) > 1:
                ax.scatter(x_pos, np.mean(day_vals), c=org_colors[oid], s=150,
                          marker='_', linewidths=3.5, zorder=6)
            day_formatted = f'{day[2:4]}/{day[0:2]}' if len(day) == 6 else day
            x_ticks.append(x_pos)
            x_labels.append(day_formatted)
            x_pos += 1
        x_pos += 0.3

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')
    ax.set_xlim(-0.8, x_pos - 0.3)

    if all_plotted_vals:
        y_floor = max(0.0, float(np.percentile(all_plotted_vals, 1)) - 0.02)
        y_ceil = float(np.percentile(all_plotted_vals, 99)) * 1.15
        ax.set_ylim(bottom=y_floor, top=max(y_ceil, 0.1))

    ax.set_ylabel('Event rate (events/10s)', fontsize=12)
    ax.set_xlabel('Recording Date (DD/MM)', fontsize=11)
    ax.set_title('Event rate by organoid  (per-recording averages)',
                 fontsize=14, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    if all_plotted_vals:
        global_mean = np.mean(all_plotted_vals)
        ax.axhline(global_mean, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(ax.get_xlim()[1], global_mean, f' mean={global_mean:.2f}',
               va='center', ha='left', fontsize=9, color='#666666')

    legend_elements = [Patch(facecolor=org_colors[oid], alpha=0.7, label=oid)
                      for oid in unique_organoids]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              title='Organoid', title_fontsize=11, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'figures', '1 - Main Results', 'spike_rate_by_organoid.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Event rate by organoid saved: {path}")


def _fig_correlation_by_organoid(datasets, organoid_ids, unique_organoids,
                                  org_colors, output_dir):
    """Pairwise correlation by organoid dot plot with per-day grouping."""

    def _extract_day_rec(name):
        date_match = re.search(r'(\d{6})', name)
        rec_match = re.search(r'(R\d+)', name)
        day = date_match.group(1) if date_match else 'unknown'
        rec = rec_match.group(1) if rec_match else ''
        return day, rec

    rng = np.random.default_rng(42)
    org_day_corr = OrderedDict()
    for oid in unique_organoids:
        org_day_corr[oid] = OrderedDict()

    all_corr_vals = []
    for ds, oid in zip(datasets, organoid_ids):
        corr_val = ds.pairwise_correlation_mean
        if not np.isfinite(corr_val):
            continue
        all_corr_vals.append(corr_val)
        day, rec = _extract_day_rec(ds.name)
        if day not in org_day_corr[oid]:
            org_day_corr[oid][day] = []
        org_day_corr[oid][day].append((rec, corr_val))

    total_days = sum(len(days) for days in org_day_corr.values())
    if total_days == 0:
        logger.warning("  No valid correlation data — skipping correlation by organoid figure")
        return

    n_org = len(unique_organoids)
    fig, ax = plt.subplots(figsize=(max(10, total_days * 0.6 + n_org * 0.8), 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x_pos = 0
    x_ticks = []
    x_labels = []

    for oid in unique_organoids:
        days = org_day_corr[oid]
        for day, recordings in sorted(days.items()):
            vals = [r[1] for r in recordings]
            day_formatted = f'{day[2:4]}/{day[0:2]}' if len(day) == 6 else day
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(x_pos + jitter, vals, c=org_colors[oid], s=60, alpha=0.7,
                      edgecolor='white', linewidth=0.5, zorder=5)
            if len(vals) > 0:
                ax.scatter(x_pos, np.mean(vals), c=org_colors[oid], s=120,
                          marker='_', linewidths=3, zorder=6)
            x_ticks.append(x_pos)
            x_labels.append(day_formatted)
            x_pos += 1
        x_pos += 0.3

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')
    ax.set_xlim(-0.8, x_pos - 0.3)
    ax.set_ylabel('Pairwise Correlation (r)', fontsize=11)
    ax.set_xlabel('Recording Date (DD/MM)', fontsize=10)
    ax.set_title('Pairwise Correlation by Organoid', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    if all_corr_vals:
        mn_c = float(np.mean(all_corr_vals))
        ax.axhline(mn_c, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(ax.get_xlim()[1], mn_c, f' mean={mn_c:.3f}',
               va='center', ha='left', fontsize=8, color='#666666')

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=org_colors[oid],
                              markersize=10, alpha=0.7, label=oid)
                      for oid in unique_organoids]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              title='Organoid', title_fontsize=10, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'figures', '1 - Main Results', 'correlation_by_organoid.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Correlation by organoid saved: {path}")


def _fig_synchrony_by_organoid(datasets, organoid_ids, unique_organoids,
                                org_colors, output_dir):
    """Synchrony index by organoid dot plot with per-day grouping."""

    def _extract_day_rec(name):
        date_match = re.search(r'(\d{6})', name)
        rec_match = re.search(r'(R\d+)', name)
        day = date_match.group(1) if date_match else 'unknown'
        rec = rec_match.group(1) if rec_match else ''
        return day, rec

    rng = np.random.default_rng(42)
    org_day_sync = OrderedDict()
    for oid in unique_organoids:
        org_day_sync[oid] = OrderedDict()

    all_sync_vals = []
    for ds, oid in zip(datasets, organoid_ids):
        sync_val = ds.synchrony_index
        if not np.isfinite(sync_val):
            continue
        all_sync_vals.append(sync_val)
        day, rec = _extract_day_rec(ds.name)
        if day not in org_day_sync[oid]:
            org_day_sync[oid][day] = []
        org_day_sync[oid][day].append((rec, sync_val))

    total_days = sum(len(days) for days in org_day_sync.values())
    if total_days == 0:
        logger.warning("  No valid synchrony data — skipping synchrony by organoid figure")
        return

    n_org = len(unique_organoids)
    fig, ax = plt.subplots(figsize=(max(10, total_days * 0.6 + n_org * 0.8), 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x_pos = 0
    x_ticks = []
    x_labels = []

    for oid in unique_organoids:
        days = org_day_sync[oid]
        for day, recordings in sorted(days.items()):
            vals = [r[1] for r in recordings]
            day_formatted = f'{day[2:4]}/{day[0:2]}' if len(day) == 6 else day
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(x_pos + jitter, vals, c=org_colors[oid], s=60, alpha=0.7,
                      edgecolor='white', linewidth=0.5, zorder=5)
            if len(vals) > 0:
                ax.scatter(x_pos, np.mean(vals), c=org_colors[oid], s=120,
                          marker='_', linewidths=3, zorder=6)
            x_ticks.append(x_pos)
            x_labels.append(day_formatted)
            x_pos += 1
        x_pos += 0.3

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')
    ax.set_xlim(-0.8, x_pos - 0.3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Synchrony Index', fontsize=11)
    ax.set_xlabel('Recording Date (DD/MM)', fontsize=10)
    ax.set_title('Synchrony Index by Organoid', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    if all_sync_vals:
        mn_s = float(np.mean(all_sync_vals))
        ax.axhline(mn_s, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(ax.get_xlim()[1], mn_s, f' mean={mn_s:.3f}',
               va='center', ha='left', fontsize=8, color='#666666')

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=org_colors[oid],
                              markersize=10, alpha=0.7, label=oid)
                      for oid in unique_organoids]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              title='Organoid', title_fontsize=10, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'figures', '1 - Main Results', 'synchrony_by_organoid.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Synchrony by organoid saved: {path}")



# =============================================================================
# STATISTICAL TESTS & DATASET OVERVIEW
# =============================================================================

def run_statistical_tests(datasets: List[DatasetMetrics], output_dir: str) -> dict:
    """
    Statistical comparison across datasets.

    Produces three by-organoid figures (spike rate, correlation, synchrony)
    and computes Kruskal-Wallis + pairwise Mann-Whitney U tests on spike rates.
    """

    n_ds = len(datasets)
    results = {
        'n_datasets': n_ds,
        'data_sources': {
            'spike_rate': 'Deconvolved spike trains (S > 0 events, OASIS output)',
            'pairwise_correlation': 'Denoised calcium traces (Pearson r, OASIS output)',
            'synchrony_index': 'Denoised calcium traces (population coupling)',
            'note': 'All metrics computed from top-N quality-selected neurons per dataset',
        },
        'tests': {},
    }

    # ── Per-neuron spike rates by dataset ────────────────────────────────
    per_ds_rates = []
    ds_names = []
    for ds in datasets:
        rates = _get_neuron_rates(ds)
        if len(rates) == 0:
            continue
        per_ds_rates.append(rates)
        ds_names.append(_abbrev(ds.name))

    per_ds_rate_means = np.array([float(np.mean(r)) for r in per_ds_rates]) if per_ds_rates else np.array([])

    # ── Kruskal-Wallis ───────────────────────────────────────────────────
    kw_result = None
    if len(per_ds_rates) >= 2 and all(len(r) >= 2 for r in per_ds_rates):
        H, p_kw = sp_stats.kruskal(*per_ds_rates)
        kw_result = {'H': float(H), 'p': float(p_kw), 'n_groups': len(per_ds_rates)}

    # ── Pairwise Mann-Whitney U ──────────────────────────────────────────
    n_pairs = len(per_ds_rates) * (len(per_ds_rates) - 1) // 2
    pairwise = []
    if n_pairs > 0:
        for i in range(len(per_ds_rates)):
            for j in range(i + 1, len(per_ds_rates)):
                U, p_raw = sp_stats.mannwhitneyu(
                    per_ds_rates[i], per_ds_rates[j], alternative='two-sided')
                p_bonf = min(p_raw * n_pairs, 1.0)
                pairwise.append({
                    'i': i, 'j': j,
                    'name_i': ds_names[i], 'name_j': ds_names[j],
                    'U': float(U), 'p_raw': float(p_raw), 'p_bonf': float(p_bonf),
                })

    results['tests'] = {
        'kruskal_wallis': kw_result,
        'pairwise_mannwhitney': pairwise,
        'spike_rate_summary': {
            'n_recordings': len(per_ds_rate_means),
            'mean': float(np.mean(per_ds_rate_means)) if len(per_ds_rate_means) > 0 else 0,
            'median': float(np.median(per_ds_rate_means)) if len(per_ds_rate_means) > 0 else 0,
            'sd': float(np.std(per_ds_rate_means, ddof=1)) if len(per_ds_rate_means) > 1 else 0,
        },
    }

    # Collect correlation and synchrony summary stats
    corr_vals = np.array([ds.pairwise_correlation_mean for ds in datasets])
    sync_vals = np.array([ds.synchrony_index for ds in datasets])
    corr_vals_valid = corr_vals[np.isfinite(corr_vals)]
    sync_vals_valid = sync_vals[np.isfinite(sync_vals)]

    for key, vals, label in [
        ('pairwise_correlation', corr_vals_valid, 'Pairwise Correlation (r)'),
        ('synchrony_index', sync_vals_valid, 'Synchrony Index'),
    ]:
        if len(vals) > 0:
            results['tests'][key] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0,
                'n': int(len(vals)),
            }

    # ── Organoid grouping (shared by all three figures) ──────────────────
    organoid_ids = [_extract_organoid_id(ds.name) for ds in datasets]
    unique_organoids = list(OrderedDict.fromkeys(organoid_ids))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    org_colors = {oid: palette[i % len(palette)] for i, oid in enumerate(unique_organoids)}

    # ── Generate figures (each in its own try/except) ────────────────────
    try:
        _fig_spike_rate_by_organoid(datasets, organoid_ids, unique_organoids,
                                     org_colors, output_dir)
    except Exception as e:
        logger.warning(f"  Spike rate by organoid figure failed: {e}")

    try:
        _fig_correlation_by_organoid(datasets, organoid_ids, unique_organoids,
                                      org_colors, output_dir)
    except Exception as e:
        logger.warning(f"  Correlation by organoid figure failed: {e}")

    try:
        _fig_synchrony_by_organoid(datasets, organoid_ids, unique_organoids,
                                    org_colors, output_dir)
    except Exception as e:
        logger.warning(f"  Synchrony by organoid figure failed: {e}")

    logger.info(f"  Statistical analysis complete")
    return results



def run_dataset_overview(datasets: List[DatasetMetrics], output_dir: str) -> dict:
    """
    Generate publication-quality visualizations for many-dataset comparisons.
    
    Produces separate figures:
    - dataset_umap.png — Clean UMAP projection (publication style)
    - dataset_heatmap.png — Clustered heatmap of standardized features  
    - metric_*.png — Individual metric plots grouped by organoid and day
    """

    n_ds = len(datasets)
    results = {'n_datasets': n_ds, 'visualizations': []}
    
    # Create organized subdirectory structure
    summary_dir = os.path.join(output_dir, 'figures', '1 - Main Results')
    by_metric_dir = os.path.join(output_dir, 'figures', '1b - Metrics')
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(by_metric_dir, exist_ok=True)
    
    # ── Extract organoid IDs and build color map ──────────────────────────
    organoid_ids = [_extract_organoid_id(ds.name) for ds in datasets]
    unique_organoids = list(OrderedDict.fromkeys(organoid_ids))
    n_org = len(unique_organoids)
    
    # Professional color palette (colorblind-friendly)
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    org_colors = {oid: palette[i % len(palette)] for i, oid in enumerate(unique_organoids)}
    ds_colors = [org_colors[oid] for oid in organoid_ids]
    
    # ── Build feature matrix ──────────────────────────────────────────────
    feature_attrs = [
        ('mean_spike_rate', 'Event Rate'),
        ('pairwise_correlation_mean', 'Correlation'),
        ('synchrony_index', 'Synchrony'),
        ('burst_rate', 'Burst Rate'),
        ('mean_burst_participation', 'Burst Participation'),
        ('cv_iei', 'IEI Variability'),
    ]
    
    X = np.zeros((n_ds, len(feature_attrs)))
    for i, ds in enumerate(datasets):
        for j, (attr, _) in enumerate(feature_attrs):
            val = _recording_metric(ds, attr)
            X[i, j] = val if val is not None else np.nan

    # Impute NaN with column median (not zero — zero clusters excluded
    # recordings at the origin, distorting the UMAP embedding)
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad]) if (~bad).any() else 0.0

    X_std = StandardScaler().fit_transform(X)
    feat_labels = [fl for _, fl in feature_attrs]
    ds_names = [_abbrev(ds.name) for ds in datasets]
    # Keep original names for genotype extraction — _abbrev loses the line field
    ds_names_orig = [ds.name for ds in datasets]
    
    rng = np.random.default_rng(42)
    
    # =====================================================================
    # UMAP — removed (unreliable with small dataset counts)
    # =====================================================================
    
    # =====================================================================
    # FIGURE: Feature Heatmap (clustered)
    # =====================================================================
    try:
        fig_height = max(12, n_ds * 0.18)
        fig = plt.figure(figsize=(14, fig_height))
        fig.patch.set_facecolor('white')
        
        # Create gridspec with proper spacing for dendrogram
        gs = gridspec.GridSpec(1, 3, width_ratios=[0.2, 1, 0.05], wspace=0.05)
        
        # Hierarchical clustering
        if n_ds > 2:
            linkage = hierarchy.linkage(pdist(X_std, 'euclidean'), method='ward')
            dendro = hierarchy.dendrogram(linkage, no_plot=True)
            row_order = dendro['leaves']
        else:
            row_order = list(range(n_ds))
        
        # Dendrogram
        ax_dendro = fig.add_subplot(gs[0])
        if n_ds > 2:
            hierarchy.dendrogram(linkage, orientation='left', ax=ax_dendro,
                                leaf_rotation=0, leaf_font_size=1,
                                above_threshold_color='#888888',
                                color_threshold=0)
        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])
        for spine in ax_dendro.spines.values():
            spine.set_visible(False)
        
        # Heatmap
        ax_heat = fig.add_subplot(gs[1])
        X_ordered = X_std[row_order, :]
        names_ordered = [ds_names[i] for i in row_order]
        colors_ordered = [ds_colors[i] for i in row_order]
        orgs_ordered = [organoid_ids[i] for i in row_order]
        
        im = ax_heat.imshow(X_ordered, aspect='auto', cmap='RdBu_r',
                            vmin=-2.5, vmax=2.5)
        
        ax_heat.set_xticks(range(len(feat_labels)))
        ax_heat.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=10)
        ax_heat.set_yticks(range(n_ds))
        
        # Y-tick labels with organoid color coding
        ylabels = [f'{orgs_ordered[i]}' for i in range(n_ds)]
        ax_heat.set_yticklabels(ylabels, fontsize=6)
        for i, (label, color) in enumerate(zip(ax_heat.get_yticklabels(), colors_ordered)):
            label.set_color(color)
            label.set_fontweight('bold')
        
        ax_heat.set_title('Standardized Features (hierarchically clustered)', 
                          fontsize=12, fontweight='bold', pad=10)
        
        # Colorbar
        ax_cbar = fig.add_subplot(gs[2])
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Z-score', fontsize=10)
        
        plt.tight_layout()
        heatmap_path = os.path.join(summary_dir, 'feature_heatmap.png')
        plt.savefig(heatmap_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        results['visualizations'].append(heatmap_path)
        logger.info(f"Heatmap saved: {heatmap_path}")
        
    except Exception as e:
        logger.warning(f"Heatmap failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    
    # =====================================================================
    # FIGURES 3+: Individual metric plots grouped by organoid with day structure
    # =====================================================================
    
    metrics_to_plot = [
        ('mean_spike_rate', 'Event Rate', 'events/10s'),
        ('mean_spike_amplitude', 'Transient Amplitude', 'ΔF/F₀'),
        ('pairwise_correlation_mean', 'Pairwise Correlation', 'r'),
        ('synchrony_index', 'Synchrony Index', ''),
        ('burst_rate', 'Burst Rate', 'bursts/10s'),
    ]
    
    for attr, title, unit in metrics_to_plot:
        try:
            # Pool data by organoid (like between_organoid_comparison)
            org_data = OrderedDict()
            for oid in unique_organoids:
                org_data[oid] = []
            
            for ds, oid in zip(datasets, organoid_ids):
                val = _recording_metric(ds, attr)
                if val is not None and np.isfinite(val):
                    org_data[oid].append(val)
            
            # Simple figure - one box per organoid
            fig_width = max(8, n_org * 1.2)
            fig, ax = plt.subplots(figsize=(fig_width, 6))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Prepare arrays for box plot
            data_arrays = [np.array(org_data[oid]) for oid in unique_organoids]
            
            # Box plots
            bp = ax.boxplot(
                data_arrays, positions=range(n_org), widths=0.5,
                patch_artist=True, showfliers=False,
                medianprops=dict(color='white', linewidth=1.5),
                whiskerprops=dict(color='#555555', linewidth=0.8),
                capprops=dict(color='#555555', linewidth=0.8),
            )
            
            for i, (patch, oid) in enumerate(zip(bp['boxes'], unique_organoids)):
                patch.set_facecolor(org_colors[oid])
                patch.set_alpha(0.6)
                patch.set_edgecolor(org_colors[oid])
                patch.set_linewidth(1.5)
            
            # Overlay individual recordings
            for i, oid in enumerate(unique_organoids):
                vals = org_data[oid]
                if len(vals) > 0:
                    jitter = rng.uniform(-0.15, 0.15, len(vals))
                    ax.scatter(i + jitter, vals,
                              color=org_colors[oid], s=40, alpha=0.7,
                              edgecolor='white', linewidth=0.5, zorder=5)
            
            # X-axis with organoid labels
            ax.set_xticks(range(n_org))
            ax.set_xticklabels(unique_organoids, fontsize=10, fontweight='bold')
            ax.set_xlim(-0.6, n_org - 0.4)
            
            # Color the x-tick labels
            for i, (tick_label, oid) in enumerate(zip(ax.get_xticklabels(), unique_organoids)):
                tick_label.set_color(org_colors[oid])
            
            # Labels and styling
            ylabel = f'{title} ({unit})' if unit else title
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlabel('Organoid', fontsize=10)
            ax.set_title(f'{title} by Organoid', fontsize=13, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
            
            # Global mean line
            all_vals = [v for vals in org_data.values() for v in vals]
            if all_vals:
                global_mean = np.mean(all_vals)
                ax.axhline(global_mean, color='#666666', linestyle='--', 
                          linewidth=1.5, alpha=0.7, zorder=1)
                ax.text(ax.get_xlim()[1] + 0.05, global_mean, f'mean={global_mean:.2f}',
                       va='center', ha='left', fontsize=9, color='#666666',
                       clip_on=False)
            
            plt.tight_layout()
            metric_path = os.path.join(by_metric_dir, f'metric_{attr.replace("_mean", "")}.png')
            plt.savefig(metric_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            results['visualizations'].append(metric_path)
            logger.info(f"Metric plot saved: {metric_path}")
            
        except Exception as e:
            logger.warning(f"Metric plot {attr} failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    return results




# =============================================================================
# GENOTYPE COMPARISON
# =============================================================================

def run_genotype_comparison(datasets: List[DatasetMetrics], output_dir: str,
                            mutant_label: str = 'CEP41 R242H') -> dict:
    """
    Compare Control vs Mutant genotypes at two levels:

    1. **Within-day**: For each organoid day (e.g. D109), compare control
       (3_x) vs mutant (1_x) recordings from the same day. This controls
       for day-to-day variation in imaging conditions.

    2. **Global pooled**: Pool ALL control recordings vs ALL mutant recordings
       across all days. More statistical power but confounded by day effects.

    3. **Paired meta-analysis**: For each day that has BOTH genotypes, compute
       a within-day effect size (Cohen's d), then combine effect sizes across
       days with a fixed-effects meta-analysis.

    Metrics compared (all using z-scored values to remove imaging confounds):
    - Spike rate (per-neuron, from deconvolved spike trains)
    - Spike amplitude (per-neuron, from denoised trace transients)
    - Pairwise correlation (per-recording, from denoised traces)
    - Synchrony index (per-recording, from denoised traces)

    Statistical tests:
    - Mann-Whitney U (non-parametric, two-sided) with Bonferroni correction
    - Cohen's d effect sizes
    - Both raw and z-scored comparisons reported side-by-side

    Parameters
    ----------
    datasets : list of DatasetMetrics
        Quality-gated datasets.
    output_dir : str
        Base output directory; figures saved under figures/2 - Genotype Comparison/

    Returns
    -------
    dict : Full results including per-day breakdowns, pooled tests, and
           meta-analysis across days.
    """


    results = {'level': 'genotype', 'tests': {}}

    geno_dir = os.path.join(output_dir, 'figures', '2 - Genotype Comparison')
    os.makedirs(geno_dir, exist_ok=True)

    # ── Parse genotypes ──────────────────────────────────────────────────
    genotypes = [_extract_genotype(ds.name) for ds in datasets]
    organoid_ids = [_extract_organoid_id(ds.name) for ds in datasets]
    line_ids = [_extract_line_id(ds.name) for ds in datasets]

    unique_geno = sorted(set(genotypes))
    n_ctrl = sum(1 for g in genotypes if g == 'Control')
    n_mut = sum(1 for g in genotypes if g == 'Mutant')
    n_unk = sum(1 for g in genotypes if g == 'Unknown')

    logger.info(f"Genotype comparison: {n_ctrl} Control, {n_mut} Mutant, "
                f"{n_unk} Unknown recordings")

    results['n_control'] = n_ctrl
    results['n_mutant'] = n_mut
    results['n_unknown'] = n_unk
    results['dataset_genotypes'] = {
        ds.name: {'genotype': g, 'organoid': o, 'line': l}
        for ds, g, o, l in zip(datasets, genotypes, organoid_ids, line_ids)
    }

    if n_ctrl < 1 or n_mut < 1:
        logger.warning("Need at least 1 Control and 1 Mutant for genotype comparison")
        results['skipped'] = True
        results['reason'] = f"Only {n_ctrl} Control and {n_mut} Mutant recordings"
        return results

    # Exclude unknowns from comparison
    geno_datasets = [(ds, g, o) for ds, g, o in zip(datasets, genotypes, organoid_ids)
                     if g in ('Control', 'Mutant')]

    # ── Helper: extract per-neuron spike rates from a dataset ────────────
    def _ds_spike_rates(ds):
        """Return spike rates for ACTIVE neurons only (rate > 0)."""
        if ds.neuron_spike_rates is not None:
            rates = ds.neuron_spike_rates
            return rates[rates > 0]
        if ds.selected_spikes is None:
            return np.array([])
        dur_s = ds.duration_seconds if ds.duration_seconds > 0 else 1.0
        rates = np.array([np.sum(ds.selected_spikes[j] > 0) / dur_s * 10.0
                         for j in range(ds.selected_spikes.shape[0])])
        return rates[rates > 0]

    def _ds_spike_amplitudes(ds):
        """Return amplitudes for ACTIVE neurons only (amplitude > 0)."""
        if ds.neuron_spike_amplitudes is not None:
            amps = ds.neuron_spike_amplitudes
            return amps[amps > 0]
        if ds.selected_traces is None or ds.selected_spikes is None:
            return np.array([])
        amps = _measure_transient_amplitudes(
            ds.selected_traces, ds.selected_spikes, ds.frame_rate)
        amps = np.array(amps) if amps else np.array([])
        return amps[amps > 0] if len(amps) > 0 else amps

    # ── Colour scheme ────────────────────────────────────────────────────
    CTRL_COLOR = '#4472C4'   # blue
    MUT_COLOR = '#ED7D31'    # orange

    rng = np.random.default_rng(42)

    # =====================================================================
    # SECTION 1: GLOBAL COMPARISON (Control vs Mutant, all days)
    # =====================================================================
    # PRIMARY: per-recording averages (each recording = one data point)
    # This prevents single recordings with many neurons from dominating.
    # SUPPLEMENTARY: per-neuron pooled data kept for distribution plots.
    # =====================================================================
    logger.info("=== Global genotype comparison (per-recording averaging) ===")

    # Pool per-neuron metrics by genotype (for supplementary)
    ctrl_ds = [ds for ds, g, _ in geno_datasets if g == 'Control']
    mut_ds = [ds for ds, g, _ in geno_datasets if g == 'Mutant']

    # Per-neuron pooled (supplementary)
    ctrl_rates_raw = np.concatenate([_ds_spike_rates(ds) for ds in ctrl_ds]) if ctrl_ds else np.array([])
    mut_rates_raw = np.concatenate([_ds_spike_rates(ds) for ds in mut_ds]) if mut_ds else np.array([])
    ctrl_amps_raw = np.concatenate([_ds_spike_amplitudes(ds) for ds in ctrl_ds
                                    if len(_ds_spike_amplitudes(ds)) > 0]) if ctrl_ds else np.array([])
    mut_amps_raw = np.concatenate([_ds_spike_amplitudes(ds) for ds in mut_ds
                                   if len(_ds_spike_amplitudes(ds)) > 0]) if mut_ds else np.array([])

    # Per-recording averages (PRIMARY — each dot = one recording)
    def _recording_means(ds_list):
        """Compute per-recording mean rate and amplitude for active neurons."""
        rec_rates = []
        rec_amps = []
        for ds in ds_list:
            rates = _ds_spike_rates(ds)
            amps = _ds_spike_amplitudes(ds)
            if len(rates) > 0:
                rec_rates.append(float(np.mean(rates)))
            if len(amps) > 0:
                rec_amps.append(float(np.mean(amps)))
        return np.array(rec_rates), np.array(rec_amps)

    ctrl_rec_rates, ctrl_rec_amps = _recording_means(ctrl_ds)
    mut_rec_rates, mut_rec_amps = _recording_means(mut_ds)

    # Per-recording metrics (already one value per recording)
    # NaN = recording excluded from corr/sync due to n_selected < 5
    ctrl_corr = np.array([ds.pairwise_correlation_mean for ds in ctrl_ds])
    mut_corr  = np.array([ds.pairwise_correlation_mean for ds in mut_ds])
    ctrl_sync = np.array([ds.synchrony_index for ds in ctrl_ds])
    mut_sync  = np.array([ds.synchrony_index for ds in mut_ds])
    ctrl_af   = np.array([ds.active_fraction for ds in ctrl_ds])
    mut_af    = np.array([ds.active_fraction for ds in mut_ds])

    n_ctrl_corr_excl = int(np.sum(~np.isfinite(ctrl_corr)))
    n_mut_corr_excl  = int(np.sum(~np.isfinite(mut_corr)))
    if n_ctrl_corr_excl + n_mut_corr_excl > 0:
        logger.info(f"  Corr/sync: excluded {n_ctrl_corr_excl} Control and "
                    f"{n_mut_corr_excl} Mutant recordings (n_selected < 5)")

    logger.info(f"  Control: {len(ctrl_rec_rates)} recordings with active neurons "
                f"({len(ctrl_rates_raw)} total neurons)")
    logger.info(f"  Mutant:  {len(mut_rec_rates)} recordings with active neurons "
                f"({len(mut_rates_raw)} total neurons)")

    def _run_mw_test(ctrl, mut, label, use_zscore=False):
        """Run Mann-Whitney U and compute Cohen's d. NaN values are excluded."""
        result = {'metric': label, 'z_scored': use_zscore}
        ctrl = ctrl[np.isfinite(ctrl)]
        mut  = mut[np.isfinite(mut)]
        if len(ctrl) < 2 or len(mut) < 2:
            result['skipped'] = True
            result['reason'] = f"n_ctrl={len(ctrl)}, n_mut={len(mut)} (after NaN removal)"
            return result
        U, p = sp_stats.mannwhitneyu(ctrl, mut, alternative='two-sided')
        # Cohen's d
        pooled_std = np.sqrt(((len(ctrl)-1)*np.var(ctrl, ddof=1) +
                              (len(mut)-1)*np.var(mut, ddof=1)) /
                             (len(ctrl) + len(mut) - 2))
        d = (np.mean(ctrl) - np.mean(mut)) / pooled_std if pooled_std > 1e-10 else 0.0
        result.update({
            'n_ctrl': len(ctrl), 'n_mut': len(mut),
            'ctrl_mean': float(np.mean(ctrl)), 'ctrl_median': float(np.median(ctrl)),
            'ctrl_sd': float(np.std(ctrl, ddof=1)),
            'mut_mean': float(np.mean(mut)), 'mut_median': float(np.median(mut)),
            'mut_sd': float(np.std(mut, ddof=1)),
            'U': float(U), 'p': float(p), 'cohens_d': float(d),
        })
        return result

    # Run tests on per-recording averages (PRIMARY)
    global_tests = {}
    global_tests['spike_rate_raw'] = _run_mw_test(ctrl_rec_rates, mut_rec_rates,
                                                   'Event rate (events/10s)')
    global_tests['spike_amplitude_raw'] = _run_mw_test(ctrl_rec_amps, mut_rec_amps,
                                                        'Transient amplitude (ΔF/F₀)')
    global_tests['pairwise_correlation'] = _run_mw_test(ctrl_corr, mut_corr,
                                                         'Pairwise Correlation (r)')
    global_tests['synchrony_index'] = _run_mw_test(ctrl_sync, mut_sync,
                                                    'Synchrony Index')
    global_tests['active_fraction'] = _run_mw_test(ctrl_af, mut_af,
                                                    'Active Fraction')

    results['tests']['global_pooled'] = global_tests

    for k, t in global_tests.items():
        if 'p' in t:
            logger.info(f"  Global {k}: p={t['p']:.4f}, d={t['cohens_d']:.3f}, "
                        f"ctrl={t['ctrl_mean']:.3f}+/-{t['ctrl_sd']:.3f}, "
                        f"mut={t['mut_mean']:.3f}+/-{t['mut_sd']:.3f}")

    # =====================================================================
    # SECTION 2: WITHIN-DAY COMPARISON (paired by organoid day)
    # =====================================================================
    logger.info("=== Within-day genotype comparison ===")

    unique_days = sorted(set(organoid_ids))
    within_day_results = OrderedDict()
    day_effect_sizes = []  # for meta-analysis

    for day in unique_days:
        day_ctrl = [ds for ds, g, o in geno_datasets if o == day and g == 'Control']
        day_mut = [ds for ds, g, o in geno_datasets if o == day and g == 'Mutant']

        if not day_ctrl or not day_mut:
            logger.info(f"  {day}: skipped (ctrl={len(day_ctrl)}, mut={len(day_mut)})")
            within_day_results[day] = {
                'n_ctrl_recordings': len(day_ctrl),
                'n_mut_recordings': len(day_mut),
                'skipped': True,
            }
            continue

        # Per-recording mean spike rates (active neurons only)
        day_ctrl_rates = np.array([float(np.mean(r)) for ds in day_ctrl
                                   for r in [_ds_spike_rates(ds)] if len(r) > 0])
        day_mut_rates = np.array([float(np.mean(r)) for ds in day_mut
                                  for r in [_ds_spike_rates(ds)] if len(r) > 0])

        day_result = {
            'n_ctrl_recordings': len(day_ctrl),
            'n_mut_recordings': len(day_mut),
            'n_ctrl_with_active': len(day_ctrl_rates),
            'n_mut_with_active': len(day_mut_rates),
        }

        # Raw test (per-recording means)
        day_result['spike_rate_raw'] = _run_mw_test(
            day_ctrl_rates, day_mut_rates, f'{day} Event Rate')

        # Per-recording metrics for this day
        day_ctrl_corr = np.array([ds.pairwise_correlation_mean for ds in day_ctrl])
        day_mut_corr  = np.array([ds.pairwise_correlation_mean for ds in day_mut])
        day_ctrl_sync = np.array([ds.synchrony_index for ds in day_ctrl])
        day_mut_sync  = np.array([ds.synchrony_index for ds in day_mut])
        # Filter NaN (n_selected < 5) before within-day tests
        day_ctrl_corr = day_ctrl_corr[np.isfinite(day_ctrl_corr)]
        day_mut_corr  = day_mut_corr[np.isfinite(day_mut_corr)]
        day_ctrl_sync = day_ctrl_sync[np.isfinite(day_ctrl_sync)]
        day_mut_sync  = day_mut_sync[np.isfinite(day_mut_sync)]

        day_result['correlation'] = _run_mw_test(
            day_ctrl_corr, day_mut_corr, f'{day} Correlation')
        day_result['synchrony'] = _run_mw_test(
            day_ctrl_sync, day_mut_sync, f'{day} Synchrony')

        within_day_results[day] = day_result

        # Store effect size for meta-analysis (per-recording spike rate)
        t = day_result['spike_rate_raw']
        if 'cohens_d' in t:
            n_c, n_m = t['n_ctrl'], t['n_mut']
            se_d = np.sqrt((n_c + n_m) / (n_c * n_m) + t['cohens_d']**2 / (2 * (n_c + n_m)))
            day_effect_sizes.append({
                'day': day, 'd': t['cohens_d'], 'se': se_d,
                'n_ctrl': n_c, 'n_mut': n_m,
            })

        logger.info(f"  {day}: ctrl={len(day_ctrl_rates)}rec/{len(day_ctrl)}total, "
                     f"mut={len(day_mut_rates)}rec/{len(day_mut)}total, "
                     f"raw {_fmt_p(day_result['spike_rate_raw']['p']) if 'p' in day_result.get('spike_rate_raw', {}) else 'skipped'}")

    results['tests']['within_day'] = within_day_results

    # =====================================================================
    # SECTION 3: PAIRED META-ANALYSIS ACROSS DAYS
    # =====================================================================
    meta_result = {'n_paired_days': len(day_effect_sizes)}
    if len(day_effect_sizes) >= 2:
        ds_arr = np.array([e['d'] for e in day_effect_sizes])
        se_arr = np.array([e['se'] for e in day_effect_sizes])
        weights = 1.0 / (se_arr**2 + 1e-10)
        pooled_d = np.sum(weights * ds_arr) / np.sum(weights)
        pooled_se = 1.0 / np.sqrt(np.sum(weights))
        z_meta = pooled_d / pooled_se if pooled_se > 1e-10 else 0.0
        p_meta = 2 * (1 - sp_stats.norm.cdf(abs(z_meta)))

        meta_result.update({
            'pooled_cohens_d': float(pooled_d),
            'pooled_se': float(pooled_se),
            'z': float(z_meta),
            'p': float(p_meta),
            'per_day': day_effect_sizes,
            'interpretation': (
                'Positive d = Control > Mutant spike rate. '
                'This meta-analysis weights each day by inverse variance, '
                'controlling for day-to-day variation in imaging conditions.'
            ),
        })
        logger.info(f"  Meta-analysis: pooled d={pooled_d:.3f}, p={p_meta:.4f} "
                     f"({len(day_effect_sizes)} paired days)")
    else:
        meta_result['skipped'] = True
        meta_result['reason'] = f"Need >= 2 days with both genotypes, got {len(day_effect_sizes)}"

    results['tests']['meta_analysis'] = meta_result

    # =====================================================================
    # FIGURES — clean white-bg individual PNGs for genotype comparison
    # =====================================================================
    CTRL_COL = CTRL_COLOR   # '#4472C4' blue
    MUT_COL  = MUT_COLOR    # '#ED7D31' orange

    def _clean_boxplot(ax, ctrl_vals, mut_vals, ylabel, test_key):
        """White-background boxplot panel for genotype comparison."""
        data = [ctrl_vals, mut_vals]
        colors = [CTRL_COL, MUT_COL]

        if all(len(d) > 0 for d in data):
            bp = ax.boxplot(data, positions=[0, 1], widths=0.5,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='#E53935', linewidth=2.5),
                            whiskerprops=dict(color='#666', linewidth=1.0),
                            capprops=dict(color='#666', linewidth=1.0))
            for patch, col in zip(bp['boxes'], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.35)
                patch.set_edgecolor(col)
                patch.set_linewidth(1.5)

        for i, (vals, col) in enumerate(zip(data, colors)):
            if len(vals) > 0:
                jitter_x = rng.uniform(-0.12, 0.12, len(vals))
                ax.scatter(i + jitter_x, vals, c=col, s=60, alpha=0.75,
                           zorder=5, edgecolors='white', linewidth=0.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', mutant_label], fontsize=16, fontweight='bold')
        for tick_label, col in zip(ax.get_xticklabels(), colors):
            tick_label.set_color(col)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)

        test = global_tests.get(test_key, {})
        if 'p' in test:
            stars = _sig_stars(test['p'])
            _cv = ctrl_vals[np.isfinite(ctrl_vals)] if len(ctrl_vals) > 0 else ctrl_vals
            _mv = mut_vals[np.isfinite(mut_vals)]   if len(mut_vals)  > 0 else mut_vals
            y_max = max(float(np.max(_cv)) if len(_cv) > 0 else 0,
                        float(np.max(_mv)) if len(_mv) > 0 else 0)
            if y_max > 0:
                _draw_sig_bracket(ax, 0, 1, y_max * 1.05, y_max * 0.06,
                                  f'{stars}\n{_fmt_p(test["p"])}', fontsize=13,
                                  color='#333')
                ax.set_ylim(top=y_max * 1.25)

    # ── Individual metric comparison figures ─────────────────────────────
    individual_panels = [
        ('active_fraction',        ctrl_af,         mut_af,       'Active fraction',                 'active_fraction'),
        ('event_rate',             ctrl_rec_rates,  mut_rec_rates,'Event rate (events/10s)',          'spike_rate_raw'),
        ('event_amplitude',        ctrl_rec_amps,   mut_rec_amps, 'Mean transient amplitude (ΔF/F₀)','spike_amplitude_raw'),
        ('pairwise_correlation',   ctrl_corr[np.isfinite(ctrl_corr)],
                                   mut_corr[np.isfinite(mut_corr)],
                                                                  'Mean pairwise correlation (r)',   'pairwise_correlation'),
        ('synchrony_index',        ctrl_sync[np.isfinite(ctrl_sync)],
                                   mut_sync[np.isfinite(mut_sync)],
                                                                  'Synchrony index',                 'synchrony_index'),
    ]

    for fname, cv, mv, ylabel, test_key in individual_panels:
        fig, ax = plt.subplots(figsize=(5.5, 6.5))
        _clean_boxplot(ax, cv, mv, ylabel, test_key)
        n_c = len(cv[np.isfinite(cv)]) if len(cv) > 0 else 0
        n_m = len(mv[np.isfinite(mv)]) if len(mv) > 0 else 0
        fig.text(0.5, 0.01,
                 f'Control: {n_c}  |  Mutant: {n_m} recordings  |  Mann-Whitney U',
                 ha='center', fontsize=11, color='#888', style='italic')
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(os.path.join(geno_dir, f'genotype_{fname}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # ── Combined activity figure (1×3: rate, amplitude, active fraction) ──
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 7))
    combined_panels = [
        (ctrl_rec_rates, mut_rec_rates, 'Event rate (events/10s)', 'spike_rate_raw'),
        (ctrl_rec_amps,  mut_rec_amps,  'Mean transient amplitude (ΔF/F₀)', 'spike_amplitude_raw'),
        (ctrl_af,        mut_af,        'Active fraction',                   'active_fraction'),
    ]
    for ax, (cv, mv, ylabel, key) in zip(axes1, combined_panels):
        _clean_boxplot(ax, cv, mv, ylabel, key)
    fig1.suptitle('Genotype comparison: activity metrics', fontsize=18, fontweight='bold')
    fig1.text(0.5, 0.01,
              f'Control: {n_ctrl}  |  Mutant: {n_mut} recordings  |  Mann-Whitney U',
              ha='center', fontsize=11, color='#888', style='italic')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig1.savefig(os.path.join(geno_dir, 'genotype_activity_combined.png'),
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)

    # ── Active fraction vs pairwise correlation scatter ──────────────────
    fig_scatter, ax_sc = plt.subplots(figsize=(8, 7))
    ctrl_corr_fin = ctrl_corr[np.isfinite(ctrl_corr)]
    mut_corr_fin  = mut_corr[np.isfinite(mut_corr)]
    # Match active fractions to recordings with valid correlation
    ctrl_af_fin = np.array([ds.active_fraction for ds in ctrl_ds
                            if np.isfinite(ds.pairwise_correlation_mean)])
    mut_af_fin  = np.array([ds.active_fraction for ds in mut_ds
                            if np.isfinite(ds.pairwise_correlation_mean)])
    ctrl_names_fin = [ds.name for ds in ctrl_ds
                      if np.isfinite(ds.pairwise_correlation_mean)]
    mut_names_fin  = [ds.name for ds in mut_ds
                      if np.isfinite(ds.pairwise_correlation_mean)]

    ax_sc.scatter(ctrl_af_fin, ctrl_corr_fin, c=CTRL_COL, s=50, alpha=0.7,
                  edgecolors='white', linewidth=0.5, label='Control', zorder=5)
    ax_sc.scatter(mut_af_fin, mut_corr_fin, c=MUT_COL, s=50, alpha=0.7,
                  edgecolors='white', linewidth=0.5, label=mutant_label, zorder=5)

    # Label outliers (IQR method: > Q3 + 1.5*IQR in either dimension)
    all_af = np.concatenate([ctrl_af_fin, mut_af_fin])
    all_corr = np.concatenate([ctrl_corr_fin, mut_corr_fin])

    def _iqr_upper_fence(values):
        """Compute upper outlier fence: Q3 + 1.5 * IQR."""
        q1, q3 = np.percentile(values, [25, 75])
        return q3 + 1.5 * (q3 - q1)

    af_thresh = _iqr_upper_fence(all_af) if len(all_af) > 5 else np.inf
    corr_thresh = _iqr_upper_fence(all_corr) if len(all_corr) > 5 else np.inf

    # Collect outlier details for reporting
    outlier_records = []

    for af_arr, corr_arr, names, col, geno_label in [
        (ctrl_af_fin, ctrl_corr_fin, ctrl_names_fin, CTRL_COL, 'Control'),
        (mut_af_fin,  mut_corr_fin,  mut_names_fin,  MUT_COL, mutant_label),
    ]:
        for i in range(len(af_arr)):
            af_out = af_arr[i] > af_thresh
            corr_out = corr_arr[i] > corr_thresh
            if af_out or corr_out:
                # Abbreviate name for label
                short = names[i].split('_')[0] + '_' + names[i].split('_')[1] if '_' in names[i] else names[i][:15]
                ax_sc.annotate(short, (af_arr[i], corr_arr[i]),
                               fontsize=6, color=col, alpha=0.8,
                               xytext=(5, 5), textcoords='offset points')
                outlier_records.append({
                    'recording': names[i],
                    'genotype': geno_label,
                    'active_fraction': float(af_arr[i]),
                    'pairwise_corr': float(corr_arr[i]),
                    'outlier_af': af_out,
                    'outlier_corr': corr_out,
                })

    # Log outlier results
    af_q1, af_q3 = np.percentile(all_af, [25, 75])
    corr_q1, corr_q3 = np.percentile(all_corr, [25, 75])
    n_ctrl_out = sum(1 for o in outlier_records if o['genotype'] == 'Control')
    n_mut_out = sum(1 for o in outlier_records if o['genotype'] != 'Control')

    logger.info(f"  Outlier detection (IQR method, Q3 + 1.5*IQR):")
    logger.info(f"    Active fraction:      Q1={af_q1:.4f}, Q3={af_q3:.4f}, "
                f"IQR={af_q3-af_q1:.4f}, upper fence={af_thresh:.4f}")
    logger.info(f"    Pairwise correlation: Q1={corr_q1:.4f}, Q3={corr_q3:.4f}, "
                f"IQR={corr_q3-corr_q1:.4f}, upper fence={corr_thresh:.4f}")
    logger.info(f"    Outliers: {len(outlier_records)} total "
                f"(Control: {n_ctrl_out}, {mutant_label}: {n_mut_out})")
    for o in outlier_records:
        flags = []
        if o['outlier_af']:
            flags.append(f"AF={o['active_fraction']:.3f}")
        if o['outlier_corr']:
            flags.append(f"corr={o['pairwise_corr']:.3f}")
        logger.info(f"      {o['recording']} ({o['genotype']}): {', '.join(flags)}")

    # Write outlier report to file
    outlier_report_path = os.path.join(geno_dir, 'outlier_report.txt')
    with open(outlier_report_path, 'w') as f_out:
        f_out.write("Outlier Detection Report: Active Fraction vs Pairwise Correlation\n")
        f_out.write("=" * 70 + "\n\n")
        f_out.write("Method: IQR (Interquartile Range) upper fence\n")
        f_out.write("Criterion: value > Q3 + 1.5 * IQR in either metric\n")
        f_out.write(f"N recordings: {len(all_af)} "
                    f"(Control: {len(ctrl_af_fin)}, {mutant_label}: {len(mut_af_fin)})\n\n")
        f_out.write("Thresholds:\n")
        f_out.write(f"  Active fraction:      Q1={af_q1:.4f}, Q3={af_q3:.4f}, "
                    f"IQR={af_q3-af_q1:.4f}, upper fence={af_thresh:.4f}\n")
        f_out.write(f"  Pairwise correlation: Q1={corr_q1:.4f}, Q3={corr_q3:.4f}, "
                    f"IQR={corr_q3-corr_q1:.4f}, upper fence={corr_thresh:.4f}\n\n")
        f_out.write(f"Outliers identified: {len(outlier_records)} "
                    f"(Control: {n_ctrl_out}, {mutant_label}: {n_mut_out})\n")
        f_out.write("-" * 70 + "\n")
        f_out.write(f"{'Recording':<35} {'Genotype':<15} {'AF':>8} {'Corr':>8} {'Flag':>15}\n")
        f_out.write("-" * 70 + "\n")
        for o in outlier_records:
            flags = []
            if o['outlier_af']:
                flags.append('AF')
            if o['outlier_corr']:
                flags.append('Corr')
            f_out.write(f"{o['recording']:<35} {o['genotype']:<15} "
                        f"{o['active_fraction']:>8.4f} {o['pairwise_corr']:>8.4f} "
                        f"{'  '.join(flags):>15}\n")
        f_out.write("-" * 70 + "\n")
    logger.info(f"  Outlier report saved to {outlier_report_path}")

    ax_sc.set_xlabel('Active fraction', fontsize=16)
    ax_sc.set_ylabel('Mean pairwise correlation (r)', fontsize=16)
    ax_sc.spines['top'].set_visible(False)
    ax_sc.spines['right'].set_visible(False)
    ax_sc.grid(alpha=0.15)
    ax_sc.legend(fontsize=14)
    ax_sc.tick_params(labelsize=13)
    plt.tight_layout()
    fig_scatter.savefig(os.path.join(geno_dir, 'genotype_af_vs_correlation.png'),
                        dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_scatter)

    # ── Figure: Event rate + amplitude: Event rate + amplitude (1×2 panel) ───────
    fig_r3, (ax_r3a, ax_r3b) = plt.subplots(1, 2, figsize=(12, 7))
    _clean_boxplot(ax_r3a, ctrl_rec_rates, mut_rec_rates,
                   'Event rate (events/10s)', 'spike_rate_raw')
    _clean_boxplot(ax_r3b, ctrl_rec_amps, mut_rec_amps,
                   'Mean transient amplitude (ΔF/F₀)', 'spike_amplitude_raw')
    ax_r3a.text(-0.08, 1.05, 'A', transform=ax_r3a.transAxes,
                fontsize=22, fontweight='bold', va='top')
    ax_r3b.text(-0.08, 1.05, 'B', transform=ax_r3b.transAxes,
                fontsize=22, fontweight='bold', va='top')
    fig_r3.text(0.5, 0.01,
                f'Control: {n_ctrl}  |  Mutant: {n_mut} recordings  |  '
                f'Mann-Whitney U, two-sided',
                ha='center', fontsize=11, color='#888', style='italic')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig_r3.savefig(os.path.join(geno_dir, 'genotype_rate_amplitude.png'),
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_r3)

    # ── Figure: Developmental trajectory: Developmental trajectory (2×2 panel) ─────
    try:
        import re as _re_r7
        from scipy import stats as _stats_r7

        def _day_sort_key_r7(d):
            nums = _re_r7.findall(r'\d+', d)
            return int(nums[0]) if nums else 0

        all_days_r7 = OrderedDict()
        for ds, g, o in geno_datasets:
            all_days_r7.setdefault(o, OrderedDict())
            all_days_r7[o].setdefault(g, []).append(ds)
        sorted_days_r7 = sorted(all_days_r7.keys(), key=_day_sort_key_r7)
        day_x_r7 = {d: i for i, d in enumerate(sorted_days_r7)}

        geno_display_r7 = {'Control': 'Control', 'Mutant': mutant_label}

        r7_metrics = [
            ('active_fraction', 'Active fraction'),
            ('mean_spike_rate', 'Event rate (events/10s)'),
            ('mean_spike_amplitude', 'Mean transient amplitude (ΔF/F₀)'),
            ('total_events', 'Total events per recording'),
        ]

        fig_r7, axes_r7 = plt.subplots(2, 2, figsize=(16, 12))
        axes_r7 = axes_r7.ravel()
        panel_labels = ['A', 'B', 'C', 'D']

        for ax, (attr, ylabel), plabel in zip(axes_r7, r7_metrics, panel_labels):
            ax.set_facecolor('white')
            ax.text(-0.08, 1.05, plabel, transform=ax.transAxes,
                    fontsize=22, fontweight='bold', va='top')

            for geno, color in [('Control', CTRL_COLOR), ('Mutant', MUT_COLOR)]:
                # Collect per-day mean and SEM
                day_means, day_sems, day_xs = [], [], []
                for day in sorted_days_r7:
                    gds = all_days_r7[day].get(geno, [])
                    if gds:
                        day_vals = [v for v in (_recording_metric(d, attr) for d in gds)
                                    if v is not None and np.isfinite(v)]
                        if day_vals:
                            day_xs.append(day_x_r7[day])
                            day_means.append(np.mean(day_vals))
                            sem = np.std(day_vals, ddof=1) / np.sqrt(len(day_vals)) if len(day_vals) > 1 else 0.0
                            day_sems.append(sem)

                if not day_xs:
                    continue

                day_xs = np.array(day_xs, dtype=float)
                day_means = np.array(day_means)
                day_sems = np.array(day_sems)

                # Slight x-offset to separate Control and Mutant
                x_offset = -0.1 if geno == 'Control' else 0.1

                ax.errorbar(day_xs + x_offset, day_means, yerr=day_sems,
                            fmt='o-', color=color, markersize=7, linewidth=2,
                            capsize=4, capthick=1.5, elinewidth=1.5,
                            alpha=0.85, zorder=6, label=geno_display_r7[geno])

                # Individual data points (small, behind error bars)
                for day in sorted_days_r7:
                    gds = all_days_r7[day].get(geno, [])
                    day_vals = [v for v in (_recording_metric(d, attr) for d in gds)
                                if v is not None and np.isfinite(v)]
                    if day_vals:
                        jitter = rng.uniform(-0.06, 0.06, len(day_vals))
                        ax.scatter(np.full(len(day_vals), day_x_r7[day]) + x_offset + jitter,
                                   day_vals, c=color, s=16, alpha=0.35,
                                   edgecolor='none', zorder=4)

            ax.set_xticks(range(len(sorted_days_r7)))
            ax.set_xticklabels(sorted_days_r7, fontsize=13, rotation=45, ha='right')
            ax.set_ylabel(ylabel, fontsize=15)
            ax.set_xlabel('Developmental day', fontsize=15)
            ax.tick_params(labelsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.15)
            ax.legend(fontsize=13)

            # Robust y-limits: set bottom to 0, clip top to 97th percentile
            all_y = []
            for line in ax.lines:
                ydata = line.get_ydata()
                if len(ydata) > 0:
                    all_y.extend(ydata.tolist())
            if len(all_y) > 3:
                y_hi = np.percentile(all_y, 97) * 1.15
                ax.set_ylim(bottom=0, top=y_hi)

        fig_r7.suptitle(f'Developmental trajectory: Control vs {mutant_label}',
                        fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_r7.savefig(os.path.join(geno_dir, 'genotype_longitudinal.png'),
                       dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_r7)
    except Exception as e:
        logger.warning(f"Longitudinal figure failed: {e}")

    global_path = os.path.join(geno_dir, 'genotype_activity_combined.png')

    # ── Figure 3: Within-day breakdown ───────────────────────────────────
    paired_days = [d for d in unique_days
                   if not within_day_results.get(d, {}).get('skipped', True)]

    if paired_days:
        n_days = len(paired_days)
        fig, axes = plt.subplots(1, n_days, figsize=(max(6, n_days * 4.5), 6),
                                 squeeze=False)
        fig.patch.set_facecolor('white')

        for i, day in enumerate(paired_days):
            ax = axes[0, i]
            ax.set_facecolor('white')

            day_ctrl = [ds for ds, g, o in geno_datasets if o == day and g == 'Control']
            day_mut = [ds for ds, g, o in geno_datasets if o == day and g == 'Mutant']

            # Per-recording mean spike rates (active neurons only)
            c_rates = np.array([float(np.mean(r)) for ds in day_ctrl
                                for r in [_ds_spike_rates(ds)] if len(r) > 0])
            m_rates = np.array([float(np.mean(r)) for ds in day_mut
                                for r in [_ds_spike_rates(ds)] if len(r) > 0])

            if len(c_rates) == 0 or len(m_rates) == 0:
                continue

            bp = ax.boxplot([c_rates, m_rates], positions=[0, 1], widths=0.5,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='white', linewidth=1.5),
                            whiskerprops=dict(color='#555', linewidth=0.8),
                            capprops=dict(color='#555', linewidth=0.8))
            for patch, col_c in zip(bp['boxes'], [CTRL_COLOR, MUT_COLOR]):
                patch.set_facecolor(col_c)
                patch.set_alpha(0.35)
                patch.set_edgecolor(col_c)

            for j, (vals, col_c) in enumerate(zip([c_rates, m_rates],
                                                   [CTRL_COLOR, MUT_COLOR])):
                jitter = rng.uniform(-0.15, 0.15, len(vals))
                ax.scatter(j + jitter, vals, c=col_c, s=40, alpha=0.7,
                           zorder=5, edgecolors='white', linewidth=0.5)

            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'Ctrl\n(n={len(c_rates)} rec)',
                                f'Mut\n(n={len(m_rates)} rec)'],
                               fontsize=9)
            for tick, col_c in zip(ax.get_xticklabels(), [CTRL_COLOR, MUT_COLOR]):
                tick.set_color(col_c)

            # Significance
            wd = within_day_results[day]
            t_raw = wd.get('spike_rate_raw', {})
            if 'p' in t_raw:
                stars = _sig_stars(t_raw['p'])
                _cr = c_rates[np.isfinite(c_rates)]
                _mr = m_rates[np.isfinite(m_rates)]
                y_max = max(float(np.max(_cr)) if len(_cr) > 0 else 0,
                            float(np.max(_mr)) if len(_mr) > 0 else 0)
                _draw_sig_bracket(ax, 0, 1, y_max * 1.05, y_max * 0.06,
                                  f"{stars} {_fmt_p(t_raw['p'])}", fontsize=8)
                ax.set_ylim(top=y_max * 1.3)

            ax.set_title(f'{day}\n({len(day_ctrl)} ctrl, {len(day_mut)} mut rec)',
                         fontsize=10, fontweight='bold')
            ax.set_ylabel('Event rate (events/10s)' if i == 0 else '', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.15)

        fig.suptitle(f'Within-Day Genotype Comparison: Control vs {mutant_label}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        within_path = os.path.join(geno_dir, 'genotype_within_day.png')
        plt.savefig(within_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    # ── Figure 4: Meta-analysis forest plot ──────────────────────────────
    if len(day_effect_sizes) >= 2:
        fig, ax = plt.subplots(figsize=(10, max(4, len(day_effect_sizes) * 0.8 + 2)))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        y_positions = list(range(len(day_effect_sizes)))
        day_labels = []

        for y, es in zip(y_positions, day_effect_sizes):
            d_val = es['d']
            se = es['se']
            ci_lo = d_val - 1.96 * se
            ci_hi = d_val + 1.96 * se

            ax.plot([ci_lo, ci_hi], [y, y], color='#555', linewidth=1.5)
            ax.scatter(d_val, y, color=CTRL_COLOR if d_val > 0 else MUT_COLOR,
                       s=80, zorder=5, edgecolor='white', linewidth=0.5)
            day_labels.append(f"{es['day']} (n={es['n_ctrl']}+{es['n_mut']})")

        # Pooled estimate
        pooled = meta_result.get('pooled_cohens_d', 0)
        pooled_se = meta_result.get('pooled_se', 0)
        y_pooled = len(day_effect_sizes) + 0.5
        ax.axhline(y_pooled - 0.3, color='#ccc', linewidth=0.5)
        ax.plot([pooled - 1.96*pooled_se, pooled + 1.96*pooled_se],
                [y_pooled, y_pooled], color='#333', linewidth=2.5)
        ax.scatter(pooled, y_pooled, color='#333', s=120, marker='D',
                   zorder=5, edgecolor='white')
        day_labels.append(f"POOLED (p={meta_result.get('p', 1):.3f})")

        ax.axvline(0, color='#999', linestyle='--', linewidth=1, zorder=1)

        ax.set_yticks(y_positions + [y_pooled])
        ax.set_yticklabels(day_labels, fontsize=9)
        ax.set_xlabel("Cohen's d (positive = Control > Mutant)", fontsize=10)
        ax.set_title('Meta-Analysis: Z-Scored Event Rate Effect Sizes by Day',
                     fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

        plt.tight_layout()
        forest_path = os.path.join(geno_dir, 'genotype_meta_analysis.png')
        plt.savefig(forest_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    # ── Figure 5: Longitudinal developmental trajectory ──────────────────
    # Shows how each metric evolves over day-age, split by genotype,
    # addressing the secondary objective of tracking the relationship over time
    try:
        import re as _re

        def _day_sort_key(d):
            nums = _re.findall(r'\d+', d)
            return int(nums[0]) if nums else 0

        # Build day -> genotype -> [datasets] mapping from geno_datasets
        all_days = OrderedDict()
        for ds, g, o in geno_datasets:
            day = o  # organoid_id is the day-age (e.g. D109)
            all_days.setdefault(day, OrderedDict())
            all_days[day].setdefault(g, []).append(ds)
        sorted_all_days = sorted(all_days.keys(), key=_day_sort_key)
        day_x = {d: i for i, d in enumerate(sorted_all_days)}

        longit_metrics = [
            ('mean_spike_rate', 'Event rate (events/10s)'),
            ('pairwise_correlation_mean', 'Mean pairwise correlation (r)'),
            ('synchrony_index', 'Synchrony index'),
            ('mean_spike_amplitude', 'Mean transient amplitude (ΔF/F₀)'),
            ('active_fraction', 'Active fraction'),
        ]

        # Helper: get per-recording metric value consistent with global
        # comparison (active-neuron-only means for rate/amplitude).
        def _longit_value(ds, attr):
            return _recording_metric(ds, attr)

        n_metrics = len(longit_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig_long, axes_long = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 6))
        fig_long.patch.set_facecolor('white')
        axes_long = axes_long.ravel()
        # Hide unused axes
        for ax_i in range(n_metrics, len(axes_long)):
            axes_long[ax_i].set_visible(False)

        geno_display = {'Control': 'Control', 'Mutant': mutant_label}
        for ax, (attr, ylabel) in zip(axes_long, longit_metrics):
            ax.set_facecolor('white')
            for geno, color in [('Control', CTRL_COLOR), ('Mutant', MUT_COLOR)]:
                x_vals, y_vals = [], []
                for day in sorted_all_days:
                    for ds in all_days[day].get(geno, []):
                        val = _longit_value(ds, attr)
                        # Skip None and NaN (corr/sync excluded due to n < 5)
                        if val is not None and np.isfinite(val):
                            x_vals.append(day_x[day])
                            y_vals.append(val)

                if not x_vals:
                    continue

                jitter = rng.uniform(-0.12, 0.12, len(x_vals))
                ax.scatter(np.array(x_vals) + jitter, y_vals, c=color,
                           s=40, alpha=0.6, edgecolor='white', linewidth=0.5,
                           zorder=5, label=geno_display[geno])

                # Mean trend line connecting days
                dmx, dmy = [], []
                for day in sorted_all_days:
                    gds = all_days[day].get(geno, [])
                    if gds:
                        day_vals = [v for v in (_longit_value(d, attr) for d in gds)
                                    if v is not None and np.isfinite(v)]
                        if day_vals:
                            dmx.append(day_x[day])
                            dmy.append(np.mean(day_vals))
                if len(dmx) > 1:
                    ax.plot(dmx, dmy, color=color, linewidth=2, alpha=0.7,
                            marker='o', markersize=6, zorder=6)

            ax.set_xticks(range(len(sorted_all_days)))
            ax.set_xticklabels(sorted_all_days, fontsize=9, rotation=45, ha='right')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel('Day Age', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.2)
            ax.legend(fontsize=9)

            # Robust y-limits: clip to 95th percentile to prevent outlier stretching
            all_y = [v for v in ax.collections[0].get_offsets()[:, 1]] if ax.collections else []
            for coll in ax.collections:
                offs = coll.get_offsets()
                if len(offs) > 0:
                    all_y.extend(offs[:, 1].tolist())
            if len(all_y) > 5:
                y_lo = max(0, np.percentile(all_y, 1) - 0.01)
                y_hi = np.percentile(all_y, 97) * 1.15
                ax.set_ylim(bottom=y_lo, top=y_hi)

        fig_long.suptitle('Developmental trajectory: mutant vs control',
                          fontsize=14, fontweight='bold')
        plt.tight_layout()
        longit_path = os.path.join(geno_dir, 'genotype_longitudinal.png')
        plt.savefig(longit_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    except Exception as e:
        logger.warning(f"Longitudinal figure failed: {e}")

    # ── Figure 5b: Per-cell-line across days ─────────────────────────────
    # For each metric, show side-by-side boxplots of every cell line at each
    # day-age so the user can see how individual lines behave over time
    try:
        import re as _re2

        # Build line -> day -> [metric values] mapping
        line_day_data = {}  # {line_id: {day: [ds, ...]}}
        for ds in datasets:
            line_id = _extract_line_id(ds.name)
            day = _extract_organoid_id(ds.name)
            line_day_data.setdefault(line_id, {}).setdefault(day, []).append(ds)

        all_lines = sorted(line_day_data.keys())
        all_days_set = set()
        for ld in line_day_data.values():
            all_days_set.update(ld.keys())

        def _day_num(d):
            nums = _re2.findall(r'\d+', d)
            return int(nums[0]) if nums else 0

        sorted_days = sorted(all_days_set, key=_day_num)
        n_lines = len(all_lines)
        n_days = len(sorted_days)

        if n_lines > 1 and n_days > 1:
            line_metrics = [
                ('mean_spike_rate', 'Event rate (events/10s)'),
                ('pairwise_correlation_mean', 'Pairwise Correlation (r)'),
                ('synchrony_index', 'Synchrony Index'),
                ('mean_spike_amplitude', 'Transient amplitude (ΔF/F₀)'),
                ('active_fraction', 'Active Fraction (selected/total)'),
            ]

            # Assign each line a colour: control lines in blue shades, mutant in orange
            import matplotlib.cm as _cm
            ctrl_lines = [l for l in all_lines if l.startswith('3')]
            mut_lines = [l for l in all_lines if not l.startswith('3')]
            blues = _cm.Blues(np.linspace(0.35, 0.85, max(len(ctrl_lines), 1)))
            oranges = _cm.Oranges(np.linspace(0.35, 0.85, max(len(mut_lines), 1)))
            line_colors = {}
            for i, l in enumerate(ctrl_lines):
                line_colors[l] = blues[i]
            for i, l in enumerate(mut_lines):
                line_colors[l] = oranges[i]

            fig_lines, axes_lines = plt.subplots(len(line_metrics), 1,
                                                  figsize=(max(14, n_days * n_lines * 0.6), 4 * len(line_metrics)))
            fig_lines.patch.set_facecolor('white')
            if len(line_metrics) == 1:
                axes_lines = [axes_lines]

            for ax, (attr, ylabel) in zip(axes_lines, line_metrics):
                ax.set_facecolor('white')

                # For each day, draw grouped boxplots — one per line
                positions = []
                box_data = []
                box_colors = []
                tick_positions = []
                tick_labels = []
                group_width = n_lines * 0.7 + 0.5
                x = 0

                for day in sorted_days:
                    day_start = x
                    has_data = False
                    for li, line_id in enumerate(all_lines):
                        ds_list = line_day_data.get(line_id, {}).get(day, [])
                        vals = [v for v in (_longit_value(ds, attr) for ds in ds_list) if v is not None]
                        if vals:
                            has_data = True
                            positions.append(x)
                            box_data.append(vals)
                            box_colors.append(line_colors.get(line_id, '#999'))
                            x += 0.7
                        else:
                            x += 0.7

                    if has_data:
                        tick_positions.append((day_start + x - 0.7) / 2)
                        tick_labels.append(day)
                    x += 1.0  # gap between days

                if box_data:
                    bp = ax.boxplot(box_data, positions=positions, widths=0.55,
                                    patch_artist=True, showfliers=True,
                                    flierprops=dict(marker='.', markersize=3, alpha=0.5))
                    for patch, color in zip(bp['boxes'], box_colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    for median in bp['medians']:
                        median.set_color('white')
                        median.set_linewidth(1.5)

                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontsize=9, rotation=45, ha='right')
                ax.set_ylabel(ylabel, fontsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', alpha=0.2)

            # Legend
            legend_handles = []
            for line_id in all_lines:
                geno = 'Ctrl' if line_id.startswith('3') else 'Mut'
                legend_handles.append(Patch(
                    facecolor=line_colors.get(line_id, '#999'),
                    alpha=0.7, label=f'{line_id} ({geno})'))
            axes_lines[0].legend(handles=legend_handles, fontsize=8,
                                  ncol=min(n_lines, 6), loc='upper right',
                                  framealpha=0.8)

            fig_lines.suptitle('Per-Cell-Line Metrics Across Days',
                               fontsize=14, fontweight='bold')
            plt.tight_layout()
            lines_path = os.path.join(geno_dir, 'genotype_per_line_longitudinal.png')
            plt.savefig(lines_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Per-line longitudinal figure saved: {lines_path}")
        else:
            logger.info("Skipping per-line figure: need >1 line and >1 day")

    except Exception as e:
        logger.warning(f"Per-line longitudinal figure failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    # ── Figure 6: Summary statistics table ───────────────────────────────
    fig_table = plt.figure(figsize=(18, 10))
    fig_table.patch.set_facecolor('white')

    fig_table.text(0.5, 0.97, 'Genotype Comparison — Statistical Summary',
                   ha='center', fontsize=14, fontweight='bold')

    # Table A: Global comparison
    ax_t = fig_table.add_axes([0.05, 0.55, 0.90, 0.35])
    ax_t.axis('off')
    fig_table.text(0.5, 0.92, 'Global Pooled Comparison (all days)',
                   ha='center', fontsize=11, fontweight='bold')

    headers = ['Metric', 'Analysis', 'Ctrl mean +/- SD', 'Mut mean +/- SD',
               'U', 'p-value', "Cohen's d", 'Sig']
    rows = []
    for key in ['spike_rate_raw', 'spike_amplitude_raw',
                'pairwise_correlation', 'synchrony_index', 'active_fraction']:
        t = global_tests.get(key, {})
        if t.get('skipped') or 'metric' not in t:
            continue
        mode = 'Z-scored' if t.get('z_scored') else 'Raw'
        rows.append([
            t['metric'], mode,
            f"{t['ctrl_mean']:.3f} +/- {t['ctrl_sd']:.3f}",
            f"{t['mut_mean']:.3f} +/- {t['mut_sd']:.3f}",
            f"{t.get('U', 0):.0f}", _fmt_p(t['p']),
            f"{t['cohens_d']:.3f}", _sig_stars(t['p']),
        ])

    if rows:
        tbl = ax_t.table(cellText=rows, colLabels=headers,
                         loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor('#E6F1FB')
                cell.set_text_props(color='white', fontweight='bold')
            elif col == 7:  # Sig column
                txt = cell.get_text().get_text()
                if '***' in txt:
                    cell.set_facecolor('#C6EFCE')
                elif '**' in txt:
                    cell.set_facecolor('#D9F2D9')
                elif '*' == txt.strip():
                    cell.set_facecolor('#FFEB9C')
                else:
                    cell.set_facecolor('white')
            else:
                cell.set_facecolor('#F2F7FC' if row % 2 == 0 else 'white')
            cell.set_edgecolor('#DDD')

    # Table B: Within-day summary
    ax_t2 = fig_table.add_axes([0.05, 0.05, 0.90, 0.40])
    ax_t2.axis('off')
    fig_table.text(0.5, 0.48, 'Within-Day Breakdown (spike rate)',
                   ha='center', fontsize=11, fontweight='bold')

    headers2 = ['Day', 'Ctrl rec', 'Mut rec', 'Ctrl rate', 'Mut rate',
                'p-value', "Cohen's d"]
    rows2 = []
    for day in unique_days:
        wd = within_day_results.get(day, {})
        if wd.get('skipped'):
            rows2.append([day, str(wd.get('n_ctrl_recordings', 0)),
                         str(wd.get('n_mut_recordings', 0)),
                         '-', '-', '-', '-'])
            continue
        t_raw = wd.get('spike_rate_raw', {})
        rows2.append([
            day,
            str(wd.get('n_ctrl_with_active', wd.get('n_ctrl_recordings', 0))),
            str(wd.get('n_mut_with_active', wd.get('n_mut_recordings', 0))),
            f"{t_raw.get('ctrl_mean', 0):.2f}",
            f"{t_raw.get('mut_mean', 0):.2f}",
            _fmt_p(t_raw['p']) if 'p' in t_raw else '-',
            f"{t_raw.get('cohens_d', 0):.3f}" if 'cohens_d' in t_raw else '-',
        ])

    if rows2:
        tbl2 = ax_t2.table(cellText=rows2, colLabels=headers2,
                           loc='center', cellLoc='center')
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8)
        tbl2.scale(1, 1.5)
        for (row, col), cell in tbl2.get_celld().items():
            if row == 0:
                cell.set_facecolor('#E6F1FB')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#F2F7FC' if row % 2 == 0 else 'white')
            cell.set_edgecolor('#DDD')

    table_path = os.path.join(geno_dir, 'genotype_statistics_table.png')
    plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Genotype comparison figures saved to {geno_dir}")

    # ── Figure 7: Mixed-effects model analysis ───────────────────────────
    # Addresses the z-score limitation: z-scoring removes ALL between-recording
    # variance including genuine biological shifts. Mixed models separate
    # biological variance (genotype) from technical variance (recording).
    try:
        _run_mixed_model_analysis(
            datasets, genotypes, line_ids, organoid_ids,
            geno_dir, results)
    except Exception as e:
        logger.warning(f"Mixed-effects model analysis failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())

    return results


def _run_mixed_model_analysis(
    datasets: List[DatasetMetrics],
    genotypes: List[str],
    line_ids: List[str],
    organoid_ids: List[str],
    geno_dir: str,
    results: dict,
) -> None:
    """
    Mixed-effects model analysis of genotype effects.

    Addresses the limitation of z-scoring (which removes genuine global shifts)
    by using hierarchical models that separate technical variance (between
    recordings) from biological variance (between genotypes).

    Models:
    1. Recording-level LMM: metric ~ genotype + (1|organoid_day)
       Uses organoid day as random effect to absorb session-level confounds
       (same day = same imaging conditions). Conservative: one obs per recording.

    2. Neuron-level LMM: metric ~ genotype + (1|recording)
       Uses recording as random effect. More power but needs correct nesting.

    3. Permutation test: model-free, shuffles genotype labels across recording means.
    """

    logger.info("Running mixed-effects model analysis...")

    # ── Build dataset-level data ─────────────────────────────────────────
    ds_records = []
    for ds, geno, line, org_id in zip(datasets, genotypes, line_ids, organoid_ids):
        if geno not in ('Control', 'Mutant'):
            continue
        # Use active-neuron-only means for rate/amplitude, consistent with
        # global comparison and longitudinal figures.
        active_rates = _get_neuron_rates(ds)
        active_amps = _get_neuron_amplitudes(ds)
        rec_rate = float(np.mean(active_rates)) if len(active_rates) > 0 else None
        rec_amp = float(np.mean(active_amps)) if len(active_amps) > 0 else None
        # Skip recordings with no active neurons for rate/amplitude
        # (correlation and synchrony are always available)
        ds_records.append({
            'name': ds.name,
            'genotype': geno,
            'genotype_code': 0 if geno == 'Control' else 1,
            'line_id': line,
            'organoid_day': org_id,
            'spike_rate': rec_rate,
            'spike_amplitude': rec_amp,
            'correlation': ds.pairwise_correlation_mean,
            'synchrony': ds.synchrony_index,
            'n_neurons': ds.n_selected,
            'mean_quality': ds.mean_quality_score,
        })

    if len(ds_records) < 6:
        logger.warning("Too few datasets for mixed model analysis")
        return

    # ── Build neuron-level data ──────────────────────────────────────────
    neuron_records = []
    for ds, geno, line, org_id in zip(datasets, genotypes, line_ids, organoid_ids):
        if geno not in ('Control', 'Mutant'):
            continue
        rates = _get_neuron_rates(ds)
        amps = _get_neuron_amplitudes(ds)
        for j in range(len(rates)):
            neuron_records.append({
                'genotype': geno,
                'genotype_code': 0 if geno == 'Control' else 1,
                'line_id': line,
                'recording': ds.name,
                'organoid_day': org_id,
                'spike_rate': rates[j],
                'spike_amplitude': amps[j] if j < len(amps) else 0.0,
            })

    # ── Try statsmodels ──────────────────────────────────────────────────
    has_statsmodels = False
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        has_statsmodels = True
    except ImportError:
        logger.warning("statsmodels not available — permutation tests only")

    metrics_recording = [
        ('spike_rate', 'Event rate (events/10s)'),
        ('spike_amplitude', 'Transient amplitude (ΔF/F₀)'),
        ('correlation', 'Pairwise Correlation (r)'),
        ('synchrony', 'Synchrony Index'),
    ]
    metrics_neuron = [
        ('spike_rate', 'Event rate (events/10s)'),
        ('spike_amplitude', 'Transient amplitude (ΔF/F₀)'),
    ]

    mixed_results = {}

    if has_statsmodels:
        ds_df = pd.DataFrame(ds_records)
        neuron_df = pd.DataFrame(neuron_records)

        n_days = ds_df['organoid_day'].nunique()
        n_recs = ds_df['name'].nunique()
        logger.info(f"  {len(ds_df)} recordings, {n_days} unique days, "
                    f"{len(neuron_df)} neurons")

        for metric, label in metrics_recording:
            mr = {'metric': metric, 'label': label}

            # ── Recording-level: metric ~ genotype + (1|organoid_day) ────
            try:
                # Drop rows with missing values for this metric (e.g. no active neurons)
                metric_df = ds_df.dropna(subset=[metric])
                n_days_metric = metric_df['organoid_day'].nunique()
                if n_days_metric >= 3:
                    model = smf.mixedlm(
                        f"{metric} ~ genotype_code",
                        data=metric_df,
                        groups=metric_df['organoid_day'],
                    )
                    fit = model.fit(reml=True, method='lbfgs')

                    coef = float(fit.params.get('genotype_code', 0))
                    se = float(fit.bse.get('genotype_code', 0))
                    p = float(fit.pvalues.get('genotype_code', 1))
                    ci = fit.conf_int()
                    ci_low = float(ci.loc['genotype_code', 0]) if 'genotype_code' in ci.index else coef - 1.96*se
                    ci_high = float(ci.loc['genotype_code', 1]) if 'genotype_code' in ci.index else coef + 1.96*se

                    mr['recording_lmm'] = {
                        'coef': coef, 'se': se, 'p': p,
                        'ci_low': ci_low, 'ci_high': ci_high,
                        'n_obs': len(metric_df), 'n_groups': n_days_metric,
                        'formula': f'{metric} ~ genotype + (1|organoid_day)',
                    }
                    logger.info(f"  {metric} rec-LMM: coef={coef:.4f}, p={p:.4f}")
                else:
                    mr['recording_lmm'] = {'skipped': f'need ≥3 days, have {n_days_metric}'}
            except Exception as e:
                mr['recording_lmm'] = {'error': str(e)}
                logger.warning(f"  {metric} rec-LMM failed: {e}")

            # ── Neuron-level: metric ~ genotype + (1|recording) ──────────
            if metric in ('spike_rate', 'spike_amplitude'):
                try:
                    n_recs_neuron = neuron_df['recording'].nunique()
                    if n_recs_neuron >= 5:
                        model = smf.mixedlm(
                            f"{metric} ~ genotype_code",
                            data=neuron_df,
                            groups=neuron_df['recording'],
                        )
                        fit = model.fit(reml=True, method='lbfgs')

                        coef = float(fit.params.get('genotype_code', 0))
                        se = float(fit.bse.get('genotype_code', 0))
                        p = float(fit.pvalues.get('genotype_code', 1))
                        ci = fit.conf_int()
                        ci_low = float(ci.loc['genotype_code', 0]) if 'genotype_code' in ci.index else coef - 1.96*se
                        ci_high = float(ci.loc['genotype_code', 1]) if 'genotype_code' in ci.index else coef + 1.96*se

                        mr['neuron_lmm'] = {
                            'coef': coef, 'se': se, 'p': p,
                            'ci_low': ci_low, 'ci_high': ci_high,
                            'n_obs': len(neuron_df), 'n_groups': n_recs_neuron,
                            'formula': f'{metric} ~ genotype + (1|recording)',
                        }
                        logger.info(f"  {metric} neuron-LMM: coef={coef:.4f}, p={p:.4f}")
                    else:
                        mr['neuron_lmm'] = {'skipped': f'need ≥5 recordings, have {n_recs_neuron}'}
                except Exception as e:
                    mr['neuron_lmm'] = {'error': str(e)}
                    logger.warning(f"  {metric} neuron-LMM failed: {e}")

            mixed_results[metric] = mr

    # ── Permutation test (always runs, model-free) ───────────────────────
    n_perms = 5000
    rng = np.random.default_rng(42)

    for metric, label in metrics_recording:
        if metric not in mixed_results:
            mixed_results[metric] = {'metric': metric, 'label': label}

        ctrl_vals = np.array([r[metric] for r in ds_records if r['genotype'] == 'Control' and r[metric] is not None and np.isfinite(r[metric])])
        mut_vals = np.array([r[metric] for r in ds_records if r['genotype'] == 'Mutant' and r[metric] is not None and np.isfinite(r[metric])])

        if len(ctrl_vals) < 2 or len(mut_vals) < 2:
            mixed_results[metric]['permutation'] = {'skipped': True}
            continue

        observed_diff = float(np.mean(mut_vals) - np.mean(ctrl_vals))
        all_vals = np.concatenate([ctrl_vals, mut_vals])
        nc = len(ctrl_vals)

        null_diffs = np.zeros(n_perms)
        for p_idx in range(n_perms):
            perm = rng.permutation(all_vals)
            null_diffs[p_idx] = np.mean(perm[nc:]) - np.mean(perm[:nc])

        p_perm = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))
        pooled_sd = float(np.std(all_vals, ddof=1))
        d_perm = observed_diff / pooled_sd if pooled_sd > 0 else 0.0

        mixed_results[metric]['permutation'] = {
            'observed_diff': observed_diff,
            'p': p_perm,
            'cohens_d': d_perm,
            'n_ctrl': len(ctrl_vals),
            'n_mut': len(mut_vals),
            'ctrl_mean': float(np.mean(ctrl_vals)),
            'ctrl_sd': float(np.std(ctrl_vals, ddof=1)),
            'mut_mean': float(np.mean(mut_vals)),
            'mut_sd': float(np.std(mut_vals, ddof=1)),
        }
        logger.info(f"  {metric} perm: diff={observed_diff:.4f}, p={p_perm:.4f}, d={d_perm:.3f}")

    results['mixed_model'] = mixed_results

    # ── Generate figures ─────────────────────────────────────────────────
    _fig_mixed_model_panel(
        mixed_results, metrics_recording, ds_records, geno_dir, results)


def _fig_mixed_model_panel(
    mixed_results: dict,
    metrics: list,
    ds_records: list,
    geno_dir: str,
    results: dict,
) -> None:
    """
    Two-panel figure per metric:
      Left:  recording-level dot plot (one dot per recording)
      Right: coefficient comparison across methods (raw units, not normalised)
    Plus a summary table.
    """


    rng = np.random.default_rng(42)

    n_metrics = len(metrics)

    # ── Main figure: paired dot-plot + method comparison ─────────────────
    fig, axes = plt.subplots(n_metrics, 2, figsize=(14, 4 * n_metrics),
                             gridspec_kw={'width_ratios': [1.2, 1]})
    fig.patch.set_facecolor('white')
    if n_metrics == 1:
        axes = axes.reshape(1, 2)

    for row, (metric, label) in enumerate(metrics):
        mr = mixed_results.get(metric, {})
        perm = mr.get('permutation', {})

        ctrl = np.array([r[metric] for r in ds_records if r['genotype'] == 'Control' and r[metric] is not None and np.isfinite(r[metric])])
        mut = np.array([r[metric] for r in ds_records if r['genotype'] == 'Mutant' and r[metric] is not None and np.isfinite(r[metric])])

        # ── Left panel: recording-level dot plot ─────────────────────────
        ax_dot = axes[row, 0]
        ax_dot.set_facecolor('white')

        # Box plots (transparent)
        bp = ax_dot.boxplot(
            [ctrl, mut], positions=[0, 1], widths=0.4,
            patch_artist=True, showfliers=False,
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(color='#888'),
            capprops=dict(color='#888'))
        bp['boxes'][0].set(facecolor=CTRL_COLOR, alpha=0.2)
        bp['boxes'][1].set(facecolor=MUT_COLOR, alpha=0.2)

        # Individual dots (one per recording)
        jc = rng.uniform(-0.1, 0.1, len(ctrl))
        jm = rng.uniform(-0.1, 0.1, len(mut))
        ax_dot.scatter(jc, ctrl, c=CTRL_COLOR, s=35, alpha=0.7,
                       edgecolors='white', linewidths=0.5, zorder=5)
        ax_dot.scatter(1 + jm, mut, c=MUT_COLOR, s=35, alpha=0.7,
                       edgecolors='white', linewidths=0.5, zorder=5)

        # Means with error bars
        ax_dot.errorbar(0, np.mean(ctrl), yerr=np.std(ctrl, ddof=1)/np.sqrt(len(ctrl)),
                        fmt='D', color=CTRL_COLOR, markersize=8, zorder=10,
                        capsize=4, linewidth=2, markeredgecolor='white')
        ax_dot.errorbar(1, np.mean(mut), yerr=np.std(mut, ddof=1)/np.sqrt(len(mut)),
                        fmt='D', color=MUT_COLOR, markersize=8, zorder=10,
                        capsize=4, linewidth=2, markeredgecolor='white')

        ax_dot.set_xticks([0, 1])
        ax_dot.set_xticklabels([f'Control\n(n={len(ctrl)})',
                                f'Mutant\n(n={len(mut)})'], fontsize=9)
        ax_dot.set_ylabel(label, fontsize=9)
        ax_dot.spines['top'].set_visible(False)
        ax_dot.spines['right'].set_visible(False)
        ax_dot.grid(axis='y', alpha=0.15)

        # Significance bracket
        p_best = perm.get('p', 1.0)
        rec_lmm = mr.get('recording_lmm', {})
        if isinstance(rec_lmm.get('p'), float) and not np.isnan(rec_lmm['p']):
            p_best = rec_lmm['p']
        stars = '***' if p_best < 0.001 else '**' if p_best < 0.01 else '*' if p_best < 0.05 else 'ns'
        ctrl_fin = ctrl[np.isfinite(ctrl)]
        mut_fin  = mut[np.isfinite(mut)]
        if len(ctrl_fin) == 0 or len(mut_fin) == 0:
            ax_dot.set_ylim(top=1.0)
        else:
            ymax = max(float(np.max(ctrl_fin)), float(np.max(mut_fin)))
            bracket_y = ymax * 1.05
            ax_dot.plot([0, 0, 1, 1], [bracket_y * 0.98, bracket_y, bracket_y, bracket_y * 0.98],
                        color='#333', linewidth=1)
            ax_dot.text(0.5, bracket_y * 1.02, f'{stars} p={p_best:.3f}',
                        ha='center', fontsize=9, fontweight='bold' if p_best < 0.05 else 'normal')
            ax_dot.set_ylim(top=bracket_y * 1.12)

        if row == 0:
            ax_dot.set_title('Recording-Level Comparison\n(each dot = one recording)',
                             fontsize=10, fontweight='bold')

        # ── Right panel: method comparison (coefficients in raw units) ───
        ax_meth = axes[row, 1]
        ax_meth.set_facecolor('white')

        methods = []
        # 1. Permutation
        if perm and not perm.get('skipped'):
            methods.append({
                'name': 'Permutation\n(recording means)',
                'coef': perm['observed_diff'],
                'ci_low': None, 'ci_high': None,
                'p': perm['p'],
                'color': '#2E86AB',
            })

        # 2. Recording-level LMM
        rec_lmm = mr.get('recording_lmm', {})
        if isinstance(rec_lmm.get('coef'), (int, float)):
            methods.append({
                'name': f'LMM recording\n(1|day, n={rec_lmm.get("n_groups","")})',
                'coef': rec_lmm['coef'],
                'ci_low': rec_lmm.get('ci_low'),
                'ci_high': rec_lmm.get('ci_high'),
                'p': rec_lmm.get('p', 1),
                'color': '#A23B72',
            })

        # 3. Neuron-level LMM
        neu_lmm = mr.get('neuron_lmm', {})
        if isinstance(neu_lmm.get('coef'), (int, float)):
            methods.append({
                'name': f'LMM neuron\n(1|recording)',
                'coef': neu_lmm['coef'],
                'ci_low': neu_lmm.get('ci_low'),
                'ci_high': neu_lmm.get('ci_high'),
                'p': neu_lmm.get('p', 1),
                'color': '#F18F01',
            })

        y_positions = list(range(len(methods)))
        for y, m in zip(y_positions, methods):
            coef = m['coef']
            p_val = m['p']
            color = m['color']

            ax_meth.plot(coef, y, 'o', color=color, markersize=9, zorder=5)

            if m['ci_low'] is not None and m['ci_high'] is not None:
                ci_l, ci_h = m['ci_low'], m['ci_high']
                # Clip extreme CIs for readability
                plot_range = max(abs(coef) * 5, 0.01)
                ci_l_clip = max(ci_l, -plot_range)
                ci_h_clip = min(ci_h, plot_range)
                ax_meth.plot([ci_l_clip, ci_h_clip], [y, y], '-',
                             color=color, linewidth=2.5, alpha=0.6, zorder=4)

            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            p_str = f'p<.001' if p_val < 0.001 else f'p={p_val:.3f}'
            ax_meth.text(coef, y + 0.25, f'{stars} {p_str}',
                         ha='center', fontsize=8, color=color,
                         fontweight='bold' if p_val < 0.05 else 'normal')

        ax_meth.axvline(0, color='#999', linewidth=1, linestyle='--', zorder=1)
        ax_meth.set_yticks(y_positions)
        ax_meth.set_yticklabels([m['name'] for m in methods], fontsize=8)
        ax_meth.set_xlabel(f'Genotype effect ({label.split("(")[0].strip()} units)',
                           fontsize=8)
        ax_meth.spines['top'].set_visible(False)
        ax_meth.spines['right'].set_visible(False)
        ax_meth.grid(axis='x', alpha=0.15)
        ax_meth.invert_yaxis()

        if row == 0:
            ax_meth.set_title('Effect Estimates by Method\n(Mutant − Control, raw units)',
                              fontsize=10, fontweight='bold')

    fig.suptitle('Genotype Effect — Multi-Method Analysis',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.text(0.5, -0.01,
             'Left: each dot is one recording (avoids pseudoreplication)  |  '
             'Right: coefficient = Mutant − Control difference  |  '
             'LMM = linear mixed model controlling for session effects',
             ha='center', fontsize=7, color='#777', style='italic')
    plt.tight_layout()
    path = os.path.join(geno_dir, 'genotype_mixed_model_forest.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Mixed model panel figure saved: {path}")

    # ── Summary table figure ─────────────────────────────────────────────
    fig_t = plt.figure(figsize=(16, max(4, len(metrics) * 1.2 + 2)))
    fig_t.patch.set_facecolor('white')
    ax_t = fig_t.add_subplot(111)
    ax_t.axis('off')

    headers = ['Metric',
               'Control\nmean ± SD', 'Mutant\nmean ± SD',
               'Permutation\np (d)',
               'LMM recording\np (coef)',
               'LMM neuron\np (coef)']

    table_data = []
    for metric, label in metrics:
        mr = mixed_results.get(metric, {})
        perm = mr.get('permutation', {})
        rec = mr.get('recording_lmm', {})
        neu = mr.get('neuron_lmm', {})

        ctrl_mean = perm.get('ctrl_mean', 0)
        ctrl_sd = perm.get('ctrl_sd', 0)
        mut_mean = perm.get('mut_mean', 0)
        mut_sd = perm.get('mut_sd', 0)

        row = [
            label,
            f'{ctrl_mean:.3f} ± {ctrl_sd:.3f}',
            f'{mut_mean:.3f} ± {mut_sd:.3f}',
        ]

        # Permutation
        if perm and not perm.get('skipped'):
            row.append(f'p={perm["p"]:.3f} (d={perm["cohens_d"]:.2f})')
        else:
            row.append('—')

        # Recording LMM
        if isinstance(rec.get('p'), float) and not np.isnan(rec['p']):
            row.append(f'p={rec["p"]:.3f} ({rec["coef"]:.4f})')
        elif rec.get('error'):
            row.append(f'failed')
        elif rec.get('skipped'):
            row.append(str(rec['skipped'])[:20])
        else:
            row.append('—')

        # Neuron LMM
        if isinstance(neu.get('p'), float) and not np.isnan(neu['p']):
            row.append(f'p={neu["p"]:.3f} ({neu["coef"]:.4f})')
        elif neu.get('error'):
            row.append(f'failed')
        else:
            row.append('—')

        table_data.append(row)

    table = ax_t.table(cellText=table_data, colLabels=headers,
                       cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    for col in range(len(headers)):
        cell = table[0, col]
        cell.set_facecolor('#E6F1FB')
        cell.set_text_props(color='white', fontweight='bold', fontsize=7)

    for row_idx, row in enumerate(table_data, start=1):
        for col_idx in range(len(row)):
            cell = table[row_idx, col_idx]
            cell.set_edgecolor('#DDD')
            cell.set_facecolor('#F8F8F8' if row_idx % 2 == 0 else 'white')
            # Highlight significant p-values
            try:
                text = row[col_idx]
                if 'p=' in text:
                    p_str = text.split('p=')[1].split(' ')[0].split(')')[0]
                    if float(p_str) < 0.05:
                        cell.set_facecolor('#D4EDDA')
                        cell.set_text_props(fontweight='bold')
            except (ValueError, IndexError):
                pass

    fig_t.suptitle('Genotype Effect — Statistical Summary',
                   fontsize=13, fontweight='bold', y=0.98)
    fig_t.text(0.5, 0.02,
               'Permutation: shuffles genotype labels across recording means (5000 iterations)  |  '
               'LMM recording: genotype fixed + (1|organoid_day)  |  '
               'LMM neuron: genotype fixed + (1|recording)',
               ha='center', fontsize=7, color='#777', style='italic')

    tpath = os.path.join(geno_dir, 'genotype_methods_comparison.png')
    plt.savefig(tpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Methods comparison table saved: {tpath}")



# =============================================================================
# BETWEEN-ORGANOID TESTS
# =============================================================================

def run_between_organoid_tests(datasets: List[DatasetMetrics], output_dir: str) -> dict:
    """
    Statistical comparison BETWEEN organoids.

    Groups recordings by organoid ID, pools neurons across recordings
    within each organoid, then compares between organoids.

    Per-neuron metrics (spike rate):
        - Kruskal-Wallis across organoids, pairwise Mann-Whitney U

    Per-recording metrics (correlation, synchrony):
        - Each organoid has multiple recordings → can run Mann-Whitney
          between organoids on these distributions

    Only runs if there are 2+ distinct organoids. Returns empty dict
    and skips figure generation if only one organoid is present.
    """

    results = {'level': 'between-organoid', 'tests': {}}

    # ── Group datasets by organoid ───────────────────────────────────────
    organoid_map = OrderedDict()  # organoid_id → list of DatasetMetrics
    for ds in datasets:
        org_id = _extract_organoid_id(ds.name)
        if org_id not in organoid_map:
            organoid_map[org_id] = []
        organoid_map[org_id].append(ds)

    org_ids = list(organoid_map.keys())
    # Sort by numeric day value (e.g. D109 → 109) so age groups are ascending
    def _day_sort_key(oid):
        import re
        m = re.search(r'\d+', oid)
        return int(m.group()) if m else 0
    org_ids.sort(key=_day_sort_key)
    # Rebuild map in sorted order
    organoid_map = OrderedDict((oid, organoid_map[oid]) for oid in org_ids)
    n_org = len(org_ids)
    results['n_organoids'] = n_org
    results['organoids'] = {
        oid: [ds.name for ds in dsets] for oid, dsets in organoid_map.items()
    }

    if n_org < 2:
        logger.info(f"Between-organoid tests: only {n_org} organoid(s) found, skipping.")
        results['skipped'] = True
        results['reason'] = f'Only {n_org} organoid(s) — need at least 2 for comparison.'
        return results

    logger.info(f"Between-organoid comparison: {n_org} organoids "
                f"({', '.join(f'{oid} ({len(dsets)} recs)' for oid, dsets in organoid_map.items())})")

    # ── Pool per-neuron spike rates and transient amplitudes by organoid ─────
    # Descriptive view: shows full distribution of individual neuron activity
    # across organoid days. Not used for statistical testing.
    org_rates = OrderedDict()
    org_amplitudes = OrderedDict()
    for oid, dsets in organoid_map.items():
        pooled_rates = []
        pooled_amps = []
        for ds in dsets:
            rates = _get_neuron_rates(ds)
            amps = _get_neuron_amplitudes(ds)
            pooled_rates.extend(rates)
            pooled_amps.extend(amps)
        
        # Only include organoids that have at least one active neuron
        if len(pooled_rates) > 0:
            org_rates[oid] = np.array(pooled_rates)
        if len(pooled_amps) > 0:
            org_amplitudes[oid] = np.array(pooled_amps)

    # Update org_ids to only include organoids with data
    org_ids = [oid for oid in organoid_map.keys() if oid in org_rates]
    n_org = len(org_ids)
    
    if n_org < 2:
        logger.warning(f"Only {n_org} organoids with active neurons — skipping between-organoid tests")
        return results

    # ── Pool per-recording metrics by organoid (active organoids only) ──
    org_corr = OrderedDict()
    org_sync = OrderedDict()
    for oid in org_ids:
        dsets = organoid_map[oid]
        _oc = np.array([ds.pairwise_correlation_mean for ds in dsets])
        _os = np.array([ds.synchrony_index for ds in dsets])
        org_corr[oid] = _oc[np.isfinite(_oc)]
        org_sync[oid] = _os[np.isfinite(_os)]

    # ── Statistical tests ────────────────────────────────────────────────
    def _run_tests(data_dict, metric_name, level):
        """Run KW + pairwise MW on a dict of {group: array}."""
        ids = list(data_dict.keys())
        arrays = [data_dict[k] for k in ids]
        test_result = {
            'metric': metric_name, 'level': level,
            'per_organoid': {},
        }

        for oid, arr in zip(ids, arrays):
            test_result['per_organoid'][oid] = {
                'n': len(arr),
                'mean': float(np.mean(arr)) if len(arr) > 0 else 0,
                'median': float(np.median(arr)) if len(arr) > 0 else 0,
                'sd': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0,
            }

        # Kruskal-Wallis (store in JSON but don't display prominently)
        valid = [a for a in arrays if len(a) >= 2]
        if len(valid) >= 2:
            H, p_kw = sp_stats.kruskal(*valid)
            test_result['kruskal_wallis'] = {'H': float(H), 'p': float(p_kw)}

        # Pairwise Mann-Whitney
        n_pairs = len(ids) * (len(ids) - 1) // 2
        pairwise = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if len(arrays[i]) < 2 or len(arrays[j]) < 2:
                    continue
                U, p_mw = sp_stats.mannwhitneyu(
                    arrays[i], arrays[j], alternative='two-sided')
                p_corr = min(p_mw * max(n_pairs, 1), 1.0)
                pairwise.append({
                    'i': i, 'j': j,
                    'a': ids[i], 'b': ids[j],
                    'U': float(U), 'p_raw': float(p_mw), 'p_bonf': float(p_corr),
                })
        test_result['pairwise'] = pairwise
        return test_result

    rate_tests = _run_tests(org_rates, 'Event rate (events/10s)', 'per-neuron, pooled')
    amp_tests = _run_tests(org_amplitudes, 'Transient amplitude (ΔF/F₀)', 'per-neuron, pooled')
    corr_tests = _run_tests(org_corr, 'Pairwise Correlation (r)', 'per-recording')
    sync_tests = _run_tests(org_sync, 'Synchrony Index', 'per-recording')

    results['tests']['spike_rate'] = rate_tests
    results['tests']['spike_amplitude'] = amp_tests
    results['tests']['pairwise_correlation'] = corr_tests
    results['tests']['synchrony_index'] = sync_tests

    # =====================================================================
    # FIGURE: Between-organoid comparison (spike rate only)
    # =====================================================================
    BG      = 'white'
    TEXT    = '#333333'
    GRID    = '#DDDDDD'
    MEDIAN  = '#E53935'

    fig_width = max(14, n_org * 1.8)
    fig = plt.figure(figsize=(fig_width, 9))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(1, 1, left=0.08, right=0.97, top=0.92, bottom=0.10)

    # Colour palette — slightly brighter for dark background
    palette = ['#5B8FD4', '#F0923B', '#7DC460', '#FFD04A', '#74B8E8',
               '#B0B0B0', '#4A7ABF', '#C04050']
    org_colors = {oid: palette[i % len(palette)] for i, oid in enumerate(org_ids)}

    rng = np.random.default_rng(42)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG)

    rate_arrays = [org_rates[oid] for oid in org_ids]
    bp = ax.boxplot(
        rate_arrays, positions=range(n_org), widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=MEDIAN, linewidth=2.0),
        whiskerprops=dict(color=TEXT, linewidth=0.8, alpha=0.5),
        capprops=dict(color=TEXT, linewidth=0.8, alpha=0.5),
    )
    for i, patch in enumerate(bp['boxes']):
        col = org_colors[org_ids[i]]
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.2)

    # Compute 3-SD outlier threshold across all organoids (used for y-axis limits
    # and scatter display only — boxplot statistics use the full unclipped data)
    all_rates_flat = np.concatenate([org_rates[oid] for oid in org_ids if len(org_rates[oid]) > 0])
    _global_mean = float(np.mean(all_rates_flat)) if len(all_rates_flat) > 1 else 0.0
    _global_sd   = float(np.std(all_rates_flat, ddof=1)) if len(all_rates_flat) > 1 else 1.0
    _y_clip      = _global_mean + 3.0 * _global_sd  # upper clip boundary

    n_zero_total     = int(np.sum(all_rates_flat == 0))
    n_excluded_total = int(np.sum(all_rates_flat > _y_clip))

    # Overlay individual neuron dots — zeros and >3 SD outliers excluded from
    # display (inactive neurons add no trend information; outliers compress scale)
    all_rates_display = all_rates_flat[(all_rates_flat > 0) & (all_rates_flat <= _y_clip)]
    y_range = np.ptp(all_rates_display) if len(all_rates_display) > 1 else 1.0
    y_jitter_scale = y_range * 0.015

    for i, (oid, rates) in enumerate(org_rates.items()):
        mask = (rates > 0) & (rates <= _y_clip)
        rates_plot = rates[mask]
        n_pts = len(rates_plot)
        if n_pts == 0:
            continue
        if n_pts == 1:
            x_pts = np.array([i])
            y_pts = rates_plot
        else:
            base = np.linspace(-0.22, 0.22, n_pts)
            rng.shuffle(base)
            noise = rng.uniform(-0.02, 0.02, n_pts)
            x_pts = i + base + noise
            y_pts = rates_plot + rng.uniform(-y_jitter_scale, y_jitter_scale, n_pts)
        ax.scatter(x_pts, y_pts, color=org_colors[oid], s=10,
                   alpha=0.45, zorder=5, linewidths=0, edgecolors='none')

    ax.set_xticks(range(n_org))
    ax.set_xticklabels(org_ids, fontsize=11, fontweight='bold',
                       color=TEXT,
                       rotation=45 if n_org > 6 else 0,
                       ha='right' if n_org > 6 else 'center')
    ax.set_ylabel('Event rate (events/10s)', fontsize=12, color=TEXT)
    # Linear y-axis; clip view to 3 SD above global mean, with a small top margin
    ax.set_ylim(bottom=0, top=_y_clip * 1.08)
    ax.tick_params(colors=TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.6, color=GRID)

    _excl_parts = []
    if n_zero_total > 0:
        _excl_parts.append(f'{n_zero_total} inactive (zero-rate) neuron(s) not shown')
    if n_excluded_total > 0:
        _excl_parts.append(f'{n_excluded_total} point(s) >3 SD above mean not shown')
    _excl_note = ('  ·  ' + '  ·  '.join(_excl_parts)) if _excl_parts else ''
    fig.text(0.5, 0.02,
             f'Descriptive overview: Active neurons only (zero-rate excluded){_excl_note}',
             ha='center', fontsize=8, color='#5A7A8A', style='italic')
    fig.suptitle('Pooled event rates by age',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.97)

    organoid_dir = os.path.join(output_dir, 'figures', '1 - Main Results')
    os.makedirs(organoid_dir, exist_ok=True)
    path = os.path.join(organoid_dir, 'between_organoid_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =====================================================================
    # FIGURE: Transient amplitude by organoid (same style as panel A)
    # =====================================================================
    fig_amp, ax_amp = plt.subplots(figsize=(max(8, n_org * 1.2), 6))
    fig_amp.patch.set_facecolor('white')
    ax_amp.set_facecolor('white')

    amp_arrays = [org_amplitudes[oid] for oid in org_ids]
    bp_amp = ax_amp.boxplot(
        amp_arrays, positions=range(n_org), widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color='#CC3333', linewidth=1.5),
        whiskerprops=dict(color='#555555', linewidth=0.8),
        capprops=dict(color='#555555', linewidth=0.8),
    )
    for i, patch in enumerate(bp_amp['boxes']):
        col = org_colors[org_ids[i]]
        patch.set_facecolor(col)
        patch.set_alpha(0.25)
        patch.set_edgecolor(col)
        patch.set_linewidth(0.8)

    # Overlay individual neurons
    for i, (oid, amps) in enumerate(org_amplitudes.items()):
        if len(amps) > 0:
            jitter = rng.uniform(-0.18, 0.18, len(amps))
            ax_amp.scatter(i + jitter, amps, color=org_colors[oid], s=8,
                          alpha=0.3, zorder=5, linewidths=0, edgecolors='none')

    ax_amp.set_xticks(range(n_org))
    ax_amp.set_xticklabels(org_ids, fontsize=10, fontweight='bold',
                           rotation=45 if n_org > 6 else 0, 
                           ha='right' if n_org > 6 else 'center')
    # Color x-tick labels
    for i, (tick_label, oid) in enumerate(zip(ax_amp.get_xticklabels(), org_ids)):
        tick_label.set_color(org_colors[oid])
    
    ax_amp.set_ylabel('Transient amplitude (ΔF/F₀)', fontsize=11)
    ax_amp.set_xlabel('Organoid', fontsize=10)
    ax_amp.grid(axis='y', alpha=0.15)
    ax_amp.spines['top'].set_visible(False)
    ax_amp.spines['right'].set_visible(False)

    ax_amp.set_title('Transient amplitude by organoid', fontsize=13, fontweight='bold')
    
    fig_amp.text(0.5, 0.01,
                 'Descriptive overview — per-neuron spike amplitudes  ·  Each dot = one neuron',
                 ha='center', fontsize=7, color='#777777', style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    amp_path = os.path.join(organoid_dir, 'spike_amplitude_by_organoid.png')
    plt.savefig(amp_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Spike amplitude figure saved: {amp_path}")

    # =====================================================================
    # TABLE: Between-organoid pairwise p-values
    # =====================================================================
    # Scale figure height with number of organoids
    table_height = max(12, 10 + n_org * 0.8)
    fig2 = plt.figure(figsize=(16, table_height))
    fig2.patch.set_facecolor('white')

    # Use fig.text() with carefully calculated y-coordinates for proper spacing
    # Calculate vertical spacing based on number of organoids
    summary_table_height = 0.08 + n_org * 0.018  # Height needed for summary table
    pvalue_table_height = 0.06 + n_org * 0.022   # Height for each p-value matrix
    
    # Position from top to bottom with explicit spacing
    y_title = 0.97
    y_summary_title = 0.93
    y_summary_table_top = 0.90
    summary_table_bottom = y_summary_table_top - summary_table_height
    
    # Calculate spacing for the three p-value tables
    remaining_space = summary_table_bottom - 0.04  # Leave margin at bottom
    table_spacing = 0.04  # Gap between tables
    single_table_block = (remaining_space - 2 * table_spacing) / 3  # Space for each table + title

    fig2.text(0.5, y_title, 'Between-Organoid Statistical Tests',
              ha='center', fontsize=14, fontweight='bold')

    # --- Table A: Descriptive stats per organoid ---
    ax_t1 = fig2.add_axes([0.05, summary_table_bottom, 0.90, summary_table_height])
    ax_t1.axis('off')

    fig2.text(0.5, y_summary_title, 'Per-Organoid Summary',
              ha='center', fontsize=11, fontweight='bold')

    headers1 = ['Organoid', 'Recordings', 'Neurons', 'Rate mean±SD',
                'Corr mean±SD', 'Sync mean±SD']
    rows1 = []
    for oid in org_ids:
        rates = org_rates[oid]
        corrs = org_corr[oid]
        syncs = org_sync[oid]
        rows1.append([
            oid, str(len(organoid_map[oid])), str(len(rates)),
            f'{np.mean(rates):.3f} ± {np.std(rates, ddof=1):.3f}' if len(rates) > 1 else (f'{np.mean(rates):.3f}' if len(rates) == 1 else '—'),
            f'{np.mean(corrs):.4f} ± {np.std(corrs, ddof=1):.4f}' if len(corrs) > 1 else (f'{np.mean(corrs):.4f}' if len(corrs) == 1 else '—'),
            f'{np.mean(syncs):.4f} ± {np.std(syncs, ddof=1):.4f}' if len(syncs) > 1 else (f'{np.mean(syncs):.4f}' if len(syncs) == 1 else '—'),
        ])

    t1 = ax_t1.table(cellText=rows1, colLabels=headers1,
                      loc='center', cellLoc='center')
    t1.auto_set_font_size(False)
    t1.set_fontsize(8)
    t1.scale(1, 1.5)
    for (row, col), cell in t1.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E6F1FB')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            i = row - 1
            cell.set_facecolor('#F2F7FC' if row % 2 == 0 else 'white')
        cell.set_edgecolor('#DDDDDD')

    # --- Tables B/C/D: Pairwise p-values for each metric ---
    metrics_for_table = [
        ('spike_rate', 'Spike Rate — Pairwise p-values (Mann-Whitney U, Bonferroni)', rate_tests),
        ('pairwise_correlation', 'Pairwise Correlation — Pairwise p-values', corr_tests),
        ('synchrony_index', 'Synchrony Index — Pairwise p-values', sync_tests),
    ]

    # Calculate positions for three p-value tables, evenly spaced below summary
    # Each table block = title + table content
    title_offset = 0.03  # Space between title and table
    for idx_m, (mkey, mtitle, mtest) in enumerate(metrics_for_table):
        # Calculate this table's vertical position
        block_top = summary_table_bottom - table_spacing - idx_m * (single_table_block + table_spacing)
        table_title_y = block_top
        table_bottom = block_top - title_offset - pvalue_table_height
        
        ax_m = fig2.add_axes([0.10, table_bottom, 0.80, pvalue_table_height])
        ax_m.axis('off')
        fig2.text(0.5, table_title_y, mtitle,
                  ha='center', fontsize=10, fontweight='bold')

        # Build pairwise matrix
        p_mat = np.ones((n_org, n_org))
        for pair in mtest.get('pairwise', []):
            p_mat[pair['i'], pair['j']] = pair['p_bonf']
            p_mat[pair['j'], pair['i']] = pair['p_bonf']

        col_h = [''] + org_ids
        rows_m = []
        for i in range(n_org):
            row_data = [org_ids[i]]
            for j in range(n_org):
                if i == j:
                    row_data.append('—')
                else:
                    p = p_mat[i, j]
                    row_data.append(f'{_fmt_p(p)} {_sig_stars(p)}')
            rows_m.append(row_data)

        tm = ax_m.table(cellText=rows_m, colLabels=col_h,
                         loc='center', cellLoc='center')
        tm.auto_set_font_size(False)
        tm.set_fontsize(8)
        tm.scale(1, 1.5)

        for (row, col), cell in tm.get_celld().items():
            if row == 0:
                cell.set_facecolor('#E6F1FB')
                cell.set_text_props(color='white', fontweight='bold')
            elif col == 0 and row > 0:
                cell.set_facecolor('#E8EFF7')
                cell.set_text_props(fontweight='bold')
            else:
                r, c = row - 1, col - 1
                if r == c:
                    cell.set_facecolor('#E0E0E0')
                else:
                    p = p_mat[r, c] if 0 <= r < n_org and 0 <= c < n_org else 1.0
                    if p < 0.001:
                        cell.set_facecolor('#C6EFCE')
                        cell.set_text_props(color='#006100', fontweight='bold')
                    elif p < 0.01:
                        cell.set_facecolor('#D9F2D9')
                        cell.set_text_props(color='#006100')
                    elif p < 0.05:
                        cell.set_facecolor('#FFEB9C')
                        cell.set_text_props(color='#9C6500')
                    else:
                        cell.set_facecolor('white')
                        cell.set_text_props(color='#999999')
            cell.set_edgecolor('#CCCCCC')

    table_path = os.path.join(organoid_dir, 'between_organoid_table.png')
    plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Between-organoid figures saved: {path}, {table_path}")
    return results




# =============================================================================
# ACTIVITY ANALYSIS
# =============================================================================

def run_activity_analysis(datasets: List[DatasetMetrics], output_dir: str,
                          mutant_label: str = 'CEP41 R242H') -> dict:
    """
    Core activity analysis: frequency, amplitude, and active fraction.

    Analyses activity across three dimensions:
    1. Genotype comparison (Control vs Mutant)
    2. Longitudinal (across organoid day/age)
    3. Combined (genotype × age)

    Only ACTIVE neurons (≥1 validated transient) contribute to frequency
    and amplitude metrics.  Active fraction is reported separately.

    Figures saved to: figures/3 - Activity Analysis/
    """

    results = {'level': 'activity_analysis'}

    activity_dir = os.path.join(output_dir, 'figures', '3 - Activity Analysis')
    os.makedirs(activity_dir, exist_ok=True)

    # ── Parse metadata ───────────────────────────────────────────────────
    genotypes = [_extract_genotype(ds.name) for ds in datasets]
    organoid_ids = [_extract_organoid_id(ds.name) for ds in datasets]

    # Sort organoid days by numeric age
    def _day_num(oid):
        m = re.search(r'\d+', oid)
        return int(m.group()) if m else 0

    unique_days = sorted(set(organoid_ids), key=_day_num)

    CTRL_COLOR = '#4472C4'
    MUT_COLOR = '#ED7D31'
    GENO_COLORS = {'Control': CTRL_COLOR, 'Mutant': MUT_COLOR, 'Unknown': '#999999'}

    n_ctrl = sum(1 for g in genotypes if g == 'Control')
    n_mut = sum(1 for g in genotypes if g == 'Mutant')
    logger.info(f"Activity analysis: {n_ctrl} Control, {n_mut} Mutant, "
                f"{len(unique_days)} days ({', '.join(unique_days)})")

    # ── Extract per-dataset metrics ──────────────────────────────────────
    ds_data = []
    for ds, geno, day in zip(datasets, genotypes, organoid_ids):
        if geno == 'Unknown':
            continue

        # Active neurons only for frequency/amplitude
        active_mask = ds.neuron_is_active if ds.neuron_is_active is not None else np.zeros(0, dtype=bool)
        active_rates = ds.neuron_spike_rates[active_mask] if ds.neuron_spike_rates is not None and active_mask.any() else np.array([])
        active_amps = ds.neuron_spike_amplitudes[active_mask] if ds.neuron_spike_amplitudes is not None and active_mask.any() else np.array([])
        # Filter out zero amplitudes from active neurons (neurons where amplitude measurement failed)
        active_amps = active_amps[active_amps > 0] if len(active_amps) > 0 else active_amps

        ds_data.append({
            'name': ds.name,
            'genotype': geno,
            'day': day,
            'day_num': _day_num(day),
            'n_selected': ds.n_selected,
            'n_active': ds.n_active,
            'active_fraction': ds.active_fraction,
            'active_rates': active_rates,
            'active_amps': active_amps,
            'mean_rate': float(np.mean(active_rates)) if len(active_rates) > 0 else 0.0,
            'mean_amp': float(np.mean(active_amps)) if len(active_amps) > 0 else 0.0,
        })

    if not ds_data:
        logger.warning("No datasets with genotype info for activity analysis")
        return results

    # ── Helper: Mann-Whitney with effect size ────────────────────────────
    def _mw_test(a, b, label=''):
        if len(a) < 2 or len(b) < 2:
            return {'label': label, 'skipped': True, 'n_a': len(a), 'n_b': len(b)}
        U, p = sp_stats.mannwhitneyu(a, b, alternative='two-sided')
        pooled_std = np.sqrt(((len(a)-1)*np.var(a, ddof=1) +
                              (len(b)-1)*np.var(b, ddof=1)) /
                             (len(a) + len(b) - 2))
        d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 1e-10 else 0.0
        return {'label': label, 'U': float(U), 'p': float(p), 'cohens_d': float(d),
                'n_a': len(a), 'n_b': len(b),
                'mean_a': float(np.mean(a)), 'mean_b': float(np.mean(b))}

    # =====================================================================
    # FIGURE 1: Genotype comparison (3 panels: frequency, amplitude, active fraction)
    # =====================================================================
    ctrl_data = [d for d in ds_data if d['genotype'] == 'Control']
    mut_data = [d for d in ds_data if d['genotype'] == 'Mutant']

    # Per-recording means for rate and amplitude (consistent with genotype comparison)
    ctrl_rates = np.array([d['mean_rate'] for d in ctrl_data if d['mean_rate'] > 0])
    mut_rates = np.array([d['mean_rate'] for d in mut_data if d['mean_rate'] > 0])
    ctrl_amps = np.array([d['mean_amp'] for d in ctrl_data if d['mean_amp'] > 0])
    mut_amps = np.array([d['mean_amp'] for d in mut_data if d['mean_amp'] > 0])

    # Per-recording active fractions
    ctrl_af = np.array([d['active_fraction'] for d in ctrl_data])
    mut_af = np.array([d['active_fraction'] for d in mut_data])

    # Statistical tests (per-recording)
    test_rate = _mw_test(ctrl_rates, mut_rates, 'Event Frequency')
    test_amp = _mw_test(ctrl_amps, mut_amps, 'Transient Amplitude')
    test_af = _mw_test(ctrl_af, mut_af, 'Active Fraction')

    results['genotype_tests'] = {
        'spike_frequency': test_rate,
        'spike_amplitude': test_amp,
        'active_fraction': test_af,
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('white')

    rng = np.random.default_rng(42)

    panels = [
        ('Event frequency\n(events/10s, per recording)', ctrl_rates, mut_rates, test_rate),
        ('Transient amplitude\n(ΔF/F₀, per recording)', ctrl_amps, mut_amps, test_amp),
        ('Active Fraction\n(per recording)', ctrl_af, mut_af, test_af),
    ]

    for col, (title, ctrl_vals, mut_vals, test) in enumerate(panels):
        ax = axes[col]
        ax.set_facecolor('white')

        data = [ctrl_vals, mut_vals]
        if all(len(d) > 0 for d in data):
            bp = ax.boxplot(data, positions=[0, 1], widths=0.5,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='white', linewidth=2.0),
                            whiskerprops=dict(color='#555', linewidth=1.0),
                            capprops=dict(color='#555', linewidth=1.0))
            for patch, col_c in zip(bp['boxes'], [CTRL_COLOR, MUT_COLOR]):
                patch.set_facecolor(col_c)
                patch.set_alpha(0.35)
                patch.set_edgecolor(col_c)
                patch.set_linewidth(1.5)

        for i, (vals, col_c) in enumerate(zip(data, [CTRL_COLOR, MUT_COLOR])):
            if len(vals) > 0:
                jitter = rng.uniform(-0.15, 0.15, len(vals))
                ax.scatter(i + jitter, vals, c=col_c, s=55, alpha=0.7,
                           zorder=5, edgecolors='white', linewidth=0.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', mutant_label], fontsize=16, fontweight='bold')
        for tick, col_c in zip(ax.get_xticklabels(), [CTRL_COLOR, MUT_COLOR]):
            tick.set_color(col_c)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.tick_params(labelsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)

        # Significance
        if 'p' in test:
            stars = _sig_stars(test['p'])
            _cv2 = ctrl_vals[np.isfinite(ctrl_vals)] if len(ctrl_vals) > 0 else ctrl_vals
            _mv2 = mut_vals[np.isfinite(mut_vals)]   if len(mut_vals)  > 0 else mut_vals
            y_max = max(float(np.max(_cv2)) if len(_cv2) > 0 else 0,
                        float(np.max(_mv2)) if len(_mv2) > 0 else 0)
            if y_max > 0:
                _draw_sig_bracket(ax, 0, 1, y_max * 1.05, y_max * 0.06,
                                  f'{stars}\n{_fmt_p(test["p"])}', fontsize=12)
                ax.set_ylim(top=y_max * 1.25)

    fig.suptitle(f'Genotype Comparison: Control vs {mutant_label} (per-recording averages)',
                 fontsize=18, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02,
             f'Control: {len(ctrl_rates)} recordings | '
             f'{mutant_label}: {len(mut_rates)} recordings | '
             f'Mann-Whitney U, two-sided  ·  Each dot = one recording',
             ha='center', fontsize=11, color='#777', style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(activity_dir, 'genotype_comparison.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # =====================================================================
    # FIGURE 2: Longitudinal — metrics across organoid day/age
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(max(18, len(unique_days) * 2.5), 6))
    fig.patch.set_facecolor('white')

    for col, (metric_name, metric_key, ylabel) in enumerate([
        ('Event Frequency', 'mean_rate', 'events/10s'),
        ('Transient Amplitude', 'mean_amp', 'ΔF/F₀'),
        ('Active Fraction', 'active_fraction', 'fraction'),
    ]):
        ax = axes[col]
        ax.set_facecolor('white')

        positions = []
        tick_labels = []
        for day_idx, day in enumerate(unique_days):
            day_ds = [d for d in ds_data if d['day'] == day]
            if not day_ds:
                continue

            for geno, color, offset in [('Control', CTRL_COLOR, -0.15), ('Mutant', MUT_COLOR, 0.15)]:
                geno_ds = [d for d in day_ds if d['genotype'] == geno]
                if not geno_ds:
                    continue
                vals = [d[metric_key] for d in geno_ds]
                x = day_idx + offset
                ax.scatter([x] * len(vals), vals, c=color, s=40, alpha=0.7,
                           zorder=5, edgecolors='white', linewidth=0.5)
                if len(vals) > 1:
                    ax.plot([x, x], [np.mean(vals) - np.std(vals), np.mean(vals) + np.std(vals)],
                            color=color, linewidth=1.5, alpha=0.5)
                    ax.plot(x, np.mean(vals), 's', color=color, markersize=8,
                            zorder=6, markeredgecolor='white', markeredgewidth=0.5)

            positions.append(day_idx)
            tick_labels.append(day)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=13, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(f'{metric_name} by Age', fontsize=15, fontweight='bold')
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)

    # Legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=CTRL_COLOR, markersize=8, label='Control'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=MUT_COLOR, markersize=8, label=mutant_label)]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=13)

    fig.suptitle('Longitudinal Activity by Organoid Age',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(activity_dir, 'longitudinal_by_age.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # =====================================================================
    # FIGURE 3: Combined — genotype × age heatmap
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(unique_days) * 2), 5))
    fig.patch.set_facecolor('white')

    for col, (metric_name, metric_key) in enumerate([
        ('Event Frequency', 'mean_rate'),
        ('Transient Amplitude', 'mean_amp'),
        ('Active Fraction', 'active_fraction'),
    ]):
        ax = axes[col]

        for geno_idx, (geno, color, marker) in enumerate([
            ('Control', CTRL_COLOR, 'o'), ('Mutant', MUT_COLOR, 's')
        ]):
            day_means = []
            day_sems = []
            day_positions = []

            for day_idx, day in enumerate(unique_days):
                geno_ds = [d for d in ds_data if d['day'] == day and d['genotype'] == geno]
                if not geno_ds:
                    continue
                vals = [d[metric_key] for d in geno_ds]
                day_means.append(np.mean(vals))
                day_sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                day_positions.append(day_idx)

            if day_means:
                display_label = mutant_label if geno == 'Mutant' else geno
                ax.errorbar(day_positions, day_means, yerr=day_sems,
                            color=color, marker=marker, markersize=7,
                            capsize=3, linewidth=1.5, label=display_label, alpha=0.8)

        ax.set_xticks(range(len(unique_days)))
        ax.set_xticklabels(unique_days, fontsize=9, rotation=45, ha='right')
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)
        if col == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Genotype × Age: Activity Metrics',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(activity_dir, 'genotype_x_age.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # ── Store results ────────────────────────────────────────────────────
    results['summary'] = {
        'n_datasets': len(ds_data),
        'n_control': len(ctrl_data),
        'n_mutant': len(mut_data),
        'n_days': len(unique_days),
        'days': unique_days,
        'total_active_ctrl': int(len(ctrl_rates)),
        'total_active_mut': int(len(mut_rates)),
        'mean_active_fraction_ctrl': float(np.mean(ctrl_af)) if len(ctrl_af) > 0 else 0,
        'mean_active_fraction_mut': float(np.mean(mut_af)) if len(mut_af) > 0 else 0,
    }

    logger.info(f"  Activity analysis complete:")
    logger.info(f"    Control: {len(ctrl_rates)} active neurons across {len(ctrl_data)} recordings "
                f"(mean active fraction: {np.mean(ctrl_af):.1%})" if len(ctrl_af) > 0 else "")
    logger.info(f"    Mutant: {len(mut_rates)} active neurons across {len(mut_data)} recordings "
                f"(mean active fraction: {np.mean(mut_af):.1%})" if len(mut_af) > 0 else "")

    return results



# =============================================================================
# ROI PEAK FIGURES
# =============================================================================

def generate_roi_peak_figures(datasets: List, output_dir: str) -> None:
    """Generate one PNG per selected ROI showing the peak transient frame,
    a pre-transient reference frame, and the full trace underneath.

    Layout (per file):
        Top row: [reference frame | peak frame]  — side by side, equal size,
                 ROI contour overlaid in cyan, scale bar bottom-right.
        Bottom:  Full denoised + raw trace, spike markers, time cursor at peak.

    ROIs are ranked by peak_SNR = max(denoised) / OASIS_sn and the rank
    number is embedded in the filename so they sort naturally.

    Reference frame selection: scan backwards from the peak frame with a
    minimum margin of 1 frame before the peak, find the lowest-intensity
    frame within the 10 seconds preceding the spike for maximum contrast.
    fluorescence in the ROI bounding box is closest to the rolling baseline
    of the raw trace.  This avoids picking a frame mid-transient.

    Output: figures/ROI Peak Frames/{recording_name}/rank{N:03d}_roi{M:04d}.png
    """

    out_root = os.path.join(output_dir, 'figures', 'ROI Peak Frames')
    os.makedirs(out_root, exist_ok=True)

    ref_margin_frames = 1      # minimum frames before peak to start ref search
    crop_radius    = 45    # pixels around centroid
    fig_w, fig_h   = 10, 5  # inches

    for ds in datasets:
        result_path = Path(ds.filepath)

        # ── Load required arrays ──────────────────────────────────────────
        denoised_path  = result_path / 'traces_denoised.npy'
        raw_path       = result_path / 'temporal_traces.npy'
        spikes_path    = result_path / 'spike_trains.npy'
        noise_path     = result_path / 'deconv_noise.npy'
        footprint_path = result_path / 'spatial_footprints.npz'
        info_path      = result_path / 'run_info.json'

        if not denoised_path.exists() or not spikes_path.exists():
            logger.warning(f"  {ds.name}: missing denoised/spikes, skipping peak figures")
            continue
        if ds.selected_roi_indices is None or len(ds.selected_roi_indices) == 0:
            continue

        C_all  = np.load(denoised_path)
        S_all  = np.load(spikes_path)
        R_all  = np.load(raw_path) if raw_path.exists() else None
        noise  = np.load(noise_path) if noise_path.exists() else None

        # ── Load movie ────────────────────────────────────────────────────
        movie = None
        dims  = None
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            movie_path = info.get('config', {}).get('movie') or info.get('movie')
            if 'dims' in info:
                dims = tuple(info['dims'])
            elif 'd1' in info and 'd2' in info:
                dims = (int(info['d1']), int(info['d2']))

            if movie_path and os.path.exists(movie_path):
                try:
                    ext = os.path.splitext(movie_path)[1].lower()
                    if ext == '.nd2':
                        import nd2
                        _m = nd2.imread(movie_path)
                        movie = (_m[:, 0] if _m.ndim == 4 else _m).astype(np.float32)
                    elif ext in ('.tif', '.tiff'):
                        from tifffile import imread as _tifread
                        movie = _tifread(movie_path).astype(np.float32)
                    elif ext == '.npy':
                        movie = np.load(movie_path).astype(np.float32)
                    logger.info(f"  {ds.name}: loaded movie {movie.shape}")
                except Exception as me:
                    logger.warning(f"  {ds.name}: movie load failed ({me}), using projections only")
                    movie = None

        # ── Load spatial footprints + reconcile dims ──────────────────────
        A_sparse = None
        if footprint_path.exists():
            try:
                A_sparse = load_npz(footprint_path)
                n_pix = A_sparse.shape[0]

                # dims from run_info may reflect the original uncropped movie
                # while footprints use the motion-corrected (cropped) size.
                # Work through several sources in order of reliability:
                dims_ok = dims is not None and dims[0] * dims[1] == n_pix

                if not dims_ok:
                    # 1. Try max_projection / mean_projection saved alongside results
                    for proj_name in ('max_projection.npy', 'mean_projection.npy'):
                        proj_p = result_path / proj_name
                        if proj_p.exists():
                            try:
                                mp = np.load(proj_p)
                                if mp.shape[0] * mp.shape[1] == n_pix:
                                    dims = mp.shape[:2]
                                    dims_ok = True
                                    logger.info(f"  {ds.name}: dims corrected to "
                                                f"{dims} from {proj_name}")
                                    break
                            except Exception:
                                pass

                if not dims_ok and dims is not None:
                    # 2. Motion correction trims symmetrically — try small trims
                    d1_orig, d2_orig = dims
                    for trim in range(1, 25):
                        d1t = d1_orig - 2 * trim
                        d2t = d2_orig - 2 * trim
                        if d1t > 0 and d2t > 0 and d1t * d2t == n_pix:
                            dims = (d1t, d2t)
                            dims_ok = True
                            logger.info(f"  {ds.name}: dims corrected to "
                                        f"{dims} (trimmed {trim}px per side)")
                            break

                if not dims_ok:
                    # 3. Try square
                    side = int(np.sqrt(n_pix))
                    if side * side == n_pix:
                        dims = (side, side)
                        dims_ok = True
                        logger.info(f"  {ds.name}: dims inferred as square {dims}")

                if not dims_ok:
                    logger.warning(f"  {ds.name}: cannot reconcile dims with "
                                   f"footprint ({n_pix} pixels) — frames disabled")
                    A_sparse = None
                else:
                    d1, d2 = dims

            except Exception as ae:
                logger.warning(f"  {ds.name}: footprint load failed: {ae}")

        # ── Global contrast for movie frames ─────────────────────────────
        if movie is not None:
            sample = movie[::max(1, len(movie)//200)]
            vmin = float(np.percentile(sample, 1))
            vmax = float(np.percentile(sample, 99.5))
            # Align movie spatial dims to footprint dims if needed
            _, mh, mw = movie.shape
            if dims is not None:
                fh, fw = dims
                if mh != fh or mw != fw:
                    yo = (mh - fh) // 2
                    xo = (mw - fw) // 2
                    if yo >= 0 and xo >= 0:
                        movie = movie[:, yo:yo+fh, xo:xo+fw]
        else:
            vmin = vmax = None

        # ── Per-ROI SNR ranking ───────────────────────────────────────────
        roi_indices = ds.selected_roi_indices
        N_sel = len(roi_indices)
        snr_scores = np.zeros(N_sel)
        for j, orig_roi in enumerate(roi_indices):
            if orig_roi >= C_all.shape[0]:
                continue
            den = C_all[orig_roi]
            if noise is not None and orig_roi < len(noise) and noise[orig_roi] > 0:
                snr_scores[j] = float(np.max(den)) / float(noise[orig_roi])
            else:
                diff = np.diff(den)
                mad = 1.4826 * float(np.median(np.abs(diff - np.median(diff)))) / np.sqrt(2)
                if mad > 0:
                    snr_scores[j] = float(np.percentile(den, 95) - np.percentile(den, 50)) / mad

        rank_order = np.argsort(snr_scores)[::-1]  # highest SNR = rank 1

        frame_rate  = ds.frame_rate
        T           = C_all.shape[1]
        t_ax        = np.arange(T) / frame_rate
        ref_margin  = ref_margin_frames
        bl_window   = max(10, int(5.0 * frame_rate))  # 5s rolling baseline window

        rec_dir = os.path.join(out_root, ds.name)
        os.makedirs(rec_dir, exist_ok=True)

        for rank_pos, sel_j in enumerate(rank_order):
            orig_roi = int(roi_indices[sel_j])
            rank_num = rank_pos + 1

            if orig_roi >= C_all.shape[0]:
                continue

            den   = C_all[orig_roi]
            raw   = R_all[orig_roi] if R_all is not None and orig_roi < R_all.shape[0] else None
            spikes = S_all[orig_roi] if orig_roi < S_all.shape[0] else np.zeros(T)

            # ── Find peak transient frame ─────────────────────────────────
            peak_frame = int(np.argmax(den))

            # ── Find reference frame ──────────────────────────────────────
            # Lowest mean ROI fluorescence in the 10s window before the peak.
            # Using raw trace minimum for maximum contrast with the transient.
            search_window_frames = max(1, int(10.0 * frame_rate))
            search_start = max(0, peak_frame - search_window_frames)
            search_end   = max(0, peak_frame - ref_margin_frames)
            if raw is not None and search_end > search_start:
                window_vals = raw[search_start:search_end]
                ref_frame = search_start + int(np.argmin(window_vals))
            else:
                ref_frame = max(0, peak_frame - ref_margin_frames)

            # ── Get spatial crop bounds ───────────────────────────────────
            cy, cx = None, None
            fp_mask = None
            if A_sparse is not None and dims is not None and orig_roi < A_sparse.shape[1]:
                fp = A_sparse[:, orig_roi].toarray().ravel()
                if len(fp) == d1 * d2:
                    fp_2d = fp.reshape(d1, d2)
                    ys, xs = np.where(fp_2d > 0)
                    if len(ys) > 0:
                        cy, cx = int(np.mean(ys)), int(np.mean(xs))
                        y0 = max(0, cy - crop_radius)
                        y1 = min(d1, cy + crop_radius)
                        x0 = max(0, cx - crop_radius)
                        x1 = min(d2, cx + crop_radius)
                        fp_mask = (fp_2d[y0:y1, x0:x1] > 0).astype(np.uint8)

            # ── Extract frame crops ───────────────────────────────────────
            def _frame_to_rgb(frame_idx):
                """Return normalised RGB uint8 crop with contour overlay."""
                if movie is None or cy is None:
                    return None
                raw_crop = movie[frame_idx, y0:y1, x0:x1].astype(np.float64)
                norm = np.clip((raw_crop - vmin) / (vmax - vmin + 1e-10), 0, 1)
                rgb = np.stack([norm * 0.25, norm * 0.9, norm * 0.25], axis=-1)
                # Cyan contour overlay
                if fp_mask is not None:
                    from scipy.ndimage import binary_dilation
                    edge = binary_dilation(fp_mask > 0) & ~(fp_mask > 0)
                    mh, mw = min(fp_mask.shape[0], rgb.shape[0]), min(fp_mask.shape[1], rgb.shape[1])
                    rgb[:mh, :mw, 0][edge[:mh, :mw]] = 0.0
                    rgb[:mh, :mw, 1][edge[:mh, :mw]] = 1.0
                    rgb[:mh, :mw, 2][edge[:mh, :mw]] = 1.0
                return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

            ref_img  = _frame_to_rgb(ref_frame)
            peak_img = _frame_to_rgb(peak_frame)

            # ── Figure ────────────────────────────────────────────────────
            has_frames = ref_img is not None and peak_img is not None
            fig = plt.figure(figsize=(fig_w, fig_h), facecolor='black')

            if has_frames:
                gs = gridspec.GridSpec(2, 2, figure=fig,
                                       height_ratios=[1.6, 1],
                                       hspace=0.12, wspace=0.06,
                                       left=0.06, right=0.97,
                                       top=0.93, bottom=0.10)
                ax_ref  = fig.add_subplot(gs[0, 0])
                ax_peak = fig.add_subplot(gs[0, 1])
                ax_tr   = fig.add_subplot(gs[1, :])
            else:
                gs = gridspec.GridSpec(1, 1, figure=fig,
                                       left=0.06, right=0.97,
                                       top=0.93, bottom=0.10)
                ax_tr = fig.add_subplot(gs[0, 0])

            # ── Top: reference frame ──────────────────────────────────────
            if has_frames:
                ax_ref.imshow(ref_img, interpolation='nearest', aspect='equal')
                ax_ref.set_title(
                    f'Reference  t={ref_frame/frame_rate:.1f}s',
                    color='#8BA4B8', fontsize=8, pad=3)
                ax_ref.axis('off')

                ax_peak.imshow(peak_img, interpolation='nearest', aspect='equal')
                ax_peak.set_title(
                    f'Peak  t={peak_frame/frame_rate:.1f}s',
                    color='#00e676', fontsize=8, pad=3)
                ax_peak.axis('off')

            # ── Bottom: full trace ────────────────────────────────────────
            ax_tr.set_facecolor('black')

            # Raw trace
            if raw is not None:
                ax_tr.plot(t_ax, raw, color='#606878', linewidth=0.6,
                           alpha=0.7, zorder=2)

            # Denoised trace
            ax_tr.plot(t_ax, den, color='#00e676', linewidth=1.1,
                       zorder=3, label='Denoised')

            # Spike markers — all accepted spikes (≥2.5σ + s_min=0.1)
            spk_frames = np.where(spikes > 0)[0]
            if len(spk_frames) > 0:
                ax_tr.scatter(t_ax[spk_frames], den[spk_frames],
                              color='#D32F2F', s=18, zorder=5, marker='v')

            # Vertical line at peak and at reference frame
            ax_tr.axvline(peak_frame / frame_rate,
                          color='#00e676', linewidth=0.9, alpha=0.5,
                          linestyle='--', zorder=4)
            ax_tr.axvline(ref_frame / frame_rate,
                          color='#8BA4B8', linewidth=0.9, alpha=0.5,
                          linestyle='--', zorder=4)

            ax_tr.set_xlim(0, t_ax[-1])
            ax_tr.set_xlabel('Time (s)', color='white', fontsize=8)
            ax_tr.set_ylabel('ΔF/F₀', color='white', fontsize=8)
            ax_tr.tick_params(colors='white', labelsize=7)
            for sp in ax_tr.spines.values():
                sp.set_color('#333344')
            ax_tr.set_facecolor('#08090f')

            # ── Title ─────────────────────────────────────────────────────
            snr_val = snr_scores[sel_j]
            n_spk   = int(np.sum(spikes > 0))
            fig.suptitle(
                f'Rank {rank_num}  ·  ROI {orig_roi}  ·  '
                f'SNR {snr_val:.1f}  ·  {n_spk} spikes  ·  {ds.name}',
                color='white', fontsize=9, fontweight='bold', y=0.98)

            fname = os.path.join(rec_dir,
                                 f'rank{rank_num:03d}_roi{orig_roi:04d}.png')
            fig.savefig(fname, dpi=150, bbox_inches='tight',
                        facecolor='black')
            plt.close(fig)

        n_saved = len(rank_order)
        logger.info(f"  {ds.name}: saved {n_saved} ROI peak figures → {rec_dir}/")

    logger.info(f"ROI peak figures complete → {out_root}/")


