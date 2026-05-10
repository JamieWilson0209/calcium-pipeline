"""
Configuration Loader
====================

YAML-based pipeline configuration.  One file, one source of truth — all
parameters for every pipeline stage live in the YAML config.

Usage
-----
    from config_loader import load_config
    cfg = load_config('config/default.yaml')
    print(cfg.imaging.frame_rate)
    print(cfg.detection.min_radius)

CLI overrides
-------------
    cfg = load_config('config/default.yaml',
                      overrides={'imaging.frame_rate': 30})

Dotted paths traverse nested sections. CLI overrides always win over
YAML values.

Backwards compatibility
-----------------------
Unknown top-level sections and unknown fields within sections are
silently skipped, so old/partial YAML files keep working as the schema
evolves.
"""

import os
import yaml
import logging
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR DECAY TIMES
# ─────────────────────────────────────────────────────────────────────────────
#
# Observable transient durations for OASIS / CaImAn / Suite2p, NOT raw
# biophysical half-decay times (which are 2–3× shorter).
#
# References:
#   CaImAn: 0.4 s (fast), 1–2 s (slow)          Giovannucci et al. 2019
#   Suite2p: 0.7 s (6f), 1.0 s (6m), 1.25 s (6s) Pachitariu et al. 2017
#   OASIS: 0.5/1.0/2.0 s (fast/med/slow)         Friedrich et al. 2017
#   GCaMP6: Chen et al. 2013, Nature 499:295
#   jGCaMP7: Dana et al. 2019, Nat Methods 16:649
#   jGCaMP8: Zhang et al. 2023, Nature 615:884

INDICATOR_DECAY_TIMES = {
    # GCaMP6 (Chen et al. 2013)
    'gcamp6f': 0.4,
    'gcamp6m': 1.0,
    'gcamp6s': 2.0,

    # GCaMP5 (Akerboom et al. 2012)
    'gcamp5k': 1.0,
    'gcamp5g': 1.0,

    # jGCaMP7 (Dana et al. 2019)
    'jgcamp7f': 0.5,   'gcamp7f': 0.5,
    'jgcamp7s': 1.25,  'gcamp7s': 1.25,
    'jgcamp7b': 0.8,
    'jgcamp7c': 0.8,

    # jGCaMP8 (Zhang et al. 2023)
    'jgcamp8f': 0.3,   'gcamp8f': 0.3,
    'jgcamp8m': 0.3,   'gcamp8m': 0.3,
    'jgcamp8s': 0.5,   'gcamp8s': 0.5,

    # Red indicators (Dana et al. 2016)
    'jrgeco1a': 0.7,
    'jrcamp1a': 0.7,   'rcamp1a': 0.7,

    # Synthetic dyes
    'ogb1': 0.7,       'ogb-1': 0.7,      'ogb1-am': 0.7,
    'fluo4': 0.4,      'fluo-4': 0.4,     'fluo4am': 0.4,   'fluo-4am': 0.4,
    'fura2': 1.0,      'fura-2': 1.0,
    'calbryte520': 0.5,
    'calbryte590': 0.5,

    # Fallback
    'default': 1.0,
}


def get_decay_time_for_indicator(indicator: str) -> float:
    """Look up the recommended decay time for a calcium indicator."""
    key = indicator.lower().replace('-', '').replace('_', '')

    if key in INDICATOR_DECAY_TIMES:
        return INDICATOR_DECAY_TIMES[key]

    # Try without 'j' prefix (jGCaMP7f → gcamp7f)
    if key.startswith('j') and key[1:] in INDICATOR_DECAY_TIMES:
        return INDICATOR_DECAY_TIMES[key[1:]]

    # Try adding 'j' prefix (gcamp7f → jgcamp7f)
    if ('j' + key) in INDICATOR_DECAY_TIMES:
        return INDICATOR_DECAY_TIMES['j' + key]

    logger.warning(
        f"Unknown indicator '{indicator}', using default "
        f"decay_time={INDICATOR_DECAY_TIMES['default']}s"
    )
    return INDICATOR_DECAY_TIMES['default']


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES — ONE PER PIPELINE STAGE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImagingConfig:
    """Acquisition parameters describing the input data."""
    frame_rate: float = 2.0          # Hz
    pixel_size_um: float = 2.0
    indicator: str = "fluo4"
    fov_size_um: List[float] = field(default_factory=lambda: [512, 512])

    @property
    def decay_time(self) -> float:
        """Decay time in seconds, resolved from indicator name."""
        return get_decay_time_for_indicator(self.indicator)


@dataclass
class MotionConfig:
    """CaImAn NoRMCorre motion correction parameters."""
    enabled: bool = True
    mode: str = "rigid"              # rigid | piecewise_rigid | auto
    max_shift: int = 20              # px
    # CaImAn defaults — overridable but rarely tuned
    pw_strides: Tuple[int, int] = (96, 96)
    pw_overlaps: Tuple[int, int] = (48, 48)
    niter_rig: int = 2
    num_frames_split: int = 100


@dataclass
class AutoRadiusConfig:
    """Auto-radius optimisation parameters."""
    enabled: bool = True
    snr_threshold: float = 5.0       # Min trace SNR for "good" neuron
    n_candidates: int = 5            # Radius settings to sweep
    radius_min: float = 3.0          # Sweep lower bound
    radius_max: float = 35.0         # Sweep upper bound


@dataclass
class DetectionConfig:
    """Seed detection + contour extraction parameters."""
    # Radii — overridden by auto_radius if enabled
    min_radius: float = 10.0         # px
    max_radius: float = 25.0         # px

    # Blob detection thresholds
    intensity_threshold: float = 0.18
    correlation_threshold: float = 0.12
    max_seeds: int = 500
    smooth_sigma: float = 4.0        # Gaussian smoothing for hotspot suppression
    border_margin: int = 20          # px from FOV edge

    # Contour extraction
    contour_method: str = "otsu"     # otsu | triangle
    contour_fallback: bool = True    # Generate circular footprint when Otsu fails
    fallback_method: str = "gaussian"  # gaussian | disk (for contour_fallback)

    # Contour-overlap merge (shape-aware fusion of overlapping contours)
    contour_merge_min_overlap: float = 0.4   # |A∩B| / min(|A|,|B|)
    contour_merge_iou: float = 0.2           # |A∩B| / |A∪B|
    contour_merge_max_growth: float = 4.0    # Reject if hull > N× max member

    # Projection selection
    use_temporal_projection: bool = True
    n_peak_frames: int = 10
    peak_percentile: float = 90.0
    use_mean_proj: bool = True
    use_std_proj: bool = True

    # Nested auto-radius config
    auto_radius: AutoRadiusConfig = field(default_factory=AutoRadiusConfig)


@dataclass
class BaselineConfig:
    """Per-trace ΔF/F₀ baseline correction parameters.

    Applies to `global_dff`, `local_dff`, and `local_background` methods.
    When `method = "direct"`, raw traces are passed to deconvolution
    untouched.
    """
    method: str = "global_dff"       # direct | global_dff | local_dff | local_background
    edge_trim: bool = False

    # Rolling baseline
    percentile: float = 8.0
    window_fraction: float = 0.25    # As fraction of T
    min_window: int = 50             # Frames
    max_window: int = 500            # Frames

    # Local-background-only parameters
    annulus_inner_gap: int = 2       # px
    annulus_outer_radius: int = 20   # px
    tissue_threshold_method: str = "otsu"  # otsu | percentile
    min_annulus_pixels: int = 50


@dataclass
class DeconvolutionConfig:
    """Spike inference parameters (OASIS)."""
    enabled: bool = True
    method: str = "oasis"            # oasis | threshold
    s_min: float = 0.1               # Min spike amplitude in ΔF/F₀
    noise_gate: float = 3.5          # σ — spikes below this are zeroed
    # OASIS internals
    penalty: float = 0.0             # L1 sparsity; 0 = auto-tune
    optimize_g: bool = True
    noise_method: str = "mean"

    # Optional pre-deconvolution temporal filter
    temporal_filter: bool = False
    filter_cutoff: float = 2.0       # Hz


@dataclass
class OutputConfig:
    """Output & visualisation options."""
    # New (v3.7+) gallery flags — split per-ROI PNGs from the interactive HTML
    per_roi_pngs: bool = False              # Per-ROI inspection PNGs (slow)
    inspection_gallery: bool = True         # Interactive HTML gallery (fast)
    # Legacy / env-var controlled
    movie_gallery: bool = False             # Full movie + contour overlay HTML
    movie_gallery_subsample: int = 4        # Temporal subsampling factor


@dataclass
class QualityConfig:
    """Quality thresholds used by group analysis."""
    drift_threshold: float = 1.0
    motion_max_threshold: float = 15.0
    motion_residual_threshold: float = 2.0
    min_roi_distance: float = 15.0          # px
    inactive_file: Optional[str] = None
    # Genotype classification: maps line-prefix → genotype label.
    # Prefixes not listed fall through to 'default'.
    genotype_map: Dict[str, str] = field(
        default_factory=lambda: {'3': 'Control', 'default': 'Mutant'}
    )
    mutant_label: str = "CEP41 R242H"       # Display label for mutant genotype
    roi_peak_figures: bool = False          # Generate per-ROI peak frame PNGs (slow)


@dataclass
class DevConfig:
    """Experimental / development features."""
    network_analysis: bool = False


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.

    Each pipeline stage gets its own section.  Sections are loaded from
    YAML; unknown sections and fields are silently skipped.
    """
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    deconvolution: DeconvolutionConfig = field(default_factory=DeconvolutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    dev: DevConfig = field(default_factory=DevConfig)

    # Optional override for decay time; auto-resolved from indicator if None.
    decay_time: Optional[float] = None

    def resolve_decay_time(self):
        """Set decay_time from indicator if not explicitly overridden."""
        if self.decay_time is None:
            self.decay_time = self.imaging.decay_time
            logger.info(
                f"Auto-resolved decay_time={self.decay_time}s "
                f"for {self.imaging.indicator}"
            )

    def validate(self) -> List[str]:
        """Return a list of validation warnings/errors."""
        issues = []

        if self.imaging.frame_rate <= 0:
            issues.append("ERROR: imaging.frame_rate must be positive")
        if self.imaging.pixel_size_um <= 0:
            issues.append("ERROR: imaging.pixel_size_um must be positive")

        key = self.imaging.indicator.lower().replace('-', '').replace('_', '')
        if key not in INDICATOR_DECAY_TIMES:
            issues.append(
                f"WARNING: Unknown indicator '{self.imaging.indicator}', "
                f"using default decay_time"
            )

        if self.detection.min_radius >= self.detection.max_radius:
            issues.append("ERROR: detection.min_radius must be < max_radius")

        if self.baseline.method not in ('direct', 'global_dff', 'local_dff', 'local_background'):
            issues.append(
                f"ERROR: baseline.method must be one of "
                f"direct / global_dff / local_dff / local_background, "
                f"got '{self.baseline.method}'"
            )

        if self.motion.mode not in ('rigid', 'piecewise_rigid', 'auto'):
            issues.append(
                f"ERROR: motion.mode must be one of rigid / piecewise_rigid / auto, "
                f"got '{self.motion.mode}'"
            )

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (for YAML/JSON output)."""
        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {
                    k: convert(v)
                    for k, v in dataclasses.asdict(obj).items()
                    if not k.startswith('_')
                }
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj
        return convert(self)

    def save(self, path: str):
        """Write configuration to a YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to: {path}")


# Backwards-compatible alias
PreprocessingConfig = BaselineConfig


# ─────────────────────────────────────────────────────────────────────────────
# YAML LOADING
# ─────────────────────────────────────────────────────────────────────────────

# Maps top-level YAML section names → dataclass types.
_SECTION_MAP = {
    'imaging':        ImagingConfig,
    'motion':         MotionConfig,
    'detection':      DetectionConfig,
    'baseline':       BaselineConfig,
    'deconvolution':  DeconvolutionConfig,
    'output':         OutputConfig,
    'quality':        QualityConfig,
    'dev':            DevConfig,
    # Backwards-compat aliases
    'preprocessing':  BaselineConfig,
}


def _load_section(dataclass_type, raw_dict: dict):
    """Instantiate a dataclass from a dict, recursively handling nested dataclasses."""
    valid_fields = {f.name: f for f in dataclasses.fields(dataclass_type)}
    kwargs = {}
    skipped = []

    for k, v in raw_dict.items():
        if k not in valid_fields:
            skipped.append(k)
            continue
        field_type = valid_fields[k].type
        # Handle nested dataclass (e.g. detection.auto_radius)
        if isinstance(v, dict) and dataclasses.is_dataclass(field_type):
            kwargs[k] = _load_section(field_type, v)
        # Handle tuple fields (e.g. pw_strides)
        elif isinstance(v, list) and str(field_type).startswith("typing.Tuple"):
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v

    if skipped:
        logger.debug(f"Skipped unknown fields in {dataclass_type.__name__}: {skipped}")
    return dataclass_type(**kwargs)


def _apply_override(config: PipelineConfig, dotted_key: str, value: Any):
    """
    Apply a single CLI override via dotted path (e.g. 'imaging.frame_rate').
    """
    parts = dotted_key.split('.')
    obj = config
    for part in parts[:-1]:
        if not hasattr(obj, part):
            logger.warning(f"Override target '{dotted_key}': no such section '{part}'")
            return
        obj = getattr(obj, part)
    last = parts[-1]
    if not hasattr(obj, last):
        logger.warning(f"Override target '{dotted_key}': no such field '{last}'")
        return
    setattr(obj, last, value)
    logger.info(f"  Override: {dotted_key} = {value!r}")


def load_config(
    config_path: str,
    validate: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> PipelineConfig:
    """
    Load configuration from a YAML file, optionally applying CLI overrides.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    validate : bool
        Run validation after loading (default True).
    overrides : dict, optional
        Dotted-key overrides, e.g. ``{'imaging.frame_rate': 30}``.
        None values are ignored (so argparse defaults don't override YAML).

    Returns
    -------
    PipelineConfig
    """
    logger.info(f"Loading configuration from: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"Expected YAML dict, got {type(raw_config).__name__}")

    config = PipelineConfig()

    for section_name, dc_type in _SECTION_MAP.items():
        if section_name in raw_config and isinstance(raw_config[section_name], dict):
            setattr(config, section_name, _load_section(dc_type, raw_config[section_name]))

    if 'decay_time' in raw_config and raw_config['decay_time'] is not None:
        config.decay_time = float(raw_config['decay_time'])

    unknown = set(raw_config.keys()) - set(_SECTION_MAP.keys()) - {'decay_time'}
    if unknown:
        logger.debug(f"Ignored unknown YAML sections: {unknown}")

    # Apply CLI overrides (None values are skipped so argparse defaults don't win)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                _apply_override(config, key, value)

    config.resolve_decay_time()

    if validate:
        issues = config.validate()
        for issue in issues:
            if issue.startswith("ERROR"):
                logger.error(issue)
            else:
                logger.warning(issue)

        errors = [i for i in issues if i.startswith("ERROR")]
        if errors:
            raise ValueError(
                f"Configuration validation failed with {len(errors)} error(s)"
            )

    logger.info("Configuration loaded successfully")
    return config


def create_default_config(output_path: str) -> PipelineConfig:
    """Create and save a default configuration file."""
    config = PipelineConfig()
    config.save(output_path)
    return config
