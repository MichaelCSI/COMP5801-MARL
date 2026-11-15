
"""Part C: reproducible evaluation and statistical analysis of peer-SoC MAPPO."""
from .metrics import summarise_record, battery_synchrony
from .statistics import paired_interval, paired_comparisons, seed_summary, robustness_table

__all__ = ["summarise_record", "battery_synchrony", "paired_interval",
           "paired_comparisons", "seed_summary", "robustness_table"]
