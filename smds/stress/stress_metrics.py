from enum import Enum


class StressMetrics(Enum):
    SCALE_NORMALIZED_STRESS = "scale_normalized_stress"
    NON_METRIC_STRESS = "non_metric_stress"
    #RAW_STRESS = "raw_stress" ToDo: Check/Resolve error
    SHEPARD_GOODNESS_SCORE = "shepard_goodness_score"
    NORMALIZED_STRESS = "normalized_stress"
    NORMALIZED_KL_DIVERGENCE = "normalized_kl_divergence"
