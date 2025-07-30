from tsaug import TimeWarp, Drift, AddNoise, Pool

def get_ecg_augmenter():
    """
    Returns a composite augmenter for ECG signals.
    """
    augmenter = (
        TimeWarp(n_speed_change=3, max_speed_ratio=2) 
        + Drift(max_drift=(0.05, 0.1), n_drift_points=5)
        + AddNoise(scale=0.05)
        + Pool(size=3)
    )
    return augmenter
