"""Signal triggers for trading decisions.

This module provides trigger functions that convert continuous
signals into discrete trading signals.

Z-Score Trigger (Spec §2.1):
    z_t = (I_t - mean(I, window)) / std(I, window)
    trigger = |z_t| > threshold

Streak Trigger (Spec §2.2):
    streak_t = consecutive same-sign count
    trigger = streak_t >= min_streak
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_zscore(
    series: pd.Series,
    window: int = 252,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.Series:
    """
    Compute rolling z-score of a series.

    Z-score measures how many standard deviations a value is
    from the rolling mean.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    window : int, default 252
        Rolling window size in periods.
    min_periods : int | None, optional
        Minimum observations required. Defaults to window // 2.
    ddof : int, default 1
        Degrees of freedom for std calculation.

    Returns
    -------
    pd.Series
        Z-score series, same index as input.
        NaN for insufficient history.

    Examples
    --------
    >>> z = compute_zscore(imbalance, window=252)
    >>> z.dropna().head()
    date        cusip
    2023-01-15  037833100    2.5
    ...

    Notes
    -----
    First (min_periods - 1) values will be NaN.
    Uses sample std (ddof=1) by default.
    """
    if min_periods is None:
        min_periods = window // 2

    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=ddof)

    zscore = (series - rolling_mean) / rolling_std
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    zscore.name = "zscore"

    return zscore


def zscore_trigger(
    series: pd.Series,
    window: int = 252,
    threshold: float = 7.0,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Generate mean-reversion trigger based on z-score.

    Triggers when the absolute z-score exceeds the threshold,
    indicating extreme deviation from the rolling mean.

    Parameters
    ----------
    series : pd.Series
        Input signal series.
    window : int, default 252
        Rolling window for z-score calculation.
    threshold : float, default 7.0
        Z-score threshold for triggering.
    min_periods : int | None, optional
        Minimum observations required.

    Returns
    -------
    pd.Series
        Boolean series. True when |z| > threshold.

    Examples
    --------
    >>> trigger = zscore_trigger(imbalance, threshold=7.0)
    >>> trigger.sum()
    42  # 42 trigger events

    Notes
    -----
    A threshold of 7 is very extreme (7 sigma event).
    In practice, lower thresholds (2-3) are more common.
    """
    zscore = compute_zscore(series, window=window, min_periods=min_periods)
    trigger = np.abs(zscore) > threshold
    trigger.name = "zscore_trigger"

    n_triggers = trigger.sum()
    logger.info(
        f"Z-score trigger (threshold={threshold}): {n_triggers} events "
        f"({100 * n_triggers / len(trigger):.2f}%)"
    )

    return trigger


def compute_streak(
    series: pd.Series,
    reset_on_zero: bool = True,
) -> pd.Series:
    """
    Compute consecutive same-sign streak length.

    Counts how many consecutive periods the series has maintained
    the same sign (positive or negative).

    Parameters
    ----------
    series : pd.Series
        Input signal series.
    reset_on_zero : bool, default True
        If True, zero values reset the streak.
        If False, zero values continue the previous sign's streak.

    Returns
    -------
    pd.Series
        Streak length series. Positive for positive streaks,
        negative for negative streaks.

    Examples
    --------
    >>> streaks = compute_streak(imbalance)
    >>> streaks.abs().max()
    8  # Longest streak was 8 days

    Notes
    -----
    The sign of the streak indicates the direction:
    - Positive streak value: consecutive positive values
    - Negative streak value: consecutive negative values
    """
    # Get sign of series
    if reset_on_zero:
        sign = np.sign(series)
    else:
        sign = np.sign(series).replace(0, np.nan).ffill().fillna(0)

    # Detect sign changes
    sign_change = sign != sign.shift(1)

    # Create groups for each streak
    streak_groups = sign_change.cumsum()

    # Count within each group
    streak_count = series.groupby(streak_groups).cumcount() + 1

    # Apply sign to streak
    streak = streak_count * sign
    streak.name = "streak"

    return streak


def streak_trigger(
    series: pd.Series,
    min_streak: int = 3,
    reset_on_zero: bool = True,
) -> pd.Series:
    """
    Generate momentum trigger based on streak length.

    Triggers when the series has maintained the same sign
    for at least min_streak consecutive periods.

    Parameters
    ----------
    series : pd.Series
        Input signal series.
    min_streak : int, default 3
        Minimum consecutive periods for trigger.
    reset_on_zero : bool, default True
        If True, zero values reset the streak.

    Returns
    -------
    pd.Series
        Boolean series. True when |streak| >= min_streak.

    Examples
    --------
    >>> trigger = streak_trigger(imbalance, min_streak=3)
    >>> trigger.sum()
    156  # 156 trigger events

    Notes
    -----
    This is a momentum signal: sustained same-direction flow
    may predict continuation.
    """
    streak = compute_streak(series, reset_on_zero=reset_on_zero)
    trigger = np.abs(streak) >= min_streak
    trigger.name = "streak_trigger"

    n_triggers = trigger.sum()
    logger.info(
        f"Streak trigger (min_streak={min_streak}): {n_triggers} events "
        f"({100 * n_triggers / len(trigger):.2f}%)"
    )

    return trigger


def combined_trigger(
    series: pd.Series,
    zscore_window: int = 252,
    zscore_threshold: float = 7.0,
    min_streak: int = 3,
    require_both: bool = False,
) -> pd.Series:
    """
    Generate combined z-score and streak trigger.

    Parameters
    ----------
    series : pd.Series
        Input signal series.
    zscore_window : int
        Window for z-score calculation.
    zscore_threshold : float
        Z-score threshold.
    min_streak : int
        Minimum streak length.
    require_both : bool, default False
        If True, require both triggers to fire.
        If False, trigger on either.

    Returns
    -------
    pd.Series
        Boolean trigger series.
    """
    z_trigger = zscore_trigger(series, window=zscore_window, threshold=zscore_threshold)
    s_trigger = streak_trigger(series, min_streak=min_streak)

    if require_both:
        trigger = z_trigger & s_trigger
    else:
        trigger = z_trigger | s_trigger

    trigger.name = "combined_trigger"
    return trigger
