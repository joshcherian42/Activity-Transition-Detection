import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
import scipy.integrate as integrate
from scipy import signal
import extract_features
import math


def parse_data(raw_data, start_time, end_time, subject, max_heart_rate, min_heart_rate, writer):

    time = [round(elem, 2) for elem in raw_data["Time_s"]][start_time:end_time]
    x_hand = raw_data["hand_accx_16g"][start_time:end_time]
    y_hand = raw_data["hand_accy_16g"][start_time:end_time]
    z_hand = raw_data["hand_accz_16g"][start_time:end_time]
    x_chest = raw_data["chest_accx_16g"][start_time:end_time]
    y_chest = raw_data["chest_accy_16g"][start_time:end_time]
    z_chest = raw_data["chest_accz_16g"][start_time:end_time]
    x_ankle = raw_data["ankle_accx_16g"][start_time:end_time]
    y_ankle = raw_data["ankle_accy_16g"][start_time:end_time]
    z_ankle = raw_data["ankle_accz_16g"][start_time:end_time]
    heart_rate = raw_data['heart_rate'][start_time:end_time]
    activities = raw_data['gesture'][start_time:end_time]

    cur_features = [  # Hand
                      mean(x_hand), mean(y_hand), mean(z_hand),
                      mean(extract_features.euclidean_distance(x_hand, y_hand, z_hand)),
                      median(x_hand), median(y_hand), median(z_hand),
                      median(extract_features.euclidean_distance(x_hand, y_hand, z_hand)),
                      stdev(x_hand), stdev(y_hand), stdev(z_hand),
                      stdev(extract_features.euclidean_distance(x_hand, y_hand, z_hand)),
                      peak_val(x_hand), peak_val(y_hand), peak_val(z_hand),
                      peak_val(extract_features.euclidean_distance(x_hand, y_hand, z_hand)),
                      extract_features.energy(x_hand), extract_features.energy(y_hand), extract_features.energy(z_hand),
                      extract_features.energy(extract_features.euclidean_distance(x_hand, y_hand, z_hand)),
                      feature_abs_integral(time, x_hand), feature_abs_integral(time, y_hand), feature_abs_integral(time, z_hand),
                      feature_abs_integral(time, pairwise(map(abs, x_hand), map(abs, y_hand), axis3=map(abs, z_hand))),
                      correlation(x_hand, y_hand), correlation(x_hand, z_hand), correlation(y_hand, z_hand),
                      power_ratio(x_hand), power_ratio(y_hand), power_ratio(z_hand),
                      power_ratio(extract_features.euclidean_distance(x_hand, y_hand, z_hand)),
                      peak_psd(x_hand), peak_psd(y_hand), peak_psd(z_hand),
                      peak_psd(extract_features.euclidean_distance(x_hand, y_hand, z_hand)),
                      entropy(x_hand), entropy(y_hand), entropy(z_hand),
                      entropy(extract_features.euclidean_distance(x_hand, y_hand, z_hand)),

                      # Chest
                      mean(x_chest), mean(y_chest), mean(z_chest),
                      mean(extract_features.euclidean_distance(x_chest, y_chest, z_chest)),
                      median(x_chest), median(y_chest), median(z_chest),
                      median(extract_features.euclidean_distance(x_chest, y_chest, z_chest)),
                      stdev(x_chest), stdev(y_chest), stdev(z_chest),
                      stdev(extract_features.euclidean_distance(x_chest, y_chest, z_chest)),
                      peak_val(x_chest), peak_val(y_chest), peak_val(z_chest),
                      peak_val(extract_features.euclidean_distance(x_chest, y_chest, z_chest)),
                      extract_features.energy(x_chest), extract_features.energy(y_chest), extract_features.energy(z_chest),
                      extract_features.energy(extract_features.euclidean_distance(x_chest, y_chest, z_chest)),
                      feature_abs_integral(time, x_chest), feature_abs_integral(time, y_chest), feature_abs_integral(time, z_chest),
                      feature_abs_integral(time, pairwise(map(abs, x_chest), map(abs, y_chest), axis3=map(abs, z_chest))),
                      correlation(x_chest, y_chest), correlation(x_chest, z_chest), correlation(y_chest, z_chest),
                      power_ratio(x_chest), power_ratio(y_chest), power_ratio(z_chest),
                      power_ratio(extract_features.euclidean_distance(x_chest, y_chest, z_chest)),
                      peak_psd(x_chest), peak_psd(y_chest), peak_psd(z_chest),
                      peak_psd(extract_features.euclidean_distance(x_chest, y_chest, z_chest)),
                      entropy(x_chest), entropy(y_chest), entropy(z_chest),
                      entropy(extract_features.euclidean_distance(x_chest, y_chest, z_chest)),

                      # Ankle
                      mean(x_ankle), mean(y_ankle), mean(z_ankle),
                      mean(extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle)),
                      median(x_ankle), median(y_ankle), median(z_ankle),
                      median(extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle)),
                      stdev(x_ankle), stdev(y_ankle), stdev(z_ankle),
                      stdev(extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle)),
                      peak_val(x_ankle), peak_val(y_ankle), peak_val(z_ankle),
                      peak_val(extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle)),
                      extract_features.energy(x_ankle), extract_features.energy(y_ankle), extract_features.energy(z_ankle),
                      extract_features.energy(extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle)),
                      feature_abs_integral(time, x_ankle), feature_abs_integral(time, y_ankle), feature_abs_integral(time, z_ankle),
                      feature_abs_integral(time, pairwise(map(abs, x_ankle), map(abs, y_ankle), axis3=map(abs, z_ankle))),
                      correlation(x_ankle, y_ankle), correlation(x_ankle, z_ankle), correlation(y_ankle, z_ankle),
                      power_ratio(x_ankle), power_ratio(y_ankle), power_ratio(z_ankle),
                      power_ratio(extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle)),
                      peak_psd(x_ankle), peak_psd(y_ankle), peak_psd(z_ankle),
                      peak_psd(extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle)),
                      entropy(x_ankle), entropy(y_ankle), entropy(z_ankle),
                      entropy(extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle)),

                      # Pairwise
                      mean(pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_chest, y_chest, z_chest))),
                      stdev(pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_chest, y_chest, z_chest))),
                      feature_abs_integral(time, pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_chest, y_chest, z_chest))),
                      extract_features.energy(pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_chest, y_chest, z_chest))),

                      mean(pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),
                      stdev(pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),
                      feature_abs_integral(time, pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),
                      extract_features.energy(pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),

                      mean(pairwise(extract_features.euclidean_distance(x_chest, y_chest, z_chest), extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),
                      stdev(pairwise(extract_features.euclidean_distance(x_chest, y_chest, z_chest), extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),
                      feature_abs_integral(time, pairwise(extract_features.euclidean_distance(x_chest, y_chest, z_chest), extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),
                      extract_features.energy(pairwise(extract_features.euclidean_distance(x_chest, y_chest, z_chest), extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),

                      mean(pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_chest, y_chest, z_chest), axis3=extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),
                      stdev(pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_chest, y_chest, z_chest), axis3=extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),
                      feature_abs_integral(time, pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_chest, y_chest, z_chest), axis3=extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),
                      extract_features.energy(pairwise(extract_features.euclidean_distance(x_hand, y_hand, z_hand), extract_features.euclidean_distance(x_chest, y_chest, z_chest), axis3=extract_features.euclidean_distance(x_ankle, y_ankle, z_ankle))),

                      # Hearate
                      normalized_mean(heart_rate, max_heart_rate, min_heart_rate, subject),
                      gradient(heart_rate),

                      extract_features.activity_mode(activities)]

    writer.writerow(cur_features)


'''
****************************************************
**               PAMAP Features                   **
****************************************************
     * Mean
     * Median
     * Standard Deviation
     * Peak Acceleration
     * Energy
     * Feature Absolute Integral
     * Correlation Between Axes
     * Power Ratio of the Frequency Bands 0-2.75Hz and 0-5Hz
     * Peak Frequency of the normalized Power Spectral Density (PSD)
     * Spectral Entropy of the normalized PSD
     * Normalized Mean Heart Rate
     * Gradient Heart Rate
'''


def mean(vals):

    vals = np.asarray(vals)
    return np.mean(vals).item()


def median(vals):

    vals = np.asarray(vals)
    return np.median(vals).item()


def stdev(vals):

    vals = np.asarray(vals)
    return np.std(vals).item()


def peak_val(vals):
    return max(vals)


def feature_abs_integral(time, vals):

    total_signal_bp = bandpass(time, vals, 0.5, 11)  # Bandpass filter the signal (0.5 - 11Hz) to highlight activities caused by human movements

    return sum(integrate.cumtrapz(total_signal_bp, time))


def bandpass(time, vals, fL, fH):

    W = fftfreq(len(vals), d=time[1] - time[0])
    f_signal = rfft(vals)

    cut_f_signal = f_signal.copy()
    cut_f_signal[(W < fL)] = 0
    cut_f_signal[(W > fH)] = 0

    return irfft(cut_f_signal)


def correlation(axis1, axis2):

    # ratio of covariance and product of the standard deviations
    return np.cov(axis1, axis2)[0][1] / (stdev(axis1) * stdev(axis2))


def power_ratio(vals):

    return bandpower(vals, [0, 2.75]) / bandpower(vals, [0, 5])


def bandpower(vals, band):
    """Compute the average power of the signal in a specific frequency band.

    Args:
        vals (list): Input signal in the time-domain.
        band(list): Lower and upper frequencies of the band of interest.

    Return:
        bp (float): Absolute band power.
    """

    band = np.asarray(band)
    vals = np.asarray(vals)
    low, high = band

    # Compute the modified periodogram (Welch)
    freqs, psd = signal.welch(vals, 100)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = integrate.simps(psd[idx_band], dx=freq_res)

    return bp


def peak_psd(vals):

    freq, amp = signal.periodogram(vals, 100)
    return max(freq)


def entropy(vals):
    # spectral entropy

    vals = np.asarray(vals)

    _, psd = signal.periodogram(vals, 100)
    psd_norm = np.divide(psd, psd.sum())

    se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    se /= np.log2(psd_norm.size)

    return se


def normalize(vals, maximum, minimum, lower, upper):

    return [(x - minimum) / (maximum - minimum) for x in vals]


def normalized_mean(vals, max_value, min_value, subject):
    if subject == '101':
        return mean(normalize(vals, max_value, min_value, 75, 193))
    elif subject == '102':
        return mean(normalize(vals, max_value, min_value, 74, 195))
    elif subject == '103':
        return mean(normalize(vals, max_value, min_value, 68, 189))
    elif subject == '104':
        return mean(normalize(vals, max_value, min_value, 58, 196))
    elif subject == '105':
        return mean(normalize(vals, max_value, min_value, 70, 194))
    elif subject == '106':
        return mean(normalize(vals, max_value, min_value, 60, 194))
    elif subject == '107':
        return mean(normalize(vals, max_value, min_value, 60, 197))
    elif subject == '108':
        return mean(normalize(vals, max_value, min_value, 66, 188))
    elif subject == '109':
        return mean(normalize(vals, max_value, min_value, 54, 189))


def gradient(vals):

    return mean(np.gradient(np.asarray(vals)).tolist())


def pairwise(axis1, axis2, **kwargs):

    axis1 = np.asarray(axis1)
    axis2 = np.asarray(axis2)

    axis_sum = axis1 + axis2

    if 'axis3' in kwargs:
        axis3 = np.asarray(kwargs.get('axis3', None))
        axis_sum += axis3

    return axis_sum.tolist()
