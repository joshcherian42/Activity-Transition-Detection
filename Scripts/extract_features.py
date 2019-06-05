import math
import itertools
import operator


def median(sorted_x):
    """Calculates the median.

    Given a sorted list of values, calculates the median of those values

    Args:
        sorted_x (list): Dictionary of dates in range of interest

    Returns:
        int: returns the median value in sorted_x
    """

    sorted_x.sort()
    if len(sorted_x) % 2 == 0:
        median = (sorted_x[len(sorted_x) / 2] + sorted_x[len(sorted_x) / 2 - 1]) / 2
    else:
        median = sorted_x[len(sorted_x) / 2]
    return median


def euclidean_distance(x, y, z):
    """Calculates the euclidean distance.

    Given a x, y, and z coordinates, returns the euclidean distances

    Args:
        x (list): list of x values
        y (list): list of y values
        z (list): list of z values

    Returns:
        list: returns the list of euclidean values
    """

    euclid = list()
    for index in range(0, len(x)):
        euclid.append(math.sqrt(math.pow(x[index], 2) + math.pow(y[index], 2) + math.pow(z[index], 2)))
    return euclid


def parse_data(window_size, time, x, y, z, start, end, activity, filename, cur_file, writer):
    """Calculates features and writes them to a file.

    Given a window of data, calculates features and writes them to the features file

    Args:
        window_size (int): Size of the window
        time (list): list of time values within the window
        x (list): list of x values within the window
        y (list): list of y values within the window
        z (list): list of z values within the window
        start (int): index of start of window
        end (int): index of end of window
        activity (list): list of activity values within the window
        filename (str): name of file to write features to
        cur_file (str): name of file containing the data from which features are being extracted
    """
    euclid = euclidean_distance(x, y, z)
    heights_e = side_height(euclid, time)

    e_peaks = peaks(euclid)

    e_valleys = valleys(euclid)

    stdev_peaks_e = 0.0

    sorted_e = list(euclid)

    if not e_peaks:
        avg_peaks_e = median(sorted_e)
    else:
        avg_peaks_e = average(e_peaks)
        stdev_peaks_e = stdev(e_peaks)

    stdev_valleys_e = 0.0

    if not e_valleys:
        avg_valleys_e = median(euclid)
    else:
        avg_valleys_e = average(e_valleys)
        stdev_valleys_e = stdev(e_valleys)

    avg_height_e = 0.0
    stdev_heights_e = 0.0

    if heights_e:
        stdev_heights_e = stdev(heights_e)
        avg_height_e = average(heights_e)
    axis_overlap = axis_order(x, y) + axis_order(y, z) + axis_order(x, z)

    # Calculate features based on csv_lines
    if activity.size != 0:  # Training Data
        cur_activity = activity_mode(activity)
        if (cur_activity != ''):

            cur_features_all = [avg_jerk(euclid, time),
                                avg_height_e,
                                stdev_heights_e,
                                energy(euclid),
                                entropy(euclid),
                                average(euclid),
                                stdev(euclid),
                                rms(euclid),
                                len(e_peaks),
                                avg_peaks_e,
                                stdev_peaks_e,
                                len(e_valleys),
                                avg_valleys_e,
                                stdev_valleys_e,
                                axis_overlap]
            cur_features_all.extend([activity_mode(activity), start, end])

            writer.writerow(cur_features_all)

    else:  # Testing Data

        cur_features = [avg_jerk(euclid, time),
                        avg_height_e,
                        stdev_heights_e,
                        energy(euclid),
                        entropy(euclid),
                        average(euclid),
                        stdev(euclid),
                        rms(euclid),
                        len(e_peaks),
                        avg_peaks_e,
                        stdev_peaks_e,
                        len(e_valleys),
                        avg_valleys_e,
                        stdev_valleys_e,
                        axis_overlap,
                        time[0], time[len(time) - 1]]

        writer.writerow(cur_features)


'''
****************************************************
**                     Features                   **
****************************************************
     * Most Common Activity
     * Average Jerk
     * Average Distance Between Axes
     * Axis Order
     * Energy
     * Entropy
     * Mean of height of sides
     * StdDev of height of sides
     * Mean of distance from peak/valley to mean
     * StdDev of distance from peak/valley to mean
     * Average
     * Standard Deviation
     * Correlation
     * RMS
     * Peaks
     * Valleys
     * Zero Crossings
'''


def most_common(L):
    """Finds the most common value in a list.

    Finds the most common value in a list L

    Args:
        L (list): a list of values

    Returns:
        str: returns the most common value in L
    """

    SL = sorted((x, i) for i, x in enumerate(L))
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        return count, -min_index
    return max(groups, key=_auxfun)[0]


def fast_slow(activities):
    """Returns fast or slow depending on the most common activity

    Finds the most common activity, and returns fast or slow depending on the activity
    Fast: Driving, Biking, Transit
    Slow: Putting on Table, Sitting, Standing, Strolling, Walking, Jogging, Running, Biking

    Args:
        activities (list): list of activities

    Returns:
        str: returns Fast or Slow based on the activity
    """

    act_types = {}
    for activity in activities:
        if activity in act_types:
            act_types[activity] = act_types[activity] + 1
        else:
            act_types[activity] = 1

    activity = str(max(act_types, key=act_types.get)).lower().strip()
    if (activity == "putting on a table"):
        activity = "putting on table"
    if activity == 'driving' or activity == 'biking' or activity == 'transit':
        return "Fast"
    else:
        return "Slow"


def activity_mode(activities):
    """Finds the most common activity

    Given a list of activities, finds the most common activity

    Args:
        activities (list): list of activities

    Returns:
        str: returns the most common activity
    """
    act_types = {}
    for activity in activities:
        if activity in act_types:
            act_types[activity] = act_types[activity] + 1
        else:
            act_types[activity] = 1

    activity = str(max(act_types, key=act_types.get)).lower().strip()
    if (activity == "putting on a table"):
        activity = "putting on table"
    return activity


def avg_jerk(x, time):
    """Returns average jerk of the window

    Calcualtes the average jerk of a window of accelerometer data

    Args:
        x (list): a list of accelerometer values in the window
        time (list): list of time values in the window

    Returns:
        double: returns the average jerk of the window of data
    """

    jerk = 0.0
    for index in range(1, len(x)):
        jerk += (x[index] - x[index - 1]) / (time[index] - time[index - 1])

    # if len(x) == 1:
    return jerk
    # else:
    #    return round(jerk/(len(x)-1))


def avg_diff(x, y):
    """Returns the average distance between each value

    Calcualtes the average distance between values in two lists of accelerometer data

    Args:
        x (list): a list of accelerometer values in the window
        y (list): another list of accelerometer values in the window

    Returns:
        double: returns the average distance between values in two lists of accelerometer data
    """

    diff = 0.0

    for index in range(1, len(x)):
        diff += x[index] - y[index]

    return diff / len(x)


def axis_order(x, y):
    """Returns the number of times two axes cross over

    Calcualtes the number of times two axes cross over

    Args:
        x (list): a list of accelerometer values in the window
        y (list): another list of accelerometer values in the window

    Returns:
        int: returns the number of times two axes cross over
    """

    changes = 0
    xgreatery = None

    for cnt in range(len(x)):

        if (cnt == 0):
            if x[cnt] > y[cnt]:
                xgreatery = True
            elif x[cnt] < y[cnt]:
                xgreatery = None
        else:
            if x[cnt] > y[cnt]:
                if not xgreatery:
                    changes += 1
            elif x[cnt] < y[cnt]:
                if xgreatery:
                    changes += 1

    return changes


def energy(x):
    """Calcultes the energy

    Calcualtes the energy of the window of accelerometer data

    Args:
        x (list): a list of accelerometer values in the window

    Returns:
        int: returns the energy
    """

    energy = 0

    for k in range(len(x)):

        ak = 0.0
        bk = 0.0
        for i in range(len(x)):
            angle = 2 * math.pi * i * k / len(x)
            ak += x[i] * math.cos(angle)
            bk += -x[i] * math.sin(angle)

        energy += (math.pow(ak, 2) + math.pow(bk, 2)) / len(x)

    return energy


def entropy(x):
    """Calcultes the entropy

    Calcualtes the entropy of the window of accelerometer data

    Args:
        x (list): a list of accelerometer values in the window

    Returns:
        int: returns the entropy
    """

    spectralentropy = 0.0
    for j in range(len(x)):
        ak = 0.0
        bk = 0.0
        aj = 0.0
        bj = 0.0
        mag_j = 0.0
        mag_k = 0.0
        cj = 0.0

        for i in range(len(x)):
            angle = 2 * math.pi * i * j / len(x)
            ak = x[i] * math.cos(angle)   # Real
            bk = -x[i] * math.sin(angle)  # Imaginary
            aj += ak
            bj += bk

            mag_k += math.sqrt(math.pow(ak, 2) + math.pow(bk, 2))

        mag_j = math.sqrt(math.pow(aj, 2) + math.pow(bj, 2))

        if mag_k != 0 and mag_j != 0:
            cj = mag_j / mag_k
            spectralentropy += cj * math.log(cj) / math.log(2)
    return -spectralentropy


def side_height(x, time):
    """Calcultes the side height

    Calcualtes the height of the sides of the data, i.e. the distances from peak to valley and valley to peak

    Args:
        x (list): a list of accelerometer values in the window

    Returns:
        list: returns a list of side heights
    """

    heights = []

    q1_check = None  # true greater than, false less than
    q3_check = None
    moved_to_middle = None
    cur_q1_points = []
    cur_q3_points = []
    peaks_valleys = []

    sorted_x = list(x)
    cur_median = median(sorted_x)
    q1 = min(x) + abs((cur_median - min(x)) / 2)
    q3 = cur_median + abs((max(x) - cur_median) / 2)

    cur_x = 0.0
    for i in range(len(x)):
        cur_x = x[i]
        if i == 0:
            if cur_x > q3:
                cur_q3_points.append(cur_x)
                q1_check = True
                q3_check = True
            elif cur_x > q1:
                q1_check = True
            else:
                cur_q1_points.append(cur_x)
        else:
            if cur_x > q3:
                q3_check = True
                q1_check = True
                if moved_to_middle:
                    if cur_q1_points:
                        peaks_valleys.append(min(cur_q1_points))  # add valley
                    del cur_q1_points[:]
                    moved_to_middle = None
                cur_q3_points.append(cur_x)
            elif cur_x > q1:
                if (q3_check and q1_check) or (not q3_check and not q1_check):
                    moved_to_middle = True

                q1_check = True
                q3_check = None
            else:
                if moved_to_middle:
                    if cur_q3_points:
                        peaks_valleys.append(max(cur_q3_points))  # add peak

                    del cur_q3_points[:]
                    moved_to_middle = None

                cur_q1_points.append(cur_x)
                q1_check = None
                q3_check = None

    for i in range(len(peaks_valleys) - 1):
        heights.append(abs(peaks_valleys[i + 1] - peaks_valleys[i]))

    return heights


def average(x):
    """Calcultes the average of a list

    Calcualtes the average value of a list of accelerometer data

    Args:
        x (list): a list of accelerometer values in the window

    Returns:
        double: returns the average
    """

    avg = 0.0
    for cnt in x:
        try:
            cnt = float(cnt)
        except ValueError:
            print cnt
        avg += cnt
    return avg / len(x)


def stdev(x):
    """Calcultes the standard deviation of a list

    Calcualtes the standard deviation value of a list of accelerometer data

    Args:
        x (list): a list of accelerometer values in the window

    Returns:
        double: returns the standard deviation
    """

    avg = average(x)
    std = 0.0
    for cur_x in x:
        std += math.pow((cur_x - avg), 2)
    return math.sqrt(std / len(x))


def sig_corr(x, y):
    """Calcultes the correlation of a list

    Calcualtes the correlation of a list of accelerometer data

    Args:
        x (list): a list of accelerometer values in the window
        y (list): a list of accelerometer values in the window

    Returns:
        double: returns the correlation
    """

    correlation = 0.0

    for cnt in range(len(x)):
        correlation += x[cnt] * y[cnt]

    return correlation / len(x)


def rms(x):
    """Calcultes the RMS of a list

    Calcualtes the root mean square (RMS) of a list of accelerometer data

    Args:
        x (list): a list of accelerometer values in the window

    Returns:
        double: returns the RMS
    """

    avg = 0.0

    for cnt in x:
        avg += math.pow(cnt, 2)

    return math.sqrt(avg / len(x))


def peaks(x):
    """Finds the peaks

    Finds the peaks within a window of accelerometer data

    Args:
        x (list): a list of accelerometer values in the window

    Returns:
        list: returns the peaks
    """

    peaks = []

    q1_check = None  # true greater than, false less than
    q3_check = None
    moved_to_middle = None
    cur_q3_points = []

    sorted_x = list(x)
    cur_median = median(sorted_x)
    q1 = min(x) + abs((cur_median - min(x)) / 2)
    q3 = cur_median + abs((max(x) - cur_median) / 2)

    cur_x = 0.0
    for i, cur_x in enumerate(x):
        if i == 0:
            if cur_x > q3:
                cur_q3_points.append(cur_x)
                q1_check = True
                q3_check = True
            elif cur_x > q1:
                q1_check = True
        else:
            if cur_x > q3:
                q3_check = True
                q1_check = True
                if moved_to_middle:
                    moved_to_middle = None
                cur_q3_points.append(cur_x)
            elif cur_x > q1:
                if (q3_check and q1_check) or (not q3_check and not q1_check):
                    moved_to_middle = True

                q1_check = True
                q3_check = None
            else:
                if moved_to_middle:
                    if cur_q3_points:
                        peaks.append(max(cur_q3_points))  # add peak

                    del cur_q3_points[:]
                    moved_to_middle = None

                q1_check = None
                q3_check = None

    return peaks


def valleys(x):
    """Finds the valleys

    Finds the valleys within a window of accelerometer data

    Args:
        x (list): a list of accelerometer values in the window

    Returns:
        list: returns the valleys
    """

    valleys = []

    q1_check = None  # true greater than, false less than
    q3_check = None
    moved_to_middle = None
    cur_q1_points = []

    sorted_x = list(x)
    cur_median = median(sorted_x)
    q1 = min(x) + abs((cur_median - min(x)) / 2)
    q3 = cur_median + abs((max(x) - cur_median) / 2)

    cur_x = 0.0

    for i, cur_x in enumerate(x):
        if i == 0:
            if cur_x > q3:
                q1_check = True
                q3_check = True
            elif cur_x > q1:
                q1_check = True
            else:
                cur_q1_points.append(cur_x)

        else:
            if cur_x > q3:
                q3_check = True
                q1_check = True
                if moved_to_middle:
                    if cur_q1_points:
                        valleys.append(min(cur_q1_points))  # add valley

                    del cur_q1_points[:]
                    moved_to_middle = None

            elif cur_x > q1:
                if (q3_check and q1_check) or (not q3_check and not q1_check):
                    moved_to_middle = True

                q1_check = True
                q3_check = None
            else:
                if moved_to_middle:
                    moved_to_middle = None

                cur_q1_points.append(cur_x)
                q1_check = None
                q3_check = None

    return valleys
