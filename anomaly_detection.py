import numpy as np
from scipy import stats

import torch

def test(test_loader, encoder, decoder, critic_x):
    reconstruction_error = list()
    critic_score = list()
    y_true = list()

    for batch, sample in enumerate(test_loader):
        reconstructed_signal = decoder(encoder(sample['signal']))
        reconstructed_signal = torch.squeeze(reconstructed_signal)

        for i in range(0, 64):
            x_ = reconstructed_signal[i].detach().numpy()
            x = sample['signal'][i].numpy()
            y_true.append(int(sample['anomaly'][i].detach()))
            reconstruction_error.append(dtw_reconstruction_error(x, x_))
        critic_score.extend(torch.squeeze(critic_x(sample['signal'])).detach().numpy())

    reconstruction_error = stats.zscore(reconstruction_error)
    critic_score = stats.zscore(critic_score)
    anomaly_score = reconstruction_error * critic_score
    y_predict = detect_anomaly(anomaly_score)
    y_predict = prune_false_positive(y_predict, anomaly_score, change_threshold=0.1)
    find_scores(y_true, y_predict)

#Other error metrics - point wise difference, Area difference.
def dtw_reconstruction_error(x, x_):
    n, m = x.shape[0], x_.shape[0]
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - x_[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n][m]

def unroll_signal(x):
    x = np.array(x).reshape(100)
    return np.median(x)

def prune_false_positive(is_anomaly, anomaly_score, change_threshold):
    #The model might detect a high number of false positives.
    #In such a scenario, pruning of the false positive is suggested.
    #Method used is as described in the Section 5, part D Identifying Anomalous
    #Sequence, sub-part - Mitigating False positives
    #TODO code optimization
    seq_details = []
    delete_sequence = 0
    start_position = 0
    end_position = 0
    max_seq_element = anomaly_score[0]
    for i in range(1, len(is_anomaly)):
        if i+1 == len(is_anomaly):
            seq_details.append([start_position, i, max_seq_element, delete_sequence])
        elif is_anomaly[i] == 1 and is_anomaly[i+1] == 0:
            end_position = i
            seq_details.append([start_position, end_position, max_seq_element, delete_sequence])
        elif is_anomaly[i] == 1 and is_anomaly[i-1] == 0:
            start_position = i
            max_seq_element = anomaly_score[i]
        if is_anomaly[i] == 1 and is_anomaly[i-1] == 1 and anomaly_score[i] > max_seq_element:
            max_seq_element = anomaly_score[i]

    max_elements = list()
    for i in range(0, len(seq_details)):
        max_elements.append(seq_details[i][2])

    max_elements.sort(reverse=True)
    max_elements = np.array(max_elements)
    change_percent = abs(max_elements[1:] - max_elements[:-1]) / max_elements[1:]

    #Appending 0 for the 1 st element which is not change percent
    delete_seq = np.append(np.array([0]), change_percent < change_threshold)

    #Mapping max element and seq details
    for i, max_elt in enumerate(max_elements):
        for j in range(0, len(seq_details)):
            if seq_details[j][2] == max_elt:
                seq_details[j][3] = delete_seq[i]

    for seq in seq_details:
        if seq[3] == 1: #Delete sequence
            is_anomaly[seq[0]:seq[1]+1] = [0] * (seq[1] - seq[0] + 1)
 
    return is_anomaly

def detect_anomaly(anomaly_score):
    window_size = len(anomaly_score) // 3
    step_size = len(anomaly_score) // (3 * 10)

    is_anomaly = np.zeros(len(anomaly_score))

    for i in range(0, len(anomaly_score) - window_size, step_size):
        window_elts = anomaly_score[i:i+window_size]
        window_mean = np.mean(window_elts)
        window_std = np.std(window_mean)

        for j, elt in enumerate(window_elts):
            if (window_mean - 3 * window_std) < elt < (window_mean + 3 * window_std):
                is_anomaly[i + j] = 0
            else:
                is_anomaly[i + j] = 1

    return is_anomaly

def find_scores(y_true, y_predict):
    tp = tn = fp = fn = 0

    for i in range(0, len(y_true)):
        if y_true[i] == 1 and y_predict[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_predict[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_predict[i] == 0:
            tn += 1
        elif y_true[i] == 0 and y_predict[i] == 1:
            fp += 1

    print ('Accuracy {:.2f}'.format((tp + tn)/(len(y_true))))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print ('Precision {:.2f}'.format(precision))
    print ('Recall {:.2f}'.format(recall))
    print ('F1 Score {:.2f}'.format(2 * precision * recall / (precision + recall)))
