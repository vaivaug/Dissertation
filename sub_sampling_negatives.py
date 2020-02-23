"""
Contains a function to form balanced training set by sub-sampling negatives
"""
import pandas as pd
from sklearn.utils import resample


def get_sub_sampling_negatives_data(train):
    """ :param train: pandas dataset of train data
        :return: pandas dataset of train data containing all positive samples
        from initial train data plus the same amount of negative samples

    Downsample majority class i.e. negatives
    """
    # create two datasets containing only positive and only negative samples
    positive = train[train.OUTPUT == 1]
    negative = train[train.OUTPUT == 0]

    # change negatives to a downsampled set of negatives
    negative = negative.sample(n=len(positive))

    # combine positives and negatives
    train = pd.concat([positive, negative])

    # shuffle the order of training samples
    train = train.sample(n=len(train)).reset_index(drop=True)

    return train


