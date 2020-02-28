"""
Contains a function to form balanced training set by over-sampling positives
"""
import pandas as pd


def get_over_sampling_positives_data(train):
    """ :param train: pandas dataset of train data
        :return: pandas dataset of train data containing all negative samples
        from initial train data plus the same amount of positive samples some of which are repeated multiple times

    Oversample minority class i.e. positives
    """
    # create two datasets containing only positive and only negative samples
    positive = train[train.OUTPUT == 1]
    negative = train[train.OUTPUT == 0]

    # change positives to a oversampled set of positives
    positive = positive.sample(n=len(negative), replace=True)

    # combine positives and negatives
    train = pd.concat([positive, negative])

    # shuffle the order of training samples
    train = train.sample(n=len(train)).reset_index(drop=True)

    return train
