"""
Contains a function to form balanced training set by sub-sampling negatives
"""
import pandas as pd


def get_sub_sampling_negatives_data(train):
    """ Downsample majority class i.e. negatives

    @param train: pandas dataframe storing the training data
    @return: train: pandas dataframe storing the training data containing all positive samples
        from initial train data plus the same amount of negative samples
    """

    # create two datasets containing only positive and only negative samples
    positive = train[train.OUTPUT == 1]
    negative = train[train.OUTPUT == 0]

    # change negatives to a downsampled set of negatives
    negative = negative.sample(n=len(positive), replace=False)

    # combine positives and negatives
    train = pd.concat([positive, negative])

    # shuffle the order of training samples
    train = train.sample(n=len(train)).reset_index(drop=True)

    return train


