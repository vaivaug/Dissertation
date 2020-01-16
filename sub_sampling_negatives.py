import pandas as pd
from sklearn.utils import resample


def get_sub_sampling_negatives_data(train):
    # downsample majority i.e. 0
    positive = train[train.OUTPUT == 1]
    negative = train[train.OUTPUT == 0]

    train = resample(negative,
                     replace=False,  # sample without replacement
                     n_samples=len(positive)  # match minority n
                     )
    # reproducible results

    # combine minority and downsampled majority
    train = pd.concat([train, positive])
    print(train.OUTPUT.value_counts())
    return train



'''
    # OLD CODE
    # split the training data into positive and negative
    positive_data = train.OUTPUT == 1
    positive = train.loc[positive_data]
    negative = train.loc[~positive_data]

    # merge the balanced data
    train = pd.concat([positive, negative.sample(n=len(positive))], axis=0)
    # shuffle the order of training samples
    train = train.sample(n=len(train)).reset_index(drop=True)
'''

