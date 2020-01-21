import pandas as pd
from sklearn.utils import resample


def get_over_sampling_positives_data(train):

    positive = train[train.OUTPUT == 1]
    negative = train[train.OUTPUT == 0]

    # upsampling positives
    train = resample(positive,
                     replace=True, # sample with replacement
                     n_samples=len(negative)  # match majority class (negatives)
                     )

    # combine minority and downsampled majority
    train = pd.concat([negative, train])
    print(train.OUTPUT.value_counts())

    return train
