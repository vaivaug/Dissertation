"""
Contains a function to split the data into train and test sets
"""


def get_train_test_datasets(notes_adm):
    """ :param notes_adm: pandas dataframe containing all the data
        :return: two pandas dataframes, train and test set

    Create train and test datasets use random_state to have the same test data during each run
    """
    notes_adm = notes_adm.reset_index(drop=True)

    # Keep 20% of the data as test data
    test = notes_adm.sample(frac=0.2, random_state=0)
    print('length of test data: ', len(test))

    # The other 80% of data is used for training
    train = notes_adm.drop(test.index)
    print('length of training data: ', len(train))

    return train, test

