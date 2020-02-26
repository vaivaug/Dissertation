"""
Contains a function to split the data into train and test sets
"""


def get_train_test_datasets(notes_adm):
    """ :param notes_adm: pandas dataframe containing all the data
        :return: two pandas dataframes, train and test set

    Create train and test datasets use random_state to have the same test data during each run
    """
    notes_adm = notes_adm.reset_index(drop=True)

    # Keep 30% of the data to form test and validation sets
    test_validation = notes_adm.sample(frac=0.3, random_state=0)
    print('length of test and validation data together: ', len(test_validation))

    # test_validation data is split into half for test and validation sets
    test = test_validation.sample(frac=0.5, random_state=0)
    validation = test_validation.drop(test.index)

    # The other 70% of data is used for training
    train = notes_adm.drop(test_validation.index)
    print('length of training data: ', len(train))

    return train, test, validation

