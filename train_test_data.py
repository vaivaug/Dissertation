

def get_train_test_datasets(notes_adm):

    # create train and test datasets
    notes_adm = notes_adm.sample(n=len(notes_adm))
    notes_adm = notes_adm.reset_index(drop=True)

    # Keep 30% of the data as test data
    '''TODO: experiment with less than 30% and maybe more than 30%'''
    test = notes_adm.sample(frac=0.9) # change to 0.2
    print('length of test data: ', len(test))

    # The other 70% of data is used for training
    train = notes_adm.drop(test.index)
    print('length of training data: ', len(train))

    return train, test

