"""
File used to store lung cancer patient notes in a separate file for analysis with the client
"""


def write_sick_ones_to_file(filedir, data):
    data = data.loc[data['OUTPUT'] == 1]
    data.to_csv(filedir)
