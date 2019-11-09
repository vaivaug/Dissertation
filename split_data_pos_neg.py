import pandas as pd
import csv


def initialise_file_columns(filename):
    with open(filename, 'a') as file:
        file_writer = csv.writer(file, delimiter=',')
        file_writer.writerow(['TEXT', 'VALUE'])


def append_file(filename, text, value):
    with open(filename, 'a') as file:
        file_writer = csv.writer(file, delimiter=',')
        file_writer.writerow([text, value])


def read_file_to_pandas(filename):
    return pd.read_csv(filename)

