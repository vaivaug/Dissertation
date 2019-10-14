import csv
import re

column_names = []
rows = []
lung_cancer_number = 0
gender_cancer_specified = 0
number_of_cancer_males = 0
number_of_cancer_females = 0


def read_file(filedir):
    global column_names
    global number_of_cancer_females, number_of_cancer_males

    with open(filedir) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        # iterate through each row
        for row in csv_reader:
            if line_count == 0:
                print("column names: ", ",  ".join(row))
                column_names = row
                line_count += 1
            else:
                rows.append(row)
                print(row)
                if is_male(row):
                    number_of_cancer_males += 1
                else:
                    number_of_cancer_females += 1

                if line_count == 10:
                    break
                # check_lung_cancer(row)
                #print("*********************************************************")
                #print("PERSON NUMBER ", len(rows))
                #print("*********************************************************")
                #print("date: ", row[3], "  ", row[10])
                line_count += 1

        print(f'Processed {line_count} lines.')


def rows_equal_length():

    for i in range(0, len(rows)):
        # at least one row has different length
        if len(rows[i]) != len(column_names):
            return False

    # all rows have the same length
    return True


def check_lung_cancer(row):

    global lung_cancer_number
    global number_of_cancer_females, number_of_cancer_males

    if "lung cancer" in row[10]:
        lung_cancer_number += 1
        print("*********************************************************")
        print("PERSON NUMBER ", len(rows))
        print("*********************************************************")
        # print(row[10])
        if is_male(row):
            number_of_cancer_males += 1
        else:
            number_of_cancer_females += 1
    '''
    for i in range(0, len(rows)):
        if "cancer" in rows[i][10]:
            print("*********************************************************")
            print("PERSON NUMBER ", i)
            print("*********************************************************")
            print(rows[i][10])
    '''


def is_male(row):

    if "female" in row[10]:
        return False

    if "male" in row[10]:
        return True



    '''
    global gender_cancer_specified
    year_description = (" yo ", " y.o. ", "y. o. ", " yr ")
    regex = r"\b(?:{})\b".format("|".join(year_description))

    if " yo " or " y.o. " or "y. o. " or " yr " in row[10]:
        res = re.split(regex, row[10])
        print(len(res))
    '''



read_file('../NOTEEVENTS.csv')

print("number of columns: ", len(column_names))
print("number of rows: ", len(rows))
print("length of rows: ", len(rows[0]))

rows_equal_length = rows_equal_length()
print("All rows have equal length: ", rows_equal_length)
print("lung cancer number: ", lung_cancer_number)
print("from which y.o is specified in: ", gender_cancer_specified)
print("males: ", number_of_cancer_males)
print("females: ", number_of_cancer_females)

