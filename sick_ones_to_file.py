

def write_sick_ones_to_file(filedir, data):
    data = data.loc[data['OUTPUT'] == 1]
    data.to_csv(filedir)