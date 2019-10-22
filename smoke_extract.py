import re


# return true if a word starting with
def find_smoke(text):
    result_list = re.findall(r'smoke|smokes|smoked|smoking|smoker|tobacco', text)

    print(result_list)
    if len(result_list) != 0:
        return True
    return False

