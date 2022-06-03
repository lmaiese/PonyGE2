import pandas as pd
import math

from src.test.visualize import clarke_error_grid

result_file1 = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\1-06\\EXP1\\" \
               "LAPTOP-QPRKET60_22_6_1_175902_248840_21316_248840\\best.txt"

result_file2 = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\3-06\\EXP1\\" \
               "LAPTOP-QPRKET60_22_6_3_173553_773404_38744_773404\\6.txt"

test_file = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose\\540\\540-ws-testing.csv"
terminator = "Phenotype:" + "\n"


def variables_substitution(p, tuple):
    var = 'X'
    while True:
        index = p.find(var)
        if index == -1:
            break
        s = p[index:index + 2]
        i = int(s[1:]) - 1
        sub = str(tuple[i])
        p = p.replace(s, sub)
    return p


def get_function(result):
    file = open(result)
    line = file.readline()
    while line != terminator:
        line = file.readline()
    line = file.readline()
    return line


def get_dataframe(test):
    df = pd.read_csv(test, skiprows=1, header=None)
    return df


def test_function(test, result):
    tf = get_function(result)
    df = get_dataframe(test)
    number_of_elements = df.shape[0]
    good_extimations = 0
    bad_extimations = 0
    guesses = []
    desired = []
    for x in range(number_of_elements):
        tuple = df.loc[x].tolist()
        func = variables_substitution(tf, tuple)
        y = tuple[-1]
        z = eval(func)
        desired.append(y * 18)
        guesses.append(z * 18)
        # print(y, z)
        if y == z:
            good_extimations += 1
        else:
            bad_extimations += 1
    print(tf)
    return good_extimations, bad_extimations, number_of_elements, guesses, desired


if __name__ == '__main__':
    ge, be, ne, guesses, desired = test_function(test_file, result_file2)
    clarke_error_grid(desired, guesses, "")
