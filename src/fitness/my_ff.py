from fitness.base_ff_classes.base_ff import base_ff
import time
import shutil
import os
import random
import pandas as pd
import pickle
import math
from sklearn.metrics import mean_squared_error

generation_range = 9
individuals_number = 0
graphFit = []
data_dir = "datasets/Glucose"
training_ext = "ws-training"
testing_ext = "ws-testing"
train_file = "datasets\\Glucose\\540\\540-ws-training.csv"
train_absolute = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose\\540\\540-ws-training.csv"


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


def addFitness(fitness, i):
    # graphFit.append(fitness)
    textfile = open("a_file.txt", "a")
    textfile.write(str(i) + ',' + str(fitness) + "\n")
    textfile.close()
    return


def compare_last_line(value):
    with open('a_file.txt') as f:
        for line in f:
            pass
        last_line = line
    y = float(last_line.split(',')[1])
    return value < y


class my_ff(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.sample = 0
        self.exceptions_count_ind = 0
        self.exceptions_count_rmse = 0

    def evaluate(self, ind, **kwargs):
        p = ind.phenotype
        self.sample += + 1
        # print("\n" + p)
        fitness = 0
        file = pd.read_csv(train_absolute, skiprows=1, header=None)
        ground_truth = []
        guesses = []
        times = []
        min = 100000
        for x in range(file.shape[0]):
            tuple = file.loc[x].tolist()
            function = variables_substitution(p, tuple)
            try:
                t0 = time.time()
                guesses.append(eval(function))
                t1 = time.time()
                ground_truth.append(tuple[-1])
                times.append(t1 - t0)
                if x % 10000 == 0 and x != 0:
                    print("\nSample {} - Iteration {}: Ok".format(self.sample, x))
            except:
                self.exceptions_count_ind += 1
                print("\nError with the individuals n° {}".format(self.exceptions_count_ind))
                print(p.ind)
                return self.default_fitness
        try:
            function_fitness = mean_squared_error(ground_truth, guesses, squared=False)
            if function_fitness < min and compare_last_line(function_fitness):
                min = function_fitness
                addFitness(min, 0)
            print(function_fitness)
            return function_fitness
        except:
            self.exceptions_count_rmse += 1
            print("\nError with the rmse n° {}".format(self.exceptions_count_rmse))
            return self.default_fitness


def open_train_file(directory, extension):
    # This function, using the extension, finds the train file, opens it using pandas e returns the dataframe
    file = None
    print(directory + "root directory")
    for root, dirs, files in os.walk(directory):
        print(root, dirs, files)
        for filename in files:
            if filename.find(extension) > 0:
                file = directory + "/" + filename
                print(file)
    if file is not None:
        df = pd.read_csv(file, skiprows=1, header=None)
        print(df)
        return df
    else:
        return None


def get_directories(src):
    # This function calculates the number of trials needed for the calculation
    directories = []
    for subdir, dirs, files in os.walk(src):
        directories.append(subdir)
    return directories


def get_normalized_lists(file):
    glucose = []
    insuline = []
    carbos = []
    for x in range(file.shape[0]):
        line = file.loc[x].tolist()
        variables = int(len(line) / 3)
        # print(line, line.__class__)
        g = []
        i = []
        c = []

        for k in range(variables):
            g.append(line[k * 3])
            i.append(line[k * 3 + 1])
            c.append(line[k * 3 + 2])

        glucose.append(g)
        insuline.append(i)
        carbos.append(c)
    return glucose, insuline, carbos


def mane():
    print("Tests for fitness function module")

    # copied_dir = copy_dir(data_dir)
    subdirs = get_directories(data_dir)
    subdirs.pop(0)
    print(subdirs)
    trials = len(subdirs)

    glucose = []
    insuline = []
    carbos = []

    for dir in subdirs:
        file = open_train_file(dir, extension=training_ext)
        glucose, insuline, carbos = get_normalized_lists(file)

        # qua si deve:
        # leggere il csv
        # prendere una riga per volta
        # trasformarla in qualche modo
        # passarla alla funzione eval
        # vedere se si trova il guess
        # etcc...

    # delete_dir(copied_dir)


if __name__ == '__main__':
    string = "X2 X3 X2 X3"
    var = '6'
    tuple = [0.1212313, 1.124155515, 2.2352363422]
    variables_substitution(string, tuple)

    x = eval("(10 + 32 - 36) + 94*math.pow(10, -1) -89*math.pow(10, -4)")
    print(x)
