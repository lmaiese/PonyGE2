import numpy as np
from fitness.base_ff_classes.base_ff import base_ff
import os
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from test.visualize import no_plot_clarke

individuals_number = 0
data_dir = "datasets/Glucose"
training_ext = "ws-training"
testing_ext = "ws-testing"
train_file = "datasets\\Glucose\\540\\540-ws-training.csv"
train_absolute = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose\\540\\540-ws-training.csv"
train_absolute_2nd = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose\\552\\552-ws-training.csv"


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


def calculate_fitness(ground_truth, guesses):
    zone = no_plot_clarke(ground_truth, guesses)
    penality = calculate_weighted_penality(zone)
    return penality


def calculate_weighted_penality(zone):
    total = 0
    penality = 0
    for x in zone:
        total += x
    b = (zone[1] / total) * 100
    c = (zone[2] / total) * 100
    d = (zone[3] / total) * 100
    e = (zone[4] / total) * 100

    penality += d * 1
    penality += e * 0.8
    penality += c * 0.6
    penality += b * 0.2

    normalization = penality / 10
    return normalization


class penality(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.sample = 0
        self.exceptions_count_ind = 0
        self.exceptions_count_rmse = 0
        self.default_fitness = np.NaN

    def evaluate(self, ind, **kwargs):
        p = ind.phenotype
        self.sample += + 1
        file = pd.read_csv(train_absolute_2nd, skiprows=1, header=None)
        ground_truth = []
        guesses = []
        for x in range(file.shape[0]):
            tuple = file.loc[x].tolist()
            function = variables_substitution(p, tuple)
            if self.sample % 10 == 0 and x == 1000:
                print("SONO VIVO, proseguo")
            try:
                guesses.append(eval(function) * 18)
                ground_truth.append(tuple[-1] * 18)
            except:
                return self.default_fitness
        try:
            return calculate_fitness(ground_truth, guesses)
        except:
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


def mane():
    pass


if __name__ == '__main__':
    string = "(math.exp ((-1) * (X7 / (-1) * ((-1) * (X22 ) ) + X34 - math.exp ((-1) * (X25 * math.sin (X25 ) ) - math.sin ((-1) * (math.log ((-1) * ((-1) * (math.sin (math.sin (math.sin (math.log (math.sin (math.log (math.cos ((-1) * ((-1) * ((-1) * (X31 ) ) ) ) ) ) ) ) - X19 * X31 * (-1) * (math.cos (math.exp (X16 ) ) ) / X25 / X13 ) + X4 ) ) ) ) ) ) + (-1) * (math.cos ((-1) * ((-1) * (math.log (math.sin ((-1) * ((-1) * (math.cos (math.cos (X34 ) * math.cos ((-1) * (math.log ((-1) * (X13 ) ) ) ) ) ) ) ) ) ) + X16 * X28 ) ) ) + math.exp ((-1) * (X31 ) ) ) ) / math.sin (X22 ) ) + math.sin (math.log (math.exp (math.cos (math.sin (X38 ) ) - X8 ) - math.cos (math.cos (math.cos (X17 * X38 + math.sin (math.log (X17 ) ) + math.cos (math.exp (X14 + X11 + math.exp (math.cos (math.sin (math.log (math.sin (math.sin (math.cos (X38 ) ) ) + X29 ) ) ) ) ) ) / math.cos (math.exp (X29 / math.sin (X5 ) + X2 / math.log (X17 ) * math.exp (X26 ) ) ) * math.log (math.exp (X32 ) ) - X35 / math.exp (X20 ) / X32 + X26 + X11 * math.exp (math.log (X38 ) ) / X20 * X26 + math.cos (math.sin (X23 ) ) + X11 ) + math.cos (X32 ) ) ) ) ) *X23 - math.log (X23 ) - math.exp (X3 ) - math.cos (math.sin (math.sin (X36 ) - math.log (X27 * math.sin (math.sin (math.log (X33 - X6 ) ) / math.sin (math.exp (X12 + X27 * math.cos (math.cos (X33 + math.exp (math.cos (math.cos (X21 ) ) ) + X12 / X39 + math.cos (math.sin (math.exp (math.log (math.cos (X6 ) ) - X18 * X33 * math.cos (X3 / math.exp (math.exp (math.sin (X12 - X27 ) ) + X15 - X33 ) / X6 ) + math.log (math.sin (math.cos (X39 - math.exp (math.exp (math.sin (math.exp (math.cos (X33 ) ) + X30 - math.cos (X9 ) ) ) / X36 ) + X15 / X27 - X12 * math.sin (math.exp (math.exp (math.sin (math.cos (X39 + math.sin (X30 ) / X30 - X27 / math.exp (math.log (X36 ) ) + math.sin (math.sin (X21 ) ) ) ) ) ) ) - X21 ) ) ) ) ) ) / math.sin (math.cos (math.log (X6 ) - X15 ) ) * math.log (math.sin (X24 ) ) + math.log (math.sin (math.log (math.sin (X30 ) + math.log (X3 + math.exp (math.log (math.exp (X3 ) + math.sin (math.exp (math.log (X33 ) ) / X24 * math.cos (X12 ) - math.exp (math.cos (X6 / math.log (math.sin (math.sin (X27 - X3 - X33 - math.log (math.cos (X9 + math.cos (math.cos (math.sin (X9 / math.log(math.log (X24 ) - X9 ) + math.sin (X6 * math.exp (X33 ) ) ) ) ) ) ) + math.cos (math.sin (X30 ) + math.exp (math.log (X24 - math.cos (X12 ) * X21 ) ) * math.log (math.cos (X9 + math.cos (X39 ) ) ) / X12 / math.log (math.exp (X36 ) ) / math.exp (X9 + math.log (math.log (math.cos (X3 ) / X18 + X36 ) - X6 ) * math.log (X33 ) * math.log (X39 ) * X36 * math.cos (math.cos(X9 ) + math.cos (math.log (X33 ) ) * math.exp (math.cos (X9 ) ) - X33 / X9 ) * X3 ) - X30 - math.exp (X36 / math.sin (math.log (X39 ) ) / X3 + X12 - math.exp (X15 ) ) * X15 * X9 + X21) / math.cos (math.exp (math.log (X18 * math.cos (math.sin (X30 * X27 - X21 ) ) + X36 ) ) ) - X3 / X18 * math.exp (X33 ) * X30 / math.sin (math.exp (math.log (math.sin (math.sin (X39 )) ) ) - X12 ) / math.log (math.exp (X30 ) ) ) * X24 ) ) + math.cos (X18 ) ) * X36 * math.cos (X9 - math.log (X6 ) ) - X15 * math.log (math.log (X27 * math.cos (X21 - X30 / math.cos (X15 ) ) * X33 ) - math.cos (math.sin (X18 * X33 + math.cos (math.sin (math.sin (math.log (math.cos (X6 * math.exp (math.log (math.exp (X27 ) ) * X21 - X3 / X24 * math.exp (X27 ) ) ) - X12/ X36 * math.sin (math.exp (math.exp (math.log (X36 ) ) ) ) + X6 - math.log (X36 ) + math.sin (math.log (X18 ) ) ) ) + math.cos (math.cos (math.log (X3 + math.cos (X12 ) * math.exp (math.log (X12 ) ) - X30 * X36 / math.cos (X18 ) ) ) ) * X27 ) ) ) - X27 / X27 ) / X3 / X18 ) - math.cos (math.exp (X21 ) / math.log (X24 * X12 / X12 * math.exp (math.exp (X9 ) ) + X24 / math.log (X6 ) * X3 + X3 / math.sin (X27 - math.cos (math.sin (X15 ) ) - X18 * X36 + math.exp (X12 - X15 / X21 * math.log (X18 ) ) - math.log (math.sin (math.cos (X18 ) ) ) - math.sin (X9 ) / X9 - X9 ) + X15 + X6 - X39 + math.exp (math.sin (math.exp (X21 - X15 - X6 ) ) ) - math.log (X36 * math.log (X30 ) ) * math.sin (X3 ) ) / X30 ) * X21 + X24 + X36 ) + X15 - math.sin(X6 ) - X12 ) * math.sin (X15 - math.sin (X27 ) ) / math.exp (X33 ) / X21 + X9 / X21 / X6 + X39 ) - math.cos (math.log (X15 ) ) + X30 + X39 + math.sin (math.cos (math.log (X24 ) ) / X3) / X18 - X33 / math.cos (X30 ) / X24 ) / X39 + math.cos (math.exp (X3 ) ) + X15 - X39 - math.exp (X3 ) ) * math.cos (X15 / math.log (math.cos (X39 ) ) - X12 - math.exp (math.cos (X21 ) / X9 / X33 * X12 + math.log (X21 ) + X12 ) ) ) ) + X9 ) - math.sin (X18 ) / math.cos (math.sin (X21 ) ) * math.log (X9 ) * X36 ) ) + math.sin (math.sin (math.sin (X9 ) ) ) ) * X39 ) )/ math.cos (X9 ) ) - math.log (X3 ) - X15 / math.sin (X3 ) ) / math.cos (math.sin (math.cos (math.exp (X30 ) ) ) / X24 ) ) * math.exp (X30 - math.log (X27 / math.cos (X15 - X36 * math.exp (math.log (X9 ) ) ) ) ) - X12 * math.cos (math.cos (X18 ) ) + X39 / X18 + X9 - math.log (math.cos (X3 ) ) - math.sin (X39 ) - math.cos (X9 ) / math.cos (math.sin (math.sin (math.log(math.exp (X12 + X12 / X27 ) ) ) ) * X12 * math.cos (math.exp (X33 ) - math.exp (math.log (X3 ) + X21 / math.sin (math.exp (math.log (X6 - X6 / math.exp (X39 + X18 ) ) * math.log (X21 ) / math.log (X24 ) * X12 ) - math.cos (X6 / X6 ) ) + math.exp (X12 ) ) - math.sin (X24 ) - math.sin (math.cos (X27 ) ) / math.exp (math.cos (X6 ) ) / X15 + math.exp (X6 ) + math.cos (X9 ) / X12 - X30 / math.exp (X21 - X33 ) * math.log (math.cos (math.exp (math.log (X24 ) ) ) ) ) * math.exp (X15 ) ) / X27 / math.exp (X18 + X27 ) ) + 56*math.pow( 10, -1 ) - 33*math.pow( 10, +9 )"
    var = '6'
    import random

    tuple = []
    for i in range(0, 57):
        n = random.randint(1, 2)
        tuple.append(n)
    gg = variables_substitution(string, tuple)

    x = eval(gg)
    print(x)
