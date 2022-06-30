import numpy as np
from fitness.base_ff_classes.base_ff import base_ff
import os
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

individuals_number = 0
data_dir = "datasets/Glucose"
training_ext = "ws-training"
testing_ext = "ws-testing"
train_file = "datasets\\Glucose\\540\\540-ws-training.csv"
train_absolute = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose\\540\\540-ws-training.csv"
train_absolute_2nd = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose\\596\\596-ws-training.csv"


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


def evaluate_zone(guess, ground):
    zone_a = (ground <= 70 and guess <= 70) or (1.2 * ground >= guess >= 0.8 * ground)

    zone_c = ((70 <= ground <= 290) and guess >= ground + 110) or (
            (130 <= ground <= 180) and (guess <= (7 / 5) * ground - 182))

    zone_e = (ground >= 180 and guess <= 70) or (ground <= 70 and guess >= 180)

    zone_d = (ground >= 240 and (70 <= guess <= 180)) or (
            ground <= 175 / 3 and 180 >= guess >= 70) or (
                     (175 / 3 <= ground <= 70) and guess >= (6 / 5) * ground)
    if zone_a:
        return 0
    elif zone_c:
        return 0.4
    elif zone_e:
        return 1
    elif zone_d:
        return 0.8
    else:
        return 0.2


def normalize_eval_zone(fitness, num):
    return fitness / num * 10


def calculate_final_fitness(eval_zone, ground_truth, guesses):
    rmse = mean_squared_error(ground_truth, guesses, squared=False)
    return rmse + eval_zone


class w_eval_zone_rmse(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.sample = 0
        self.default_fitness = np.NaN

    def evaluate(self, ind, **kwargs):
        p = ind.phenotype
        self.sample += + 1
        fitness = 0
        ground_truth = []
        guesses = []
        file = pd.read_csv(train_absolute_2nd, skiprows=1, header=None)
        for x in range(file.shape[0]):
            tuple = file.loc[x].tolist()
            function = variables_substitution(p, tuple)
            try:
                ground_truth.append(tuple[-1])
                guesses.append(eval(function))
                zone = evaluate_zone(guesses[-1], ground_truth[-1])
                fitness = fitness + zone
                if self.sample % 10 == 0 and x == 1000:
                    print("SONO VIVO, proseguo")
            except:
                print("SONO NELL'ECCEZIONE, male")
                return self.default_fitness
        try:
            return calculate_final_fitness(normalize_eval_zone(fitness, file.shape[0]), ground_truth, guesses)
        except:
            return self.default_fitness


def mane():
    return


if __name__ == '__main__':
    mane()
