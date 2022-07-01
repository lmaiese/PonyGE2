import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.test.my_test import variables_substitution, get_function, \
    terminator, get_dataframe, test_function, open_last_gen, test_file
from src.test.visualize import no_plot_clarke


def list_elements_from_folder(path):
    list = []
    dir = os.listdir(path)
    for elem in dir:
        list.append(open_last_gen(elem))
    print(list)
    return list


def plot_results(guess, ground):
    if len(guess) != len(ground):
        print("DIFFERENT LENGTH, break")
        return None
    # define data
    df = pd.DataFrame({'time_slice': np.array([i for i in range(len(guess))]),
                       'values': guess})

    df2 = pd.DataFrame({'time_slice': np.array([i for i in range(len(ground))]),
                        'values': ground})

    # plot both time series
    plt.plot(df.time_slice, df.values, label='predicted', linewidth=3)
    plt.plot(df2.time_slice, df2.values, color='red', label='real', linewidth=3)

    # add title and axis labels
    plt.title('Predicted Glucose vs Real Glucose')
    plt.xlabel('Time Slice')
    plt.ylabel('Glucose (md/dL)')

    # add legend
    plt.legend()

    # display plot
    plt.show()


def calculate_mean_and_variance(rmses, maes, zones):
    pass


if __name__ == '__main__':
    folder = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\final_run"
    best_run = "best_run"
    final_elements = list_elements_from_folder(folder)
    rmses = []
    maes = []
    guesses = []
    desireds = []
    nes = []
    functions = []
    zones = []

    for elem in final_elements:
        try:
            filepath = open_last_gen(elem)
            ne, guess, desired, function = test_function(test_file, filepath)
            zones.append(no_plot_clarke(desired, guess))
            rmses.append(mean_squared_error(desired, guess, squared=False))
            maes.append(mean_absolute_error(desired, guess))
            nes.append(ne)
            guesses.append(guess)
            desireds.append(desired)
            functions.append(function)
        except:
            print("Error while evaluating the current function. Sorry!\n\n")

    calculate_mean_and_variance(rmses, maes, zones)

    try:
        filepath = open_last_gen(best_run)
        ne, guess, desired, function = test_function(test_file, filepath)
        plot_results(desired, guess)

    except:
        pass
