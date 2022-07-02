import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.test.my_test import test_function, open_last_gen, calculate_percentages
from src.test.visualize import no_plot_clarke, clarke_error_grid

def roll_vector(array, num):
    new = np.roll(array, num)
    return new


def list_elements_from_folder(path):
    list = []
    dir = os.listdir(path)
    for elem in dir:
        list.append(open_last_gen(path+"\\"+elem))
    return list

def classic_plot():
    # define data
    df = pd.DataFrame({'date': np.array([datetime.datetime(2020, 1, i + 1)
                                         for i in range(12)]),
                       'sales': [3, 4, 4, 7, 8, 9, 14, 17, 12, 8, 8, 13]})

    df2 = pd.DataFrame({'date': np.array([datetime.datetime(2020, 1, i + 1)
                                          for i in range(12)]),
                        'returns': [1, 1, 2, 3, 3, 3, 4, 3, 2, 3, 4, 7]})

    # plot both time series
    plt.plot(df.date, df.sales, label='sales', linewidth=3)
    plt.plot(df2.date, df2.returns, color='red', label='returns', linewidth=3)

    # add title and axis labels
    plt.title('Sales by Date')
    plt.xlabel('Date')
    plt.ylabel('Sales')

    # add legend
    plt.legend()

    # display plot
    plt.show()

def plot_results(guess, ground):
    if len(guess) != len(ground):
        print("DIFFERENT LENGTH, break")
        return None
    # define data
    df = pd.DataFrame({'time': np.array([i for i in range(len(guess))]),
                       'glucose': guess})

    df2 = pd.DataFrame({'time': np.array([i for i in range(len(ground))]),
                        'glucose': ground})

    # plot both time series
    plt.plot(df.time, df.glucose, color='yellow', label='predicted', linewidth=1)
    plt.plot(df2.time, df2.glucose, color='green', label='real', linewidth=1)

    # add title and axis labels
    plt.title('Predicted Glucose vs Real Glucose')
    plt.xlabel('Time')
    plt.ylabel('Glucose (md/dL)')

    # add legend
    plt.legend()

    # display plot
    plt.show()


def calculate_mean_and_variance(rmses, maes, zones):
    As = []
    Bs = []
    Cs = []
    Ds = []
    Es = []


    for zone in zones:
        percentage = calculate_percentages(zone)
        As.append(percentage[0])
        Bs.append(percentage[1])
        Cs.append(percentage[2])
        Ds.append(percentage[3])
        Es.append(percentage[4])

    rmse = np.mean(np.array(rmses))
    mae = np.mean(np.array(maes))
    A = np.mean(np.array(As))
    B = np.mean(np.array(Bs))
    C = np.mean(np.array(Cs))
    D = np.mean(np.array(Ds))
    E = np.mean(np.array(Es))

    print("MEANS")

    print("RMSE {}\nMAE {}\nZONES {} {} {} {} {}".format(round(rmse, 2), round(mae, 2), round(A, 2), round(B, 2),
                                                          round(C, 2), round(D, 2), round(E, 2)))

    rmse = np.var(np.array(rmses))
    mae = np.var(np.array(maes))
    A = np.var(np.array(As))
    B = np.var(np.array(Bs))
    C = np.var(np.array(Cs))
    D = np.var(np.array(Ds))
    E = np.var(np.array(Es))

    print("VARIANCES")

    print("RMSE {}\nMAE {}\nZONES {} {} {} {} {}".format(round(rmse, 2), round(mae, 2), round(A, 2), round(B, 2),
                                                          round(C, 2), round(D, 2), round(E, 2)))

    return

if __name__ == '__main__':
    test_file = "PonyGE2\\datasets\\Glucose\\596\\596-ws-testing.csv"
    folder = "PonyGE2\\results\\final_run"
    best_run = "PonyGE2\\results\\genetic_operators\\mutation_50\\w_eval_zone_mae"
    final_elements = list_elements_from_folder(folder)
    rmses = []
    maes = []
    guesses = []
    desireds = []
    nes = []
    functions = []
    zones = []
    percentages = []

    for elem in final_elements:
        try:
            roll_value = 13
            ne, guess, desired, function = test_function(test_file, elem)
            desired = roll_vector(desired,roll_value)

            zone = no_plot_clarke(desired, guess)
            zones.append(zone)

            percentage = calculate_percentages(zone)
            percentages.append(percentage)

            rmse = (mean_squared_error(desired, guess, squared=False))
            rmses.append(rmse)

            mae = mean_absolute_error(desired, guess)
            maes.append(mae)

            nes.append(ne)
            guesses.append(guess)
            desireds.append(desired)
            functions.append(function)
            try:
                print(
                    "Function: {} \nLast gen: {} RMSE: {} ({}) MAE: {} ({}) SUM: {} ({})\nZONES: {} PERCENTAGES: {}"
                    " A+B%: {} (A% {})% D+E%: {}%\n\n".format(
                        function, elem.split("\\")[-1].split(".")[0],
                        str(round(rmse, 2)), str(round((rmse / 18), 2)),
                        str(round(mae, 2)), str(round((mae / 18), 2)),
                        str(round((rmse + mae), 2)), str(round((rmse + mae) / 18, 2)),
                        str(zone),
                        str(percentage),
                        str(round(percentage[0] + percentage[1], 2)),
                        str(round(percentage[0], 2)),
                        str(round(percentage[3] + percentage[4], 2))
                    ))
            except:
                print("no print")
        except:
            print("Error while evaluating the current function. Sorry!\n\n")

    try:
        calculate_mean_and_variance(rmses, maes, zones)
    except:
        print("no mean and average")

    try:
        filepath = open_last_gen(best_run)
        ne, guess, desired, function = test_function(test_file, filepath)
        plot_results(desired, guess)
    except:
        print("no plot1")


    try:
        roll_value = 13
        filepath = open_last_gen(best_run)
        ne, guess, desired, function = test_function(test_file, filepath)
        desired = roll_vector(desired, roll_value)
        plot_results(desired, guess)
        plt, zone = clarke_error_grid(desired, guess, None)
        percentage = calculate_percentages(zone)
        rmse = (mean_squared_error(desired, guess, squared=False))
        mae = mean_absolute_error(desired, guess)

        try:
            print(
                "Function: {} \nLast gen: {} RMSE: {} ({}) MAE: {} ({}) SUM: {} ({})\nZONES: {} PERCENTAGES: {}"
                " A+B%: {} (A% {})% D+E%: {}%\n\n".format(
                    function, elem.split("\\")[-1].split(".")[0],
                    str(round(rmse, 2)), str(round((rmse / 18), 2)),
                    str(round(mae, 2)), str(round((mae / 18), 2)),
                    str(round((rmse + mae), 2)), str(round((rmse + mae) / 18, 2)),
                    str(zone),
                    str(percentage),
                    str(round(percentage[0] + percentage[1], 2)),
                    str(round(percentage[0], 2)),
                    str(round(percentage[3] + percentage[4], 2))
                ))
        except:
            print("no print")

    except:
        print("no plot2")

