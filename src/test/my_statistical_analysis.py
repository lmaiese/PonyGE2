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
        list.append(open_last_gen(path + "\\" + elem))
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


def plot_results(guess, ground, name):
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
    plt.title(name + ' Predicted Glucose vs Real Glucose')
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

    print("SORTED PERCENTAGES")

    sA = np.sort(As)
    sB = np.sort(Bs)
    sC = np.sort(Cs)
    sD = np.sort(Ds)
    sE = np.sort(Es)

    print("AS")
    print(sA)
    print("BS")
    print(sB)
    print("CS")
    print(sC)
    print("DS")
    print(sD)
    print("ES")
    print(sE)

    return


def list_all_test_files(dataset_folder):
    list = []
    dir = os.listdir(dataset_folder)
    for elem in dir:
        list.append(dataset_folder + "\\" + elem + "\\" + elem + "-ws-testing.csv")
    return list


if __name__ == '__main__':
    dataset_folder = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose"
    folder = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\final_run"
    best_run = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\genetic_operators\\mutation_50\\w_eval_zone_mae"
    test_file = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose\\596\\596-ws-testing.csv"
    worst_run = ""

    best_run_file = ""
    worst_run_file = ""

    test_files = list_all_test_files(dataset_folder)
    final_elements = list_elements_from_folder(folder)
    final_elements.append(open_last_gen(best_run))

    best_sum = 100
    worst_sum = 0

    try:
        print("\nSTEP 3 RUN IS: {}\nOVER: {}".format(best_run, test_file))
        ne, guess, desired, function = test_function(test_file, open_last_gen(best_run))
        plot_results(guess, desired, "STEP 3 RUN")
    except:
        print("no plot step 3 ")

    try:
        plt, zone = clarke_error_grid(desired, guess, "STEP 3 RUN")
        percentage = calculate_percentages(zone)
        rmse = (mean_squared_error(desired, guess, squared=False))
        mae = mean_absolute_error(desired, guess)

        try:
            print(
                "Function: {} \nLast gen: {} RMSE: {} ({}) MAE: {} ({}) SUM: {} ({})\nZONES: {} PERCENTAGES: {}"
                " A+B%: {} (A% {})% D+E%: {}%\n\n".format(
                    function, best_run.split("\\")[-1].split(".")[0],
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

    print("NUMERO DI ESECUZIONI FATTE = {}".format(len(final_elements)))
    print("NUMERO DI PAZIENTI = {}\n\n".format(len(test_files)))

    rmses = []
    maes = []
    guesses = []
    desireds = []
    nes = []
    functions = []
    zones = []
    percentages = []

    per_patients_maes_mean = {}
    per_patients_rmses_mean = {}
    per_patients_maes_variance = {}
    per_patients_rmses_variance = {}

    per_patients_As_mean = {}
    per_patients_Bs_mean = {}
    per_patients_Cs_mean = {}
    per_patients_Ds_mean = {}
    per_patients_Es_mean = {}

    per_patients_As_var = {}
    per_patients_Bs_var = {}
    per_patients_Cs_var = {}
    per_patients_Ds_var = {}
    per_patients_Es_var = {}

    for test_file in test_files:
        print("CURRENT TEST FILE: {}".format(test_file))
        patients_maes = []
        patients_rmses = []
        patients_As = []
        patients_Bs = []
        patients_Cs = []
        patients_Ds = []
        patients_Es = []
        for elem in final_elements:
            try:
                ne, guess, desired, function = test_function(test_file, elem)

                rmse = (mean_squared_error(desired, guess, squared=False))
                rmses.append(rmse)
                patients_rmses.append(rmse)

                mae = mean_absolute_error(desired, guess)
                maes.append(mae)
                patients_maes.append(mae)

                zone = no_plot_clarke(desired, guess)
                zones.append(zone)

                percentage = calculate_percentages(zone)
                percentages.append(percentage)
                patients_As.append(percentage[0])
                patients_Bs.append(percentage[1])
                patients_Cs.append(percentage[2])
                patients_Ds.append(percentage[3])
                patients_Es.append(percentage[4])

                if rmse + mae < best_sum:
                    best_sum = rmse + mae
                    best_run = elem
                    best_run_file = test_file

                if rmse + mae > worst_sum:
                    worst_sum = rmse + mae
                    worst_run = elem
                    worst_run_file = test_file

                nes.append(ne)
                guesses.append(guess)
                desireds.append(desired)
                functions.append(function)

                pr = True
                try:
                    if pr:
                        print(
                            "Function: {} \nLast gen: {} RMSE: {} ({}) MAE: {} ({}) SUM: {} ({})\nZONES: {} "
                            "PERCENTAGES: {} A+B%: {} (A% {})% D+E%: {}%\n\n".format(
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
                print("Error while evaluating the current function. Sorry!")
                print("FILE: {}\nELEMENT: {}\n\n".format(test_file, elem))
        try:
            per_patients_maes_mean[test_file] = np.mean(np.array(patients_maes))
            per_patients_rmses_mean[test_file] = np.mean(np.array(patients_rmses))
            per_patients_maes_variance[test_file] = np.var(np.array(patients_maes))
            per_patients_rmses_variance[test_file] = np.var(np.array(patients_rmses))

            per_patients_As_mean[test_file] = np.mean(np.array(patients_As))
            per_patients_Bs_mean[test_file] = np.mean(np.array(patients_Bs))
            per_patients_Cs_mean[test_file] = np.mean(np.array(patients_Cs))
            per_patients_Ds_mean[test_file] = np.mean(np.array(patients_Ds))
            per_patients_Es_mean[test_file] = np.mean(np.array(patients_Es))

            per_patients_As_var[test_file] = np.var(np.array(patients_As))
            per_patients_Bs_var[test_file] = np.var(np.array(patients_Bs))
            per_patients_Cs_var[test_file] = np.var(np.array(patients_Cs))
            per_patients_Ds_var[test_file] = np.var(np.array(patients_Ds))
            per_patients_Es_var[test_file] = np.var(np.array(patients_Es))
        except:
            print("error per patients mean")

    try:
        calculate_mean_and_variance(rmses, maes, zones)
    except:
        print("no mean and average")

    try:
        print("\nSORTED RMSES AND MAE")
        rmses = np.sort(rmses)
        maes = np.sort(maes)

        print(rmses)
        print(maes)
    except:
        print("no sort")

    try:
        print("\nSORTED PERCENTAGES")
        p = []
        for zone in zones:
            p.append(calculate_percentages(zone))


        rmses = np.sort(rmses)
        maes = np.sort(maes)

        print(rmses)
        print(maes)
    except:
        print("no sort")

    try:
        print("\nBEST RUN IS: {}\nOVER: {}".format(best_run, best_run_file))
        ne, guess, desired, function = test_function(best_run_file, best_run)
        plot_results(guess, desired, "BEST RUN")
    except:
        print("no plot1")

    try:
        plt, zone = clarke_error_grid(desired, guess, "BEST RUN")
        percentage = calculate_percentages(zone)
        rmse = (mean_squared_error(desired, guess, squared=False))
        mae = mean_absolute_error(desired, guess)

        try:
            print(
                "Function: {} \nLast gen: {} RMSE: {} ({}) MAE: {} ({}) SUM: {} ({})\nZONES: {} PERCENTAGES: {}"
                " A+B%: {} (A% {})% D+E%: {}%\n\n".format(
                    function, best_run.split("\\")[-1].split(".")[0],
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

    try:
        print("\nWORST RUN IS: {}\nOVER: {}".format(worst_run, worst_run_file))
        ne, guess, desired, function = test_function(worst_run_file, worst_run)
        plot_results(guess, desired, "WORST RUN")
    except:
        print("no plot1")

    try:
        plt, zone = clarke_error_grid(desired, guess, "WORST RUN")
        percentage = calculate_percentages(zone)
        rmse = (mean_squared_error(desired, guess, squared=False))
        mae = mean_absolute_error(desired, guess)

        try:
            print(
                "Function: {} \nLast gen: {} RMSE: {} ({}) MAE: {} ({}) SUM: {} ({})\nZONES: {} PERCENTAGES: {}"
                " A+B%: {} (A% {})% D+E%: {}%\n\n".format(
                    function, worst_run.split("\\")[-1].split(".")[0],
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

    try:
        print("\nRMSE/MAE MEAN OVER DIFFERENT patients:\n")
        for rmse in per_patients_rmses_mean:

            print("patients {}: {} & {} & {}".format(rmse.split("\\")[-1].split(".")[0],
                                                     str(round(per_patients_rmses_mean[rmse], 2)),
                                                     str(round(per_patients_maes_mean[rmse], 2)),
                                                     str(round(per_patients_rmses_mean[rmse] +
                                                               per_patients_maes_mean[rmse], 2))))
        print("\nRMSE/MAE VARIANCE OVER DIFFERENT patients:\n")
        for rmse in per_patients_rmses_mean:
            print("patients {}: {} & {} & {}".format(rmse.split("\\")[-1].split(".")[0],
                                                     str(round(per_patients_rmses_variance[rmse], 2)),
                                                     str(round(per_patients_maes_variance[rmse], 2)),
                                                     str(round(
                                                        per_patients_rmses_variance[rmse] + per_patients_maes_variance[
                                                            rmse], 2))
                                                    ))
        print("\nZONES MEAN OVER DIFFERENT patients:\n")
        for rmse in per_patients_rmses_mean:
            print("patients {}: {} & {} & {} & {} & {}".format(rmse.split("\\")[-1].split(".")[0],
                                                     str(round(per_patients_As_mean[rmse], 2)),
                                                     str(round(per_patients_Bs_mean[rmse], 2)),
                                                     str(round(per_patients_Cs_mean[rmse], 2)),
                                                     str(round(per_patients_Ds_mean[rmse], 2)),
                                                     str(round(per_patients_Es_mean[rmse], 2))))
        print("\nZONES VAR OVER DIFFERENT patients:\n")
        for rmse in per_patients_rmses_mean:
            print("patients {}: {} & {} & {} & {} & {}".format(rmse.split("\\")[-1].split(".")[0],
                                                     str(round(per_patients_As_var[rmse], 2)),
                                                     str(round(per_patients_Bs_var[rmse], 2)),
                                                     str(round(per_patients_Cs_var[rmse], 2)),
                                                     str(round(per_patients_Ds_var[rmse], 2)),
                                                     str(round(per_patients_Es_var[rmse], 2))))
    except:
        print("no patients rmse mae means")
