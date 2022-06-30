import os
import math
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from src.test.visualize import clarke_error_grid, no_plot_clarke

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
    tf = get_function(result).split("\n")[0]
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
        try:
            z = eval(func)
            desired.append(y * 18)
            guesses.append(z * 18)
            if y == z:
                good_extimations += 1
            else:
                bad_extimations += 1
        except:
            pass
    return good_extimations, bad_extimations, number_of_elements, guesses, desired, tf


def calculate_percentages(zone):
    percentages = [0] * 5

    total = 0
    for x in zone:
        total += x
    percentages[0] = round((zone[0] / total) * 100, 2)
    percentages[1] = round((zone[1] / total) * 100, 2)
    percentages[2] = round((zone[2] / total) * 100, 2)
    percentages[3] = round((zone[3] / total) * 100, 2)
    percentages[4] = round((zone[4] / total) * 100, 2)
    return percentages


def open_last_gen(path):
    file_list = os.listdir(path)
    filename = 0
    for file in file_list:
        try:
            name = int(file.split(".")[0])
            if name > filename:
                filename = name
        except:
            pass
    return path + "\\" + str(filename) + ".txt"


def save_stats(stats_path, param, rmse, mae, zone, percentages):
    pass


if __name__ == '__main__':
    test_file = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose\\596\\596-ws-testing.csv"

    stats_path = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\results_stats.csv"

    exp_results = {
        "ff_eval_zone": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\eval_zone",
        "ff_eval_zone_d": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\eval_zone_d",
        "ff_percentage_penality": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\penality",
        "ff_percentage_penality_e": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\penality_e",
        "ff_rmse": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\rmse",
        "ff_mae": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\mae",
        "ff_w_penality_mae": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\w_penality_mae",
        "ff_w_penality_rmse": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\w_penality_rmse",
        "ff_w_penality_mae_rmse": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\"
                                  "w_penality_mae_rmse",
        "ff_w_eval_zone_mae": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\w_eval_zone_mae",
        "ff_w_eval_zone_rmse": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\w_eval_zone_rmse",
        "ff_w_eval_zone_mae_rmse": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\"
                                   "w_eval_zone_mae_rmse",
        "ff_mae_rmse": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\mae_rmse",
        "ff_pen_eval_mae_rmse": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\ff_analysis\\"
                                "w_penality_eval_zone_mae_rmse",
        "eval_zone_mutation_25": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\genetic_operators\\"
                                 "mutation_25\\eval_zone",
        "eval_zone_mutation_50": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\genetic_operators\\"
                                 "mutation_50\\eval_zone",
        "eval_zone_mutation_100": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\genetic_operators\\"
                                  "mutation_100\\eval_zone",
        "eval_zone_mae_mutation_25": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\genetic_operators\\"
                                     "mutation_25\\w_eval_zone_mae",
        "eval_zone_mae_mutation_50": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\genetic_operators\\"
                                     "mutation_50\\w_eval_zone_mae",
        "eval_zone_mae_mutation_100": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\genetic_operators\\"
                                      "mutation_100\\w_eval_zone_mae",
        "eval_zone_mutation_50_cross90": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\genetic_operators\\"
                                         "mutation_50_crossover_090\\eval_zone",
        "eval_zone_mutation_50_cross95": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\genetic_operators\\"
                                         "mutation_50_crossover_095\\eval_zone",
        "eval_zone_mutation_50_cross99": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\genetic_operators\\"
                                         "mutation_50_crossover_099\\eval_zone",
        "eval_zone_mae_mutation_25_cross90": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                             "genetic_operators\\mutation_50_crossover_090\\w_eval_zone_mae",
        "eval_zone_mae_mutation_25_cross95": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                             "genetic_operators\\mutation_50_crossover_095\\w_eval_zone_mae",
        "eval_zone_mae_mutation_25_cross99": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                             "genetic_operators\\mutation_50_crossover_099\\w_eval_zone_mae",
        "eval_zone_mutation_25_cross85_tournament7": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                                     "tournament\\mutation_25_crossover_085_tournament_7\\eval_zone",
        "eval_zone_mutation_25_cross85_tournament14": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                                      "tournament\\mutation_25_crossover_085_tournament_14\\eval_zone",
        "eval_zone_mutation_25_cross85_tournament28": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                                      "tournament\\mutation_25_crossover_085_tournament_28\\eval_zone",
        "eval_zone_mae_mutation_50_cross85_tournament7": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                                         "tournament\\mutation_50_crossover_085_tournament_7\\"
                                                         "w_eval_zone_mae",
        "eval_zone_mae_mutation_50_cross85_tournament14": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                                          "tournament\\mutation_50_crossover_085_tournament_14\\"
                                                          "w_eval_zone_mae",
        "eval_zone_mae_mutation_50_cross85_tournament28": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                                          "tournament\\mutation_50_crossover_085_tournament_28\\"
                                                          "w_eval_zone_mae",
        "eval_zone_mutation50_cross85_tournament7_elite20": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                                            "elite_size\\20\\eval_zone",
        "eval_zone_mutation50_cross85_tournament7_elite40": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                                            "elite_size\\40\\eval_zone",
        "eval_zone_mutation50_cross85_tournament7_elite50": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\results\\"
                                                            "elite_size\\50\\eval_zone",
        "eval_zone_mae_mutation50_cross85_tournament7_elite20": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\"
                                                                "results\\elite_size\\20\\w_eval_zone_mae",
        "eval_zone_mae_mutation50_cross85_tournament7_elite40": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\"
                                                                "results\\elite_size\\40\\w_eval_zone_mae",
        "eval_zone_mae_mutation50_cross85_tournament7_elite50": "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\"
                                                                "results\\elite_size\\50\\w_eval_zone_mae",
    }

    see_clarke = {
        "ff_eval_zone": False,
        "ff_eval_zone_d": False,
        "ff_percentage_penality": False,
        "ff_percentage_penality_e": False,
        "ff_rmse": False,
        "ff_mae": False,
        "ff_w_penality_mae": False,
        "ff_w_penality_rmse": False,
        "ff_w_penality_mae_rmse": False,
        "ff_w_eval_zone_mae": True,
        "ff_w_eval_zone_rmse": False,
        "ff_w_eval_zone_mae_rmse": False,
        "ff_mae_rmse": False,
        "ff_pen_eval_mae_rmse": False,
        "eval_zone_mutation_25": False,
        "eval_zone_mutation_50": False,
        "eval_zone_mutation_100": False,
        "eval_zone_mae_mutation_25": False,
        "eval_zone_mae_mutation_50": False,
        "eval_zone_mae_mutation_100": False,
        "eval_zone_mutation_50_cross90": False,
        "eval_zone_mutation_50_cross95": False,
        "eval_zone_mutation_50_cross99": False,
        "eval_zone_mae_mutation_25_cross90": False,
        "eval_zone_mae_mutation_25_cross95": False,
        "eval_zone_mae_mutation_25_cross99": False,
        "eval_zone_mutation_25_cross85_tournament7": False,
        "eval_zone_mutation_25_cross85_tournament14": False,
        "eval_zone_mutation_25_cross85_tournament28": False,
        "eval_zone_mae_mutation_50_cross85_tournament7": False,
        "eval_zone_mae_mutation_50_cross85_tournament14": False,
        "eval_zone_mae_mutation_50_cross85_tournament28": False,
        "eval_zone_mutation50_cross85_tournament7_elite20": False,
        "eval_zone_mutation50_cross85_tournament7_elite40": False,
        "eval_zone_mutation50_cross85_tournament7_elite50": False,
        "eval_zone_mae_mutation50_cross85_tournament7_elite20": False,
        "eval_zone_mae_mutation50_cross85_tournament7_elite40": False,
        "eval_zone_mae_mutation50_cross85_tournament7_elite50": False,
    }

    calculate_results = {
        "ff_eval_zone": False,
        "ff_eval_zone_d": False,
        "ff_percentage_penality": False,
        "ff_percentage_penality_e": False,
        "ff_rmse": False,
        "ff_mae": False,
        "ff_w_penality_mae": False,
        "ff_w_penality_rmse": False,
        "ff_w_penality_mae_rmse": False,
        "ff_w_eval_zone_mae": False,
        "ff_w_eval_zone_rmse": False,
        "ff_w_eval_zone_mae_rmse": False,
        "ff_mae_rmse": False,
        "ff_pen_eval_mae_rmse": False,
        "eval_zone_mutation_25": True,
        "eval_zone_mutation_50": False,
        "eval_zone_mutation_100": False,
        "eval_zone_mae_mutation_25": False,
        "eval_zone_mae_mutation_50": True,
        "eval_zone_mae_mutation_100": False,
        "eval_zone_mutation_50_cross90": True,
        "eval_zone_mutation_50_cross95": True,
        "eval_zone_mutation_50_cross99": True,
        "eval_zone_mae_mutation_25_cross90": True,
        "eval_zone_mae_mutation_25_cross95": True,
        "eval_zone_mae_mutation_25_cross99": True,
        "eval_zone_mutation_25_cross85_tournament7": True,
        "eval_zone_mutation_25_cross85_tournament14": True,
        "eval_zone_mutation_25_cross85_tournament28": True,
        "eval_zone_mae_mutation_50_cross85_tournament7": True,
        "eval_zone_mae_mutation_50_cross85_tournament14": True,
        "eval_zone_mae_mutation_50_cross85_tournament28": True,
        "eval_zone_mutation50_cross85_tournament7_elite20": True,
        "eval_zone_mutation50_cross85_tournament7_elite40": True,
        "eval_zone_mutation50_cross85_tournament7_elite50": True,
        "eval_zone_mae_mutation50_cross85_tournament7_elite20": True,
        "eval_zone_mae_mutation50_cross85_tournament7_elite40": True,
        "eval_zone_mae_mutation50_cross85_tournament7_elite50": True,
    }

    calculate_all = False
    see_all_clarke = False

    for result in exp_results:
        if calculate_results[result] or calculate_all:
            print("Experiment: {}".format(result))
            try:
                filepath = open_last_gen(exp_results[result])
                ge, be, ne, guesses, desired, function = test_function(test_file, filepath)
                filename = exp_results[result].split("\\")[-1]
                rmse = mean_squared_error(desired, guesses, squared=False)
                mae = mean_absolute_error(desired, guesses)
                if see_clarke[result] or see_all_clarke:
                    plt, zone = clarke_error_grid(desired, guesses, result + " " + filepath.split("\\")[-1])
                else:
                    zone = no_plot_clarke(desired, guesses)
                percentages = calculate_percentages(zone)
                print(
                    "Function: {} \nLast gen: {} RMSE: {} ({}) MAE: {} ({}) SUM: {} ({})\nZONES: {} PERCENTAGES: {}"
                    " A+B%: {} (A% {})% D+E%: {}%\n\n".format(
                        function, filepath.split("\\")[-1].split(".")[0],
                        str(round(rmse, 2)), str(round((rmse / 18), 2)),
                        str(round(mae, 2)), str(round((mae / 18), 2)),
                        str(round((rmse + mae), 2)), str(round((rmse + mae) / 18, 2)),
                        str(zone),
                        str(percentages),
                        str(round(percentages[0] + percentages[1], 2)),
                        str(round(percentages[0], 2)),
                        str(round(percentages[3] + percentages[4], 2))
                    ))
                # save_stats(stats_path, exp_results[result], rmse, mae, zone, percentages)
            except:
                print("Error while evaluating the current function. Sorry!\n\n")
