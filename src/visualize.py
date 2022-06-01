import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(statis'tics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot

def clarke_error_grid(ref_values, pred_values, title_string):

    #Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))

    # #Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    # if max(ref_values) > 400 or max(pred_values) > 400:
    #     print "Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(max(ref_values), max(pred_values))
    # if min(ref_values) < 0 or min(pred_values) < 0:
    #     print "Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(min(ref_values),  min(pred_values))

    #Clear plot

    #Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='green', s=8)
    plt.title(title_string + " Clarke Error Grid")
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')

    #Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400)/(400))

    #Plot zone lines
    plt.plot([0,400], [0,400], ':', c='black')                      #Theoretical 45 regression line
    plt.plot([0, 175/3], [70, 70], '-', c='black')
    #plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400],'-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290],[180, 400],'-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320],'-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180],'-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    #Add zone titles
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)

    #Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zone[0] += 1    #Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1    #Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
            zone[2] += 1    #Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
            zone[3] += 1    #Zone D
        else:
            zone[1] += 1    #Zone B
    plt.show()
    return plt, zone

# def parkes_error_zone_detailed(act, pred, diabetes_type):
#     """
#     This function outputs the Parkes Error Grid region (encoded as integer)
#     for a combination of actual and predicted value
#     for type 1 and type 2 diabetic patients
#     Based on the article 'Technical Aspects of the Parkes Error Grid':
#     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3876371/
#     """
#     #Set up plot
#     plt.scatter(act, pred, marker='o', color='green', s=8)
#     plt.title(" Parkes Error Grid")
#     plt.xlabel("Reference Concentration (mg/dl)")
#     plt.ylabel("Prediction Concentration (mg/dl)")
#     plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
#     plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
#     plt.gca().set_facecolor('white')
#
#     #Set axes lengths
#     plt.gca().set_xlim([0, 400])
#     plt.gca().set_ylim([0, 400])
#     plt.gca().set_aspect((400)/(400))
#     def above_line(x_1, y_1, x_2, y_2, strict=False):
#         if x_1 == x_2:
#             return False
#
#         y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
#         return pred > y_line if strict else pred >= y_line
#
#     def below_line(x_1, y_1, x_2, y_2, strict=False):
#         return not above_line(x_1, y_1, x_2, y_2, not strict)
#
#     def parkes_type_1(act, pred):
#         # Zone E
#         if above_line(0, 150, 35, 155) and above_line(35, 155, 50, 550):
#             return 7
#         # Zone D - left upper
#         if (pred > 100 and above_line(25, 100, 50, 125) and
#                 above_line(50, 125, 80, 215) and above_line(80, 215, 125, 550)):
#             return 6
#         # Zone D - right lower
#         if (act > 250 and below_line(250, 40, 550, 150)):
#             return 5
#         # Zone C - left upper
#         if (pred > 60 and above_line(30, 60, 50, 80) and
#                 above_line(50, 80, 70, 110) and above_line(70, 110, 260, 550)):
#             return 4
#         # Zone C - right lower
#         if (act > 120 and below_line(120, 30, 260, 130) and below_line(260, 130, 550, 250)):
#             return 3
#         # Zone B - left upper
#         if (pred > 50 and above_line(30, 50, 140, 170) and
#                 above_line(140, 170, 280, 380) and (act < 280 or above_line(280, 380, 430, 550))):
#             return 2
#         # Zone B - right lower
#         if (act > 50 and below_line(50, 30, 170, 145) and
#                 below_line(170, 145, 385, 300) and (act < 385 or below_line(385, 300, 550, 450))):
#             return 1
#         # Zone A
#         plt.show()
#
#         return 0
#
#     def parkes_type_2(act, pred):
#         # Zone E
#         if (pred > 200 and above_line(35, 200, 50, 550)):
#             return 7
#         # Zone D - left upper
#         if (pred > 80 and above_line(25, 80, 35, 90) and above_line(35, 90, 125, 550)):
#             return 6
#         # Zone D - right lower
#         if (act > 250 and below_line(250, 40, 410, 110) and below_line(410, 110, 550, 160)):
#             return 5
#         # Zone C - left upper
#         if (pred > 60 and above_line(30, 60, 280, 550)):
#             return 4
#         # Zone C - right lower
#         if (below_line(90, 0, 260, 130) and below_line(260, 130, 550, 250)):
#             return 3
#         # Zone B - left upper
#         if (pred > 50 and above_line(30, 50, 230, 330) and
#                 (act < 230 or above_line(230, 330, 440, 550))):
#             return 2
#         # Zone B - right lower
#         if (act > 50 and below_line(50, 30, 90, 80) and below_line(90, 80, 330, 230) and
#                 (act < 330 or below_line(330, 230, 550, 450))):
#             return 1
#         # Zone A
#         return 0
#
#     if diabetes_type == 1:
#         return parkes_type_1(act, pred)
#
#     if diabetes_type == 2:
#         return parkes_type_2(act, pred)
#
#     raise Exception('Unsupported diabetes type')

# clarke_error_zone_detailed = np.vectorize(clarke_error_zone_detailed)
# parkes_error_zone_detailed = np.vectorize(parkes_error_zone_detailed)
