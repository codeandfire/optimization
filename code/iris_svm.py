"""Short script to apply an SVM to the Iris dataset."""

import argparse

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# command-line arguments
# ----------------------

parser = argparse.ArgumentParser(description='Apply an SVM to the Iris dataset.')
subparsers = parser.add_subparsers(dest='subparsers_name', title='subcommands')
parser_dataplot = subparsers.add_parser('dataplot', help='Plot the Iris dataset.')
parser_classify = subparsers.add_parser(
    'classify',
    help='Classify two classes using an SVM and plot the resulting fit.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser_dataplot.add_argument(
    'attribute',
    choices=['sepal', 'petal'],
    help='Attribute to plot: sepal length/width or petal length/width.')

parser_classify.add_argument('class1', choices=['setosa', 'versicolor', 'virginica'],
                             help='Name of class 1')
parser_classify.add_argument('class2', choices=['setosa', 'versicolor', 'virginica'],
                             help='Name of class 2')
parser_classify.add_argument(
    'attribute',
    choices=['sepal', 'petal'],
    help='Attribute to use for classification: sepal length/width or petal length/width.')
parser_classify.add_argument('--eta', type=float, default=5.0, help='Slack penalty parameter.')
parser_classify.add_argument(
    '--hard',
    action='store_true',
    help='Apply a hard SVM classifier instead of a soft one.')

args = parser.parse_args()

# load the data
iris_dataset = pd.read_csv(
    'iris/iris.data',
    header=None,
    names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species'])

# perform some sanity checks
assert iris_dataset.shape == (150, 5)
assert list(iris_dataset['species'].unique()) == ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# retain relevant attributes only
if args.attribute == 'sepal':
    iris_dataset = iris_dataset[['sepal-length', 'sepal-width', 'species']]
else:
    iris_dataset = iris_dataset[['petal-length', 'petal-width', 'species']]

if args.subparsers_name == 'dataplot':
    fig, ax = plt.subplots(1, 1)

    for species in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
        # extract the data corresponding to that species
        data = iris_dataset[iris_dataset['species'] == species]

        ax.scatter(data.iloc[:, 0], data.iloc[:, 1], label=species)

    ax.set_xlabel('length')
    ax.set_ylabel('width')
    ax.legend()

    filename = 'plots/dataplot-{}.png'.format(args.attribute)
    plt.savefig(filename)
    print('Data plot saved to', filename)

elif args.subparsers_name == 'classify':
    # extract the data corresponding to the two classes
    data_class1 = iris_dataset[
        iris_dataset['species'] == 'Iris-' + args.class1].drop('species', axis=1).to_numpy()
    data_class2 = iris_dataset[
        iris_dataset['species'] == 'Iris-' + args.class2].drop('species', axis=1).to_numpy()

    # frame the optimization problem
    a = cp.Variable((2, ))
    b = cp.Variable()
    if args.hard:
        constraints = [(data_class1 @ a) + b >= 1, (data_class2 @ a) + b <= -1]
        problem = cp.Problem(cp.Minimize(cp.norm(a)), constraints)
    else:
        u = cp.Variable((len(data_class1) + len(data_class2), ))
        constraints = [(data_class1 @ a) + b >= 1 - u[ : len(data_class1)],
                       (data_class2 @ a) + b <= -1 + u[len(data_class1) : ],
                       u >= 0]
        problem = cp.Problem(cp.Minimize(cp.norm(a) + args.eta * cp.sum(u)), constraints)

    # solve the optimization problem
    problem.solve()

    # if the optimal value of the objective is infinity then the problem is
    # essentially infeasible.
    if problem.value == float('inf'):
        raise RuntimeError("Infeasible problem! Try using a soft SVM instead.")

    fig, ax = plt.subplots(1, 1)
    
    # plot the data points of the two classes
    ax.scatter(data_class1[:, 0], data_class1[:, 1], label='Iris-' + args.class1)
    ax.scatter(data_class2[:, 0], data_class2[:, 1], label='Iris-' + args.class2)

    # plot the optimal slab fitted by the SVM.
    x_min, x_max = ax.get_xlim()
    x = np.linspace(x_min, x_max)
    y1 = (1 - a.value[0] * x - b.value) / a.value[1]    # hyperplane 1 of the slab
    y2 = (-1 - a.value[0] * x - b.value) / a.value[1]   # hyperplane 2 of the slab
    ax.plot(x, y1, color='black')
    ax.plot(x, y2, color='black')

    ax.set_xlabel('length')
    ax.set_ylabel('width')
    ax.legend()

    filename = 'plots/classify-{}-{}-{}.png'.format(args.attribute, args.class1, args.class2)
    plt.savefig(filename)
    print('Classification plot saved to', filename)
