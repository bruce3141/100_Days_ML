# Overview
# Here we use a Genetic Algorithm to find hyperparameters for a Support vector regression (SVR), as applied to fitting the
# UCI Energy efficiency data set:
#  https://archive.ics.uci.edu/ml/datasets/energy+efficiency
# We use the sklearn library for the SVR. The SVM is a popular machine learning tool for classification and regression,
# first identified by Vladimir Vapnik and his colleagues in 1992. SVM regression is considered a nonparametric
# technique because it relies on kernel functions.

# The data set uses 12 different building shapes simulated in Ecotect. The buildings differ with respect to the glazing area,
# the glazing area distribution, and the orientation, amongst other parameters. The data includes simulating various settings as
# functions of the afore-mentioned characteristics to obtain 768 building shapes. The dataset comprises 768 samples and 8 features,
# aiming to predict two real valued responses. It can also be used as a multi-class classification problem if the response is
# rounded to the nearest integer.

import numpy as np
import random as rd
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib as mpl

from time import time
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import seaborn as sns

np.set_printoptions(linewidth=500)
pd.set_option('display.max_columns', 500)
mpl.rcParams['toolbar'] = 'None'
plt.style.use(['dark_background'])

# Use k-fold cross validation to calculate the objective:
def k_fold_model_evaluation(X, model_hp1, model_hp2, k_folds=10):
    kf = model_selection.KFold(n_splits=k_folds)
    objective_val = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = svm.SVR(kernel='rbf', C=model_hp1, gamma=model_hp2)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)  # accuracy
        objective_val += 1 - accuracy  # objective function = the error
    return objective_val/k_folds


# Each hyper-parameter corresponds to a chromosome. Each of the chromosomes are then strung together to form a genotype.
def phenotype(param_min, param_max, chromosome):
    s = 0
    # Sum the chromosone:
    for i, c in enumerate(reversed(chromosome)):
        s += c*2**i
    precision = (param_max - param_min) / (2**len(chromosome) - 1)
    decoded = s*precision + param_min
    return decoded


# Grid search. Warning: this can take a long time.  On my machine and with n_grid_steps = 20,
# the function took 5 hours for 20x20 grid-search in my case.
def grid_search(C_min, C_max, gamma_min, gamma_max, n_grid_steps):
    model_grid = svm.SVR(kernel='rbf')  # Initialize model:
    gr1 = np.linspace(C_min, C_max, n_grid_steps).round(2).tolist()
    gr2 = np.linspace(gamma_min, gamma_max, n_grid_steps).round(3).tolist()
    param_grid = {'C': gr1, 'gamma': gr2}
    gridsearch = GridSearchCV(model_grid, param_grid, return_train_score=True)
    start = time()
    gridsearch.fit(X, y)
    stop = time()
    print("Grid Search took %.2f seconds." % (stop - start))
    gs_best_results_dict = gridsearch.best_params_
    gs_best_score = gridsearch.best_score_
    print(gs_best_results_dict)
    print('best score (grid search) = ', gs_best_score, '; objective val = ', 1 - gs_best_score)
    # --- Evaluate best parameter values:
    final_model_grid = svm.SVR(kernel='rbf', C=gs_best_results_dict['C'], gamma=gs_best_results_dict['gamma'])
    final_model_grid.fit(X, y)
    y_grid = final_model_grid.predict(X)
    pvt_gs = pd.pivot_table(pd.DataFrame(gridsearch.cv_results_), values='mean_test_score', index='param_C', columns='param_gamma')
    sns.heatmap(pvt_gs)
    return gs_best_score, gs_best_results_dict


def random_search(C_min, C_max, gamma_min, gamma_max):
    kfolds, iterations = 10, 20
    scale_C, scale_gamma = C_max - C_min, gamma_max - gamma_min
    # Initialize model:
    model_rnd = svm.SVR(kernel='rbf')
    # Parameter distributions:
    param_dist = {'C': scipy.stats.uniform(loc=C_min, scale=scale_C), 'gamma': scipy.stats.uniform(loc=gamma_min, scale=scale_gamma)}
    rand_search = RandomizedSearchCV(model_rnd, param_distributions=param_dist, n_iter=iterations, cv=kfolds, return_train_score=True)
    start = time()
    rand_search.fit(X, y)
    stop = time()
    print("Random Search took %.2f seconds." % (stop - start))
    rs_best_results_dict = rand_search.best_params_
    rs_best_score = rand_search.best_score_
    print('Random Search results: ', rs_best_results_dict)
    print('best score (random search) = ', rs_best_score, '; objective val = ', 1 - rs_best_score)
    return rs_best_score, rs_best_results_dict


def mutate(child, probability):
    mutated_child = []
    for k, gene in enumerate(child):
        random_number = np.random.rand()
        if random_number < probability:
            if child[k] == 0:
                child[k] = 1
            else:
                child[k] = 0
            mutated_child = child
        else:
            mutated_child = child
    return mutated_child


# --- Read and shuffle data:
data = pd.read_excel('ENB2012_data.xlsx')
data = data.sample(frac=1)  # random shuffle data; fraction = 100% of data

# --- Extract features and labels, scale the data:
X_initial = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
Y = pd.DataFrame(data, columns=['Y1'])
xb = pd.get_dummies(X_initial, columns=['X6', 'X8'])  # convert X6 and X8 categorical data to 1-hot
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(xb)
n_data_points = len(X)
print('# of observations = ', n_data_points)
# Need this for k-fold cross validation:
y = np.reshape(Y.values, (n_data_points,))

# --- Split for testing:
fraction = 0.9
n_final_train_samples = int(len(X)*fraction)
X_train_default, y_train_default = X[:n_final_train_samples], y[:n_final_train_samples]
X_test_default, y_test_default = X[n_final_train_samples:], y[n_final_train_samples:]

# --- Genetic Algorithm Parameters:
generations = 5
n_population_members = 20
n_contestants = 5
n_families = 3
prob_for_mutation = 0.25  # probability of mutation
k_folds = 10  # n-fold validations
n_hyper_params = 2

# --- Search space:
C_min, C_max = 50, 200000
gamma_min, gamma_max = 0.01, 0.4
n_grid_steps = 5  # for grid search

rs_C_min, rs_C_max = C_min, C_max
rs_gamma_min, rs_gamma_max = gamma_min, gamma_max

scale_1 = rs_C_max-rs_C_min
scale_2 = rs_gamma_max-rs_gamma_min

# --- Grid search:
# best_score, gs_best_results_dict = grid_search(C_min, C_max, gamma_min, gamma_max, n_grid_steps)
# plt.show()

# Take a peek at a uniform distribution for C:
vals = scipy.stats.uniform.rvs(loc=rs_C_min, scale=scale_1, size=500)
fig = plt.figure()
plt.hist(vals, histtype='stepfilled', alpha=0.5)

# --- Random search:
rs_best_score, rs_best_results_dict = random_search(C_min, C_max, gamma_min, gamma_max)
print('\n')

# --- Create a sample population to extract families:
genes_per_chromosone = 15  # should be adequate to store range
chrom_length = n_hyper_params*genes_per_chromosone
initial_chromosome = [rd.randint(0, 1) for k in range(chrom_length)]
population = np.empty((0, chrom_length))
for i in range(n_population_members):
    rd.shuffle(initial_chromosome)
    population = np.vstack((population, initial_chromosome))

# --- Print population hyper-params and inspect sample space:
hyper_params = np.empty((0, 4))
print('Default training set samples = ', n_final_train_samples)
print('Default test set samples = ', n_data_points - n_final_train_samples)
print('Training samples in each k-fold = ', n_data_points - int(n_data_points/k_folds))
print('Test samples in each k-fold = ', int(n_data_points/k_folds))
print('Evaluating sample space:')
for i in range(n_population_members):
    print('evaluate:', i)
    select_chrom = np.reshape(population[i], (n_hyper_params, genes_per_chromosone))  # separates the hyper-params
    decoded_C = phenotype(C_min, C_max, select_chrom[0])
    decoded_gamma = phenotype(gamma_min, gamma_max, select_chrom[1])
    obj_test = k_fold_model_evaluation(X, decoded_C, decoded_gamma)
    obj_model = svm.SVR(kernel='rbf', C=decoded_C, gamma=decoded_gamma)
    obj_model.fit(X_train_default, y_train_default)
    accuracy_model = obj_model.score(X_test_default, y_test_default)  # accuracy
    objective_model = 1 - accuracy_model  # objective function = the error
    decoded_temp = [obj_test, objective_model, decoded_C, decoded_gamma]
    hyper_params = np.vstack((hyper_params, np.asarray(decoded_temp)))
df = pd.DataFrame(hyper_params)

# Sort initial parameter space (with comparision to random search) by objective value:
a_df = df.sort_values(df.columns[0])

# --- Check Random Sample Results:
obj_model = svm.SVR(kernel='rbf', C=rs_best_results_dict['C'], gamma=rs_best_results_dict['gamma'])
obj_model.fit(X_train_default, y_train_default)
accuracy_model = obj_model.score(X_test_default, y_test_default)  # accuracy
objective_model = 1-accuracy_model  # objective function = the error
row_list = np.array([1-rs_best_score, objective_model, rs_best_results_dict['C'], rs_best_results_dict['gamma']]).reshape(1, -1).tolist()
row = pd.DataFrame(row_list)
blank_row = pd.DataFrame(np.array([' ', ' ', ' ', ' ']).reshape(1, -1).tolist())
a_df = a_df.append([blank_row, row])
a_df.columns = ['obj', 'model obj', 'C', 'gamma']
# Random search results are in the last row:
a_df = a_df.reset_index(drop=True)
print('Initial Evaluation of Sample Space:')
print(a_df)

# --- Keep track of results:
new_population = np.empty((0, chrom_length))
min_in_gen_x1 = []
min_in_gen_x2 = []
worst_best_in_gen_x = []
final_best_in_gen_x = np.empty((0, 5))

# --- Loop over Generations:
for g in range(generations):

    print('\n')
    print('---------------------')
    print('Generation #', g)
    print('---------------------')
    print('---------------------')

    all_parent_1_in_gen_x = np.empty((0, chrom_length + 1))
    all_parent_2_in_gen_x = np.empty((0, chrom_length + 1))
    all_parents_in_gen_x = np.empty((0, chrom_length + 1))
    all_in_gen_x1 = np.empty((0, chrom_length + 1))
    all_in_gen_x2 = np.empty((0, chrom_length + 1))
    all_in_gen_x = np.empty((0, chrom_length + 1))

    # --- Each new generation starts with a new population:
    if g > 0:
        print('Population updated.')
        new_population = np.empty((0, chrom_length))

    print('population size = ', len(population))

    # --- Loop over the Families to select parents:
    for f in range(n_families):
        print('Family: ', f)
        print('---------------------')

        # --- Create tournament to get 2 best parents:
        parents = [None, None]
        for k in range(2):
            candidate_ids = rd.sample(range(0, n_population_members), n_contestants)  # unique random contestants
            decoded_C = [None for d in range(n_contestants)]
            decoded_gamma = [None for d in range(n_contestants)]
            chromosome = [None for d in range(n_contestants)]
            objective = [None for d in range(n_contestants)]
            for c_idx in range(n_contestants):
                chromosome[c_idx] = population[candidate_ids[c_idx]]
                select_chrom = np.reshape(chromosome[c_idx], (n_hyper_params, genes_per_chromosone))  # separates the hyper-params
                decoded_C[c_idx] = phenotype(C_min, C_max, select_chrom[0])
                decoded_gamma[c_idx] = phenotype(gamma_min, gamma_max, select_chrom[1])
                objective[c_idx] = k_fold_model_evaluation(X, decoded_C[c_idx], decoded_gamma[c_idx])
            # Select the optimal value from the k-folds:
            winner = np.min(objective)
            winner_indx = int(np.argmin(objective))
            parents[k] = np.concatenate(([winner], chromosome[winner_indx])).tolist()

        # --- Create children:
        n_children = 2
        children = [None for c in range(n_children)]

        # --- 2-point crossover:
        ch_1, ch_2 = rd.sample(range(1, chrom_length+1), n_children)
        if ch_1 < ch_2:
            mid_seg_1 = parents[0][ch_1:ch_2 + 1]
            mid_seg_2 = parents[1][ch_1:ch_2 + 1]
            first_seg_1 = parents[0][1:ch_1]
            sec_seg_1 = parents[0][ch_2 + 1:]
            first_seg_2 = parents[1][1:ch_1]
            sec_seg_2 = parents[1][ch_2 + 1:]
        else:
            mid_seg_1 = parents[0][ch_2:ch_1 + 1]
            mid_seg_2 = parents[1][ch_2:ch_1 + 1]
            first_seg_1 = parents[0][1:ch_2]
            sec_seg_1 = parents[0][ch_1 + 1:]
            first_seg_2 = parents[1][1:ch_2]
            sec_seg_2 = parents[1][ch_1 + 1:]

        child_1 = np.concatenate((first_seg_1, mid_seg_2, sec_seg_1))
        child_2 = np.concatenate((first_seg_2, mid_seg_1, sec_seg_2))

        # --- Mutations:
        mutated_child_1 = mutate(child_1, prob_for_mutation)
        mutated_child_2 = mutate(child_2, prob_for_mutation)

        # --- Mutated Child-1 (decode hyper-params and score):
        select_chrom = np.reshape(mutated_child_1, (n_hyper_params, genes_per_chromosone))  # separate params
        decoded_C = phenotype(C_min, C_max, select_chrom[0])
        decoded_gamma = phenotype(gamma_min, gamma_max, select_chrom[1])
        objective_1 = k_fold_model_evaluation(X, decoded_C, decoded_gamma)  # evaluate using k-folds:
        mutated_child_1_with_score = [objective_1] + mutated_child_1.tolist()

        # --- Mutated Child-2 (decode hyper-params and score):
        select_chrom = np.reshape(mutated_child_2, (n_hyper_params, genes_per_chromosone))  # separate params
        decoded_C = phenotype(C_min, C_max, select_chrom[0])
        decoded_gamma = phenotype(gamma_min, gamma_max, select_chrom[1])
        objective_2 = k_fold_model_evaluation(X, decoded_C, decoded_gamma)  # evaluate using k-folds:
        mutated_child_2_with_score = [objective_2] + mutated_child_2.tolist()

        # -- Children:
        all_in_gen_x1 = np.vstack((all_in_gen_x1, np.array(mutated_child_1_with_score)))
        all_in_gen_x2 = np.vstack((all_in_gen_x2, np.array(mutated_child_2_with_score)))
        all_in_gen_x = np.vstack((all_in_gen_x1, all_in_gen_x2))

        # -- Parents:
        all_parent_1_in_gen_x = np.vstack((all_parent_1_in_gen_x, np.array(parents[0])))
        all_parent_2_in_gen_x = np.vstack((all_parent_2_in_gen_x, np.array(parents[1])))
        all_parents_in_gen_x = np.vstack((all_parent_1_in_gen_x, all_parent_2_in_gen_x))

    # --- Rank children and parents by objective value:
    best_children_df = pd.DataFrame(all_in_gen_x)
    best_parents_df = pd.DataFrame(all_parents_in_gen_x)
    best_children = best_children_df.sort_values(best_children_df.columns[0])
    best_children = best_children[:int(n_population_members/2)]
    best_parents = best_parents_df.sort_values(best_parents_df.columns[0])
    best_parents = best_parents[:int(n_population_members/2)]

    # --- Sort total population by objective value:
    new_population_df = pd.concat([best_children, best_parents], ignore_index=True)
    new_population_df = new_population_df.sort_values(new_population_df.columns[0])
    print('New ranked population: ')
    print(new_population_df.iloc[:, 0:1])
    new_population = new_population_df.values

    # --- Best in this generation:
    best_score = new_population[0, 0]
    R2 = 1 - best_score
    select_chrom = np.reshape(new_population[0, 1:].tolist(), (n_hyper_params, genes_per_chromosone))  # separate params
    decoded_C = phenotype(C_min, C_max, select_chrom[0])
    decoded_gamma = phenotype(gamma_min, gamma_max, select_chrom[1])

    # --- Keep the best performers in each generation:
    best_in_gen_x_with_params = [g] + [new_population[0, 0]] + [R2] + [decoded_C] + [decoded_gamma]
    final_best_in_gen_x = np.vstack((final_best_in_gen_x, best_in_gen_x_with_params))
    print('Best in generation: ')
    print(pd.DataFrame(final_best_in_gen_x))

# --- At completing all generations, extract the most evolved member:
final_candidates_df = pd.DataFrame(final_best_in_gen_x)
# sort final candidates by objective value:
final_candidates_df = final_candidates_df.sort_values(final_candidates_df.columns[1])  # best are at the top of the table:
final_candidates = final_candidates_df.values
best_candidate = final_candidates[0, :]
print('\n')
print('best_candidates:')
final_candidates_df.columns = ['generation', 'objective', 'R^2', 'C', 'gamma']
print(final_candidates_df)

plt.show()


# Example results:
# final_best_in_generation:
# [[0.         0.93985718 0.06014282 1.         0.         1.         1.         0.        ]
#  [1.         0.94165477 0.05834523 1.         1.         1.         0.         1.        ]
#  [2.         0.93985718 0.06014282 1.         0.         1.         1.         0.        ]
#  [3.         0.93985718 0.06014282 1.         0.         1.         1.         0.        ]]

# param_1 =  961.3330484939115 ; param_2 =  0.08107852412488174










































