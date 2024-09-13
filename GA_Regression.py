import pandas as pd
import numpy as np
import random
import operator
from sklearn.metrics import r2_score
import copy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut


class GA_elite:

    def __init__(self, X, y, popSize, eliteSize, mutationRate, select, generations):
        self.popSize = popSize
        self.eliteSize = eliteSize
        self.mutationRate = mutationRate
        self.X = X
        self.y = y
        self.select = select
        self.generations = generations

        print('-------Start generating the initial population-------')
        self.InitPop = self.initialPopulation()


    def initialPopulation(self):
        num_features = self.X.shape[1]
        population = np.zeros((self.popSize, num_features))

        for i in range(self.popSize):
            num = random.randint(1, self.select)
            selected_features = [1] * num + [0] * (num_features - num)
            random.shuffle(selected_features)
            population[i] = selected_features

        return population

    def Fitness(self, fpop):
        feature_indices = np.nonzero(fpop)[0]
        if len(feature_indices) == 0:
            return 0, 0, 0

        loo = LeaveOneOut()

        true_list = []
        pred_list = []
        for train_indices, test_indices in loo.split(self.X):
            rf = LinearRegression()
            rf.fit(self.X[train_indices][:, feature_indices], self.y[train_indices])
            y_pred = rf.predict(self.X[test_indices][:, feature_indices])

            true_list.append(self.y[test_indices][0])
            pred_list.append(y_pred[0])

        r2 = r2_score(true_list, pred_list)

        if r2 < 0:
            r2 = 0

        return r2

    def rankRoutes(self, population):
        fitnessResults = {}
        for i in range(0, len(population)):
            fit = self.Fitness(population[i])
            fitnessResults[i] = fit
            print(str(i) + ': fit = ' + str(fit))
        return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


    def selection(self, popRanked):
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        for i in range(0, self.eliteSize):
            selectionResults.append(popRanked[i][0])
        for i in range(0, len(popRanked) - self.eliteSize):
            pick = 100 * random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i, 3]:
                    selectionResults.append(popRanked[i][0])
                    break
        return selectionResults


    def matingPool(self, population, selectionResults):
        matingpool = []
        for i in range(0, len(selectionResults)):
            index = selectionResults[i]
            matingpool.append(population[index])
        return matingpool


    def breedPopulation(self, matingpool):
        children = []
        length = len(matingpool) - self.eliteSize

        for i in range(0, self.eliteSize):
            children.append(matingpool[i])

        for i in range(0, length):
            index = random.randint(1, len(matingpool)-1)
            child = self.breed(matingpool[index])
            children.append(child)
        return children


    def breed(self, parent2):
        parent1 = self.elite

        parent1_indices = np.nonzero(parent1)[0]
        parent2_indices = np.nonzero(parent2)[0]
        indices = np.union1d(parent1_indices, parent2_indices)

        num_features = self.X.shape[1]
        child = [0] * num_features

        if len(indices) > self.select:
            x = self.select
        else:
            x = len(indices)
        num = random.randint(1, x)

        new = random.sample(list(indices), num)
        for mutation_point in new:
            child[mutation_point] = 1

        return np.array(child)


    def mutatePopulation(self, population):
        mutatedPop = []
        length = len(population) - self.eliteSize
        pool = random.sample(population, len(population))

        for i in range(0, self.eliteSize):
            mutatedPop.append(population[i])

        for ind in range(0, length):
            mutatedInd = self.mutate(pool[ind])
            mutatedPop.append(mutatedInd)

        return mutatedPop


    def mutate(self, individual_old):
        individual = copy.deepcopy(individual_old)

        r = random.random()
        if (r < self.mutationRate):
            indices_selected = np.nonzero(individual)[0]
            indices_all = np.array(range(0, len(individual)))

            indice_old = random.sample(list(indices_selected), 1)[0]
            indice_new = random.sample(list(np.setdiff1d(indices_all, indices_selected)), 1)[0]

            individual[indice_old] = 0
            individual[indice_new] = 1

        if (r < 0.05):
            num_features = self.X.shape[1]
            num = random.randint(1, self.select)
            selected_features = [1] * num + [0] * (num_features - num)
            random.shuffle(selected_features)
            individual = selected_features


        return np.array(individual)


    def nextGeneration(self, currentGen):
        popRanked = self.rankRoutes(currentGen)

        # save
        print(popRanked[0:10])
        bestIndex = popRanked[0][0]
        self.elite = currentGen[bestIndex]
        np.savetxt('Result.csv', np.array(self.elite), delimiter=',')

        selectionResults = self.selection(popRanked)
        matingpool = self.matingPool(currentGen, selectionResults)
        children = self.breedPopulation(matingpool)
        nextGeneration = self.mutatePopulation(children)
        return nextGeneration


    def fit(self):
        print('--------------------Start iterating-------------------')
        pop = self.InitPop
        for i in range(0, self.generations):
            print('The '+str(i)+'-th iteration:')
            pop = self.nextGeneration(pop)

        bestRouteIndex = self.rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        return bestRoute


if __name__ == "__main__":
    X = pd.read_csv('Degree_FC_T.csv', header=None)
    y = X.iloc[0, 1:].values
    X = X.iloc[1:, 1:].values
    X = X.T

    GA_model = GA_elite(X=X,
                        y=y,
                        popSize=1000,
                        eliteSize=10,
                        mutationRate=0.4,
                        select=50,
                        generations=100)
    best = GA_model.fit()