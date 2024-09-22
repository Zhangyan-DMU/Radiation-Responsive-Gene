To effectively identify radiation-responsive genes, we developed a LET-regression model utilizing genetic algorithms (GA). Essentially, GA was employed to select the optimal gene combinations, allowing the LET-regression model to achieve optimal performance. The genes chosen by the GA are evidently the radiation-responsive ones.

The input features of the model were FC-matrix (expression levels and degrees) of genes, and the target variable was the LET values. The four CSV files are datasets used for modeling.

The regression model used in this algorithm was multiple linear regression, and the training strategy was leave-one-out cross-validation.

The specific process of the genetic algorithm is as follows: At the initialization stage, the number of features chosen by each individual was randomly determined between 1 and 50. During the crossover operation, we merged the features from both parents and then randomly picked a subset of them. The mutation process first involves randomly removing a feature from the individual. Next, a new feature from the pool of unselected features was randomly added to the individual.
