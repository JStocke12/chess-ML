import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.preprocessing as pp


def main():
    games = pd.read_csv("games.csv") #https://www.kaggle.com/datasets/datasnaek/chess
    #print(games.head())

    """games["opening_name_basic"] = games["opening_name"].map(lambda s: s.split(":")[0].split(" | ")[0].split(" #")[0])
    print(pd.unique(games["opening_name_basic"]))
    print(len(pd.unique(games["opening_name_basic"])))
    print(len(games["opening_name"]))"""

    opening_eco_raw = games.loc[:, ['opening_eco']]

    opening_one_hot = pp.OneHotEncoder()

    opening_enc = opening_one_hot.fit_transform(opening_eco_raw)

    #opening_enc = pd.get_dummies(opening_eco_raw) #print(pd.get_dummies(games['opening_eco']))

    winner_ordinal = pp.OrdinalEncoder()

    winner_enc = winner_ordinal.fit_transform(games.loc[:,['winner']])[:,0]

    print(winner_ordinal.categories_)

    print(opening_one_hot.categories_[0])

    X_train, X_test, y_train, y_test = ms.train_test_split(
        opening_enc, winner_enc, test_size=0.3, random_state=1
    )

    reg = lm.LinearRegression(fit_intercept=False)
    reg.fit(X_train, y_train)

    print(reg.predict(opening_one_hot.transform([['A00'], ['A01']])))

    print(reg.predict(opening_one_hot.transform(opening_one_hot.categories_[0].reshape(-1,1))))

    print(reg.predict(X_test))
    print(y_test)
    print(reg.score(X_train,y_train))
    print(reg.score(X_test,y_test))

    #plt.plot(line_xs,reg.predict(line_xs))

    #plt.scatter(X_test,y_test)

    #plt.show()

    #print(reg.score(X_train.to_numpy().reshape((-1,1)),y_train))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
