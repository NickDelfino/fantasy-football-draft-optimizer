import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import pandas as pd

def main():

    fantasyStats2017 = pd.read_csv("./fantasy-football-data-2017.csv")

    fantasyStats2017.pop('Rk')
    fantasyStats2017.pop('Player')
    fantasyStats2017.pop('Tm')

    print("Create the target for the 2017 data.")
    fantasyStats2017['target'] = fantasyStats2017.apply(lambda row: get_target_column(row.FDPt), axis=1)

    print("Clean up and normalize data.")
    fantasyStats2017['FantPos'] = fantasyStats2017['FantPos'].apply(normalize_player_position)
    fantasyStats2017.fillna(0, inplace=True)

    print("Spot check 2017 data target after creation.")
    print(fantasyStats2017.head(5))

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=30))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    train, test = train_test_split(fantasyStats2017, test_size=.2)
    train, validation = train_test_split(train, test_size=.2)

    train_data = train.copy()
    train_target = train_data.pop('target')

    validation_data = validation.copy()
    validation_target = validation_data.pop('target')

    model.fit(train_data, train_target, epochs=10, batch_size=16, validation_data=(validation_data,validation_target))

    test_data = test.copy()
    test_target = test_data.pop('target')

    loss, acc = model.evaluate(test_data, test_target, batch_size=16)
    print("acc: " + str(acc))
    print("loss: " + str(loss))

def get_target_column(fantasyPoints):
    if fantasyPoints > 150:
        return 1
    else:
        return 0

def normalize_player_position(pos):
    if pos == "RB":
        return 1
    elif pos == "QB":
        return 2
    elif pos == "WR":
        return 3
    elif pos == "TE":
        return 4
    else:
        return 0

if __name__ == "__main__":
    main()