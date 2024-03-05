import math

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import scipy.interpolate as sci
import scipy.optimize as sco
import scipy.stats as scs
import tensorflow as tf
import yfinance as yf
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

nltk.download('vader_lexicon')
import tensorflow_datasets as tfds

np.random.seed(5555)


data_file_directory = 'C:/Users/matth/Documents/Computer Science Code/Fins3645 Assignment/Data Files/'
def main():
    station_1_ETL()
    station_2 = station_2_features_engineering()
    station_3 = Station3_modelDesign_and_Station4_implementation()
    Station3_modelDesign_and_Station4_implementation.sentiment_analyser(station_3)
    Station3_modelDesign_and_Station4_implementation.plotMovingAverage(station_2.df_stocks()[:][:], window=6, plot_anomalies=True, plot_intervals=True)
    Station3_modelDesign_and_Station4_implementation.modern_portfolio_theory(station_3,station_2.df_stocks())

def station_1_ETL():
    global df_client_details
    df_client_details = client_details_load_and_clean()
    global df_asx200_top10
    df_asx200_top10 = asx200_top10_load_and_clean()
    global df_economic_indicators
    df_economic_indicators = economic_indicators_load_and_clean()
    global js_news_dump
    js_news_dump = json_news_dump_load_and_clean()
    global list_of_stocks
    list_of_stocks = (list(df_client_details.columns[3:]))
    global yahoo_data
    yahoo_data = extractYahoo_equity(list_of_stocks[0] + '.AX', '2019-01-01', '2021-01-01', data_file_directory)

class station_2_features_engineering():
    def df_stocks(self):
        df_stocks = df_asx200_top10
        df_stocks = df_stocks.drop('AS51 Index', axis=1)
        # Calculate returns and covariance matrix of stock price
        rets = np.log(df_stocks.astype('float') / df_stocks.shift(1).astype('float'))

        rets.hist(bins=100, figsize=(9, 11))
        plt.show()
        rets.head()
        rets.mean() * 252
        rets.cov() * 252

        print(rets.describe())
        print(rets.mean() * 252)
        print(rets.cov() * 252)
        return df_stocks

    def economic_indicators (self):
        economic_data = df_economic_indicators
        economic_data = economic_data.T.reset_index().set_axis(economic_data.T.reset_index().iloc[0], axis=1).iloc[
                        1:].rename_axis(None, axis=1)
        # RESET DATA INPUTS INTO ADDITIONAL FEATURES
        # Features are for economic indicators
        features_considered = ['Quarterly Indicators', 'CPI (%q/q)', 'Real GDP Growth (%q/q, sa)',
                               'External Debt as a % of GDP']
        features = economic_data[features_considered]
        features.index = economic_data['Quarterly Indicators']
        print(features.head())
        features.plot(subplots=True)
        plt.show()
        # Consumer sentiment for economic indicators
        plt.plot(economic_data['Monthly Indicators'], economic_data['Consumer Sentiment Index'], label='Consumer Sentiment')
        plt.legend(loc='upper right')
        plt.xticks(rotation=45)
        plt.show()
        return features

    def brownian(self):
        # Preparation for Brownian motion and self-learning algorithms
        S0 = 100.
        r = 0.05
        sigma = 0.2
        T = 1.0
        M = 50
        I = 25000

        # call the function
        paths = gen_paths(S0, r, sigma, T, M, I)
        log_returns = np.log(paths[1:] / paths[0:-1])

        plt.plot(log_returns[:, :10])
        plt.grid(True)
        plt.xlabel('time steps')
        plt.ylabel('log returns')
        plt.savefig("plot2.png")

        # Plot for behaviour of normally distributed returns
        paths = gen_paths(100, -0.05, 0.38, 1.0, 50, 25000)
        log_returns = np.log(paths[1:] / paths[0:-1])
        normality_tests(log_returns.flatten())

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        ax1.hist(paths[-1], bins=30)
        ax1.grid(True)
        ax1.set_xlabel('index level')
        ax1.set_ylabel('frequency')
        ax1.set_title('regular data')
        ax2.hist(np.log(paths[-1]), bins=30)
        ax2.grid(True)
        ax2.set_xlabel('log index level')
        ax2.set_title('log data')
        plt.savefig("plog.png")
        ##

    def client_details (self):
        # Feature engineering for client details, creating a dataframe without risk profile and age group without
        # affecting the original dataframe so risk profile can be referenced later.
        df_clients = df_client_details
        df_clients.drop(['risk_profile', 'age_group'], axis=1, inplace=True)
        df_clients.set_index(df_clients.client_ID)

#Returns the Sharpe Ratio
def min_func_sharpe(weights, rets):
    return -statistics(weights,rets)[2]
def min_func_variance(weights, rets):
    return statistics(weights, rets)[1] ** 2

def min_func_port(weights, rets):
    return statistics(weights, rets)[1]



# ****** STATION #3 MODEL DESIGN ******
class Station3_modelDesign_and_Station4_implementation():

    def sentiment_analyser(self):
        #VADER
        df = js_news_dump
        # Instantiate the sentiment intensity analyzer
        vader = SentimentIntensityAnalyzer()

        # Iterate through the headlines and get the polarity scores using vader
        scores = df['Headline'].apply(vader.polarity_scores).tolist()
        # Convert the 'scores' list of dicts into a DataFrame
        scores_df = pd.DataFrame(scores)

        print(scores_df.head(20))

        # Join the DataFrames of the news and the list of dicts
        df = df.join(scores_df, rsuffix='_right')
        df.to_json(data_file_directory + 'vader_scores.json')
        print(df.head(10))

        # Group by date and ticker columns from scored_news and calculate the mean
        # Get a list of all column names excluding select columns
        cols_to_include = [col for col in df.columns if col not in ['Equity', 'Date/Time', 'Source', 'Headline']]

        mean_scores = df.groupby(['Equity'])[cols_to_include].mean()

        print(mean_scores)

        # Boxplot of time-series of sentiment score for each date
        mean_scores.plot(kind='box')
        plt.grid()
        plt.title('Box plot of sentiment score overtime')
        plt.show()

        # get the lastest month sentiment score as adjustment
        latest_month_score = mean_scores.tail(20).mean()
        latest_month_score.fillna(0, inplace=True)
        latest_month_score.plot(kind='bar')
        plt.title('Sentiment Score')
        plt.grid()
        plt.show()

        # adjusted weights - based on relative strength of sentiment score
        average_score = latest_month_score.mean()
        latest_month_score = (latest_month_score - average_score) / len(latest_month_score)
        latest_month_score.plot(kind='bar')
        plt.title('Adjustment on Portfolio Weights -- Sentiment Score')
        plt.grid()
        print(latest_month_score)
        plt.show()

        # NLP Sentiment analytics, code to run has been commented out

        # self.sentiment_training_model() #UNCOMMENT THIS CODE TO TRAIN DATA

        # PROCESS SENTIMENT DATA IN AGGREGATE TERMS
        df2 = pd.read_json(data_file_directory + 'sentiment_index.json')
        mean_scores = df.groupby(['Equity'])[cols_to_include].mean()

        print(mean_scores)

        # Boxplot of time-series of sentiment score for each date
        mean_scores.plot(kind='box')
        plt.grid()
        plt.title('Box plot of sentiment score overtime')
        plt.show()

        # get the lastest month sentiment score as adjustment
        latest_month_score = mean_scores.tail(20).mean()
        latest_month_score.fillna(0, inplace=True)
        latest_month_score.plot(kind='bar')
        plt.title('Sentiment Score')
        plt.grid()
        plt.show()

        # adjusted weights - based on relative strength of sentiment score
        average_score = latest_month_score.mean()
        latest_month_score = (latest_month_score - average_score) / len(latest_month_score)
        latest_month_score.plot(kind='bar')
        plt.title('Adjustment on Portfolio Weights -- Sentiment Score')
        plt.grid()
        print(latest_month_score)
        plt.show()
        print(df2.sort_index())
    def modern_portfolio_theory(self, df):
        df_stocks = pd.DataFrame()
        df_stocks = df
        # number of assets
        noa = len(df_stocks.columns)

        # Calculate returns and covariance matrix
        rets = np.log(df_stocks.astype('float') / df_stocks.shift(1).astype('float'))
        print(rets.mean() * 252)
        print(rets.cov() * 252)

        # Setup random porfolio weights
        weights = np.random.random(noa)
        weights /= np.sum(weights)

        # Derive Porfolio Returns & simulate various 2500x combinations
        print(np.sum(rets.mean() * weights) * 252)
        print(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

        prets = []
        pvols = []
        for p in range(2500):
            weights = np.random.random(noa)
            weights /= np.sum(weights)
            prets.append(np.sum(rets.mean() * weights) * 252)
            pvols.append(np.sqrt(np.dot(weights.T,
                                        np.dot(rets.cov() * 252, weights))))
        prets = np.array(prets)
        pvols = np.array(pvols)

        plt.figure(figsize=(8, 4))
        plt.scatter(pvols, prets, c=prets / pvols, marker='o')
        plt.grid(True)
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label='Sharpe ratio')
        plt.show()

        # Conditions for the optimization problem to be solved
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for x in range(noa))

        opts = sco.minimize(min_func_sharpe, (noa * [1. / noa, ]), rets, method='SLSQP',
                            bounds=bounds, constraints=cons)
        print("***Maximization of Sharpe Ratio***")
        print(opts['x'].round(3))
        print(statistics(opts['x'], rets).round(3))

        optv = sco.minimize(min_func_variance, noa * [1. / noa, ], rets, method='SLSQP',
                            bounds=bounds, constraints=cons)
        print("****Minimizing Variance***")
        print(optv['x'].round(3))
        print(statistics(optv['x'],rets).round(3))

        bonds = tuple((0, 1) for x in weights)
        trets = np.linspace(0.0, 0.25, 50)
        tvols = []
        for tret in trets:
            cons = ({'type': 'eq', 'fun': lambda x: statistics(x,rets)[0] - tret},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            res = sco.minimize(min_func_port, noa * [1. / noa, ], rets, method='SLSQP',
                               bounds=bonds, constraints=cons)
            tvols.append(res['fun'])
        tvols = np.array(tvols)

        plt.figure(figsize=(8, 4))
        plt.scatter(pvols, prets,
                    c=prets / pvols, marker='o')
        # random portfolio composition
        plt.scatter(tvols, trets,
                    c=trets / tvols, marker='x')
        # efficient frontier
        plt.plot(statistics(opts['x'], rets)[1], statistics(opts['x'], rets)[0],
                 'r*', markersize=15.0)
        # portfolio with highest Sharpe ratio
        plt.plot(statistics(optv['x'], rets)[1], statistics(optv['x'], rets)[0],
                 'y*', markersize=15.0)
        # minimum variance portfolio
        plt.grid(True)
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label='Sharpe ratio')
        plt.show()

        ind = np.argmin(tvols)
        evols = tvols[ind:]#
        erets = trets[ind:]#
        tck = sci.splrep(evols, erets)
        # returns an array with 3 numbers
        opt = sco.fsolve(equations, [0.01, 0.5, 0.15], args = (tck,))

        plt.figure(figsize=(8, 4))
        plt.scatter(pvols, prets,
                    c=(prets - 0.01) / pvols, marker='o')
        # random portfolio composition
        plt.plot(evols, erets, 'g', lw=4.0)
        # efficient frontier
        cx = np.linspace(0.0, 0.3)
        plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
        # capital market line
        plt.plot(opt[2], splines_approx(opt[2], tck), 'r*', markersize=15.0)
        plt.grid(True)
        plt.axhline(0, color='k', ls='--', lw=2.0)
        plt.axvline(0, color='k', ls='--', lw=2.0)
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label='Sharpe ratio')
        plt.show()

        cons = ({'type': 'eq', 'fun': lambda x: statistics(x,rets)[0] - splines_approx(opt[2],tck)},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = sco.minimize(min_func_port, noa * [1. / noa, ], rets, method='SLSQP',
                           bounds=bounds, constraints=cons)

        print("***Optimal Tangent Portfolio***")
        print(res['x'].round(3))

        # END OF MODERN PORTFOLIO THEORY MODEL

        # TensorFlow Predictive analytics
        # DATA BLOCK: load the dataset and select
        column_to_analyse = 0
        df = df_stocks.iloc[:,[column_to_analyse]]
        stocks_data = df.values
        stocks_data = stocks_data.astype('float32')
        plt.plot(stocks_data)
        plt.show()

        # TRANSFORMATION BLOCK: normalize data be Mean and SD
        scaler = MinMaxScaler(feature_range=(0, 1))
        stocks_data = scaler.fit_transform(stocks_data)

        # DATA CONTROL BLOCK: split data into train and test components manually
        train_size = int(len(stocks_data) * 0.7)
        test_size = len(stocks_data) - train_size
        train, test = stocks_data[0:train_size, :], stocks_data[train_size:len(stocks_data), :]
        # trigger X,y data build
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        # reshape input to be [samples, time steps, features] used by Keras for feeding
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        # MODEL BLOCK: setup model parameters and inputs for processing
        batch_size = 1
        model = Sequential()
        model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(100):
            model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
            model.reset_states()
        trainPredict = model.predict(trainX, batch_size=batch_size)
        model.reset_states()
        testPredict = model.predict(testX, batch_size=batch_size)

        # invert scaled results back to norm
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(stocks_data)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(stocks_data)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(stocks_data) - 1, :] = testPredict

        #future_predictions = []
        last_values = stocks_data[-look_back:]
        last_values = np.reshape(last_values, (1, look_back, 1))
        future_predictions = []

        for i in range(50):
            prediction = model.predict(last_values, batch_size=batch_size)
            future_predictions.append(prediction[0,0])
            last_values = np.append(last_values[0,1:], prediction)
            last_values = np.reshape(last_values, (1, look_back, 1))
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        # Create a new array that includes the original data and the future predictions
        extended_stocks_data = np.append(stocks_data, future_predictions)

        # Create a new plot for the extended data
        extendedPlot = np.empty_like(extended_stocks_data)
        extendedPlot[:] = np.nan
        extendedPlot[len(stocks_data):len(extended_stocks_data)] = future_predictions.flatten()
        plt.plot(extendedPlot, label='Future Predict Plot')

        # plot baseline and predictions
        plt.plot(scaler.inverse_transform(stocks_data), label = 'Historical Plot')
        plt.plot(trainPredictPlot, label = 'train Predict Plot')
        plt.plot(testPredictPlot, label = 'test Predict Plot')
        plt.title("Predictive Analytics Plot " + df_stocks.columns[column_to_analyse])
        plt.gca().legend()

        plt.show()

        #ARIMA
        tt_ratio = 0.70  # Train to Test ratio
        Station3_modelDesign_and_Station4_implementation.ARIMA(rets.iloc[:,0], tt_ratio)

        #

    def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
        """
            series - dataframe with timeseries
            window - rolling window size
            plot_intervals - show confidence intervals
            plot_anomalies - show anomalies
        """
        rolling_mean = series.rolling(window=window).mean()
        plt.title("Moving average with window size = {}".format(window))
        plt.plot(rolling_mean, "g", label="Rolling mean trend")

        # Plot confidence intervals for smoothed values
        if plot_intervals:
            mae = mean_absolute_error(series[window:], rolling_mean[window:])
            deviation = np.std(series[window:] - rolling_mean[window:])
            lower_bond = rolling_mean - (mae + scale * deviation)
            upper_bond = rolling_mean + (mae + scale * deviation)
            plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond, "r--")

            # Having the intervals, find abnormal values
            if plot_anomalies:
                anomalies = pd.DataFrame(index=series.index, columns=series.columns)
                anomalies[series<lower_bond] = series[series<lower_bond]
                anomalies[series>upper_bond] = series[series>upper_bond]
                plt.plot(anomalies, "ro", markersize=10)

        plt.plot(series[window:], label="Actual values")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()

    # ARIMA
    def ARIMA(data, tt_ratio):
        X = data.values
        size = int(len(X) * tt_ratio)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order = (5,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('progress:%',round(100*(t/len(test))),'\t predicted=%f, expected=%f' % (yhat, obs), end="\r")
        error = mean_squared_error(test, predictions)
        print('\n Test MSE: %.3f' % error)

        preds = np.append(train, predictions)
        plt.plot(list(preds), color='green', linewidth=3, label="Predicted Data")
        plt.plot(list(data), color='blue', linewidth=2, label="Original Data")
        plt.axvline(x=int(len(data)*tt_ratio)-1, linewidth=5, color='red')
        plt.legend()
        plt.show()

    # LSTM RNN
    def LSTM_RNN(x_train, x_test, y_train, y_test):
        model = Sequential()
        model.add(LSTM(units=32, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=32))
        model.add(Dense(units = 1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model.summary()
        model.fit(x_train, y_train, epochs = 16, batch_size = 32)

        predictions = model.predict(x_test)
        plt.plot(y_test.values[:4320], color='blue', label='Original Usage')
        plt.plot(predictions[:,0][:4320] , color='red', label='Predicted Usage')
        plt.title('Energy Usage Prediction')
        plt.xlabel('Date')
        plt.ylabel('kW')
        plt.legend()
        plt.show()

    def sentiment_training_model(self):
        df = pd.read_json(data_file_directory + 'news_dump.json')
        dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                                  as_supervised=True)
        train_dataset, test_dataset = dataset['train'], dataset['test']
        encoder = info.features['text'].encoder

        BUFFER_SIZE = 10000
        BATCH_SIZE = 64

        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.padded_batch(BATCH_SIZE, ([-1],[]))
        test_dataset = test_dataset.padded_batch(BATCH_SIZE, ([-1],[]))

        # Model Definition
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(encoder.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])
        history = model.fit(train_dataset, epochs=2,
                            validation_data=test_dataset,
                            validation_steps=30)
        test_loss, test_acc = model.evaluate(test_dataset)

        print('Test Loss: {}'.format(test_loss))
        print('Test Accuracy: {}'.format(test_acc))
        # HERE YOU RUN TRAINED MODEL (TRAINED ON MOVIES) ACROSS YOUR NEWS SNIPPETS
        sentiment_index = []
        for row in df['Headline']:
            sentiment_index.append(sample_predict(row, pad=True)[0])

        # RESULTS SENTIMENT INDEX AND MODEL ACCURACY MEASURES
        print(sentiment_index)
        plot_graphs(history, 'accuracy')
        plot_graphs(history, 'loss')
        df1 = pd.DataFrame(sentiment_index)
        df1.to_json(data_file_directory + 'sentiment_index.json')

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)



# Loads Json news dump file
def json_news_dump_load_and_clean():
    file_name = 'news_dump.json'
    directory = data_file_directory + file_name
    js = pd.read_json(directory)
    return js

# Loads Economic indicators Excel and cleans it by removing unnecessary rows and columns
def economic_indicators_load_and_clean():
    file_name = 'Economic_Indicators.xlsx'
    directory = data_file_directory + file_name
    # Open excel dropping the first 3 rows (as they provide no data)
    df = pd.read_excel(directory, skiprows=[0, 1, 2])
    # remove first two columns as they have incomplete data
    df = df.drop(df.columns[[1, 2]], axis=1)

    # Get information of dataframe which can be displayed later in orange later but for now terminal
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns')
    return df

# Loads Economic indicators Excel and cleans it double checking calculations make sense as well as removes unnecessary wordings
def client_details_load_and_clean():
    file_name = 'Client_Details.xlsx'
    directory = data_file_directory + file_name
    df = pd.read_excel(directory, sheet_name="Data")

    # Get information of dataframe which can be displayed later in orange later but for now terminal
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns')

    # Rename columns by replacing AT equity with ''
    df.columns = [col.replace(' AT Equity', '') for col in df.columns]

    # Ensure that equity is equal to 1
    if df.iloc[:, 3:].sum(axis=1).le(0.9999).all() | df.iloc[:, 3:].sum(axis=1).ge(1.0001).all():
        Exception(" client's equity does not equal to 1 ")
        raise IOError

    return df

# Loads asx200top10 Excel and cleans it by making it into a better format for station 2 and then also performs feature
# engineering by removing all columns other than last price (PX)
def asx200_top10_load_and_clean():
    file_name = 'ASX200top10.xlsx'
    directory = data_file_directory + file_name
    df = pd.read_excel(directory, sheet_name="Bloomberg raw")
    # Get information of dataframe which can be displayed later in orange later but for now terminal
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns')

    # Get only last px to be used for calculations
    df = df.shift(periods=-1, axis=1)
    df = df.loc[0:, ~df.columns.str.contains('^Unnamed')]
    # Rename columns by replacing AT equity with ''
    df.columns = [col.replace(' AT Equity', '') for col in df.columns]
    df = df.drop([0])
    return df

# Function allows for assets to get new data from yahoo finance in order to use relevant up-to-date data
def extractYahoo_equity(asset, start_date, end_date, directory):
    # call API routines specific to an asset
    equity = yf.download(asset,
                         start=start_date,
                         end=end_date,
                         progress=False,
                         auto_adjust=False)
    # drop extracts into ET fileDB
    equity.to_csv(directory + str(asset) + '.prices.csv')
    equity.to_json(directory + str(asset) + 'prices.json')
    return equity

# For brownian simulations
def gen_paths(S0, r, sigma, T, M, I):
    ''' Generate Monte Carlo paths for geometric Brownian motion.
    Reference :YH Ch.11 pp.309
    Parameters
    ==========
    S0 : float
        initial stock/index value
    r : float
        constant short rate
    sigma : float
        constant volatility
    T : float
        final time horizon
    M : int
        number of time steps/intervals
    I : int
        number of paths to be simulated

    Returns
    =======
    paths : ndarray, shape (M + 1, I)
        simulated paths given the parameters
    '''
    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)
    return paths

# For graph and distribution analysis
def normality_tests(arr):
    print("Skew of data set  %14.3f" % scs.skew(arr))
    print("Skew test p-value %14.3f" % scs.skewtest(arr)[1])
    print("Kurt of data set  %14.3f" % scs.kurtosis(arr))
    print("Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1])
    print("Norm test p-value %14.3f" % scs.normaltest(arr)[1])

#Helper Functions
def statistics(weights, rets):
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])
def splines_approx(x, tck):
    ''' Efficient frontier function (splines approximation). '''
    return sci.splev(x, tck, der=0)
def efficient_frontier_first_dirivative(x, tck):
    ''' First derivative of efficient frontier function. '''
    return sci.splev(x, tck, der=1)

def equations(p, tck, rf = 0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - splines_approx(p[2], tck)
    eq3 = p[1] - efficient_frontier_first_dirivative(p[2], tck)
    return eq1, eq2, eq3

# CORE OPERATIONAL FUNCTIONS
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt.show()

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/2, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/2, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sample_pred_text, pad, model):
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                              as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    encoder = info.features['text'].encoder
    encoded_sample_pred_text = encoder.encode(sample_pred_text)
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
    return (predictions)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()