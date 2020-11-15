# identify correlations between independent variables
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression
import seaborn as sns
from sklearn import preprocessing
from datetime import datetime, date
import matplotlib.pyplot as plt


class WildfireAnalysis:
    def __init__(self, file_name):
        # df = pd.read_csv('wildfire_weather_aggregated_data.csv')
        self.df = pd.read_csv(file_name)

        self.X = None
        self.y = None
        self.cols = ['DISCOVERY_DATE', 'distance', 'air_temperature', 'relative_humidity', 'wind_speed', 'precipitation', 'AQI_AGGREGATE']
        self.clean_data()

    def clean_data(self):
        """
        Cleans data, handles NaN values, converts date into number of days since start of year
        :return:
        """

        def number_of_days(x):
            return (x - datetime(2014, 12, 31)).days

        self.df = self.df[['FOD_ID', 'STAT_CAUSE_DESCR', 'DISCOVERY_DATE', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'distance',
                 'air_temperature', 'relative_humidity', 'wind_speed', 'precipitation', 'AQI_AGGREGATE']]

        # Fill missing values
        self.df['precipitation'] = self.df['precipitation'].fillna(0)
        self.df['wind_speed'] = self.df['wind_speed'].fillna(0)
        self.df['relative_humidity'] = self.df['relative_humidity'].fillna(50)

        # Drop any rows have NaN values
        self.df = self.df.dropna()

        # Convert date to number of days since start of year
        self.df['DISCOVERY_DATE'] = pd.to_datetime(self.df['DISCOVERY_DATE'])
        self.df['DISCOVERY_DATE'] = self.df['DISCOVERY_DATE'].apply(number_of_days)

        # Performs One-hot encoding of fire cause
        # df['STAT_CAUSE_DESCR'] = df['STAT_CAUSE_DESCR'].astype('category')
        # cause_one_hot = pd.get_dummies(df['STAT_CAUSE_DESCR'], prefix='cause')
        # df = pd.concat([df, cause_one_hot], sort=False, axis=1)

        self.normalize()

    def plot_all_features_vs_y(self):
        for col in self.X.columns:
            self.scatter_plot(self.X[col], self.y, col, 'FIRE_SIZE')

    def plot_all_histograms(self):
        pass

    def scatter_plot(self, x, y, x_label, y_label):
        plt.scatter(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def normalize(self):
        self.X = self.df[self.cols]
        self.y = self.df['FIRE_SIZE']

        # Normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        self.X_normalized = pd.DataFrame(min_max_scaler.fit_transform(self.X), columns=self.cols)

    def select_k_best_features(self, score_func):
        best_features = SelectKBest(score_func=score_func, k='all')
        fit = best_features.fit(self.X_normalized, self.y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(self.X.columns)

        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
        print(featureScores)

    def correlation_heatmap(self):
        corrmat = self.df.corr()
        top_corr_features = corrmat.index
        plt.figure(figsize=(20, 20))
        # plot heat map
        g = sns.heatmap(self.df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
        plt.show()


if __name__=='__main__':
    wildfire_analysis = WildfireAnalysis('../data/combined.csv')
    wildfire_analysis.select_k_best_features(f_regression)
    wildfire_analysis.select_k_best_features(mutual_info_regression)
    wildfire_analysis.plot_all_features_vs_y()
    wildfire_analysis.correlation_heatmap()