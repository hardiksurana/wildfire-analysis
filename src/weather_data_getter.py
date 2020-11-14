import os
from MesoPy import Meso
import pandas as pd
from datetime import timedelta
from src.helper import haversine
from scipy.spatial.distance import cdist

class WeatherDataGetter:
    def __init__(self, API_KEY, station_metadata_file, wildfire_occurences_file, wildfire_weather_file,
                 year_threshold, state_specified=None, station_radius_threshold=10):
        # Init meso
        self.meso = Meso(token=API_KEY)

        self.station_metadata = None
        # If file is not specified, retrieve data from API
        if os.path.isfile(station_metadata_file):
            self.station_metadata = pd.read_csv(station_metadata_file)
        else:
            self.station_metadata = self.get_station_metadata(station_metadata_file)

        # Class Vairables
        self.wildfires_df = pd.read_csv(wildfire_occurences_file)
        self.year_thresold = year_threshold
        self.state_specified = state_specified
        self.wildfire_weather_file = wildfire_weather_file
        self.station_radius_threshold = station_radius_threshold

        self.preprocess_wildfires()

    def get_5_nearest(self, lat, lon, n):
        """
        Takes position in the form of lat, lon and returns n nearest mesowest stations
        :param lat: latitude
        :param lon: longitude
        :param n: number of stations
        :return: n nearest stations
        """

        all_points = self.station_metadata[['latitude', 'longitude']]
        self.station_metadata['distance'] = cdist([(lat, lon)], all_points).T

        n_smallest = self.station_metadata.nsmallest(n=n, columns='distance')
        n_smallest['miles'] = [haversine(lon, lat, row['longitude'], row['latitude']) for _, row in
                               n_smallest.iterrows()]

        n_smallest = n_smallest[n_smallest['miles'] <= self.station_radius_threshold]

        return n_smallest['STID'].tolist()

    def preprocess_wildfires(self):
        """
        Preprocesses wildfires database, filtering by year_threshold and state (if specified)
        :return: None
        """
        # Convert dates to datetimes. Add an year column
        self.wildfires_df['DISCOVERY_DATE'] = pd.to_datetime(self.wildfires_df['DISCOVERY_DATE'])
        self.wildfires_df['CONT_DATE'] = pd.to_datetime(self.wildfires_df['CONT_DATE'])
        self.wildfires_df['year'] = pd.DatetimeIndex(self.wildfires_df['DISCOVERY_DATE']).year

        # Filter to all records greater than an year and sort row in descending order by date
        self.wildfires_df = self.wildfires_df[self.wildfires_df['DISCOVERY_DATE'].dt.year
                                         >= self.year_thresold].sort_values(by=['DISCOVERY_DATE'], ascending=False)

        # Filter to only a particular state
        if self.state_specified:
            self.wildfires_df = self.wildfires_df[self.wildfires_df['STATE'] == self.state_specified]


    def get_station_metadata(self, file_name):
        """
        Gets all stations metadata from mesowest API and saves it in a file
        :param file_name: file where station data should be stored
        :return: None
        """
        vars = ['air_temp', 'relative_humidity', 'wind_speed', 'precip_accum']
        metadata = self.meso.metadata(country='us', status='ACTIVE', var=vars, obrange='20110101, 20160101')

        out = []
        for i in range(len(metadata['STATION'])):
            try:
                out.append([metadata['STATION'][i]['STID'], metadata['STATION'][i]['LATITUDE'],
                            metadata['STATION'][i]['LONGITUDE']])
            except:
                pass
        df = pd.DataFrame(out, columns=['STID', 'latitude', 'longitude'])
        df.to_csv(file_name)


    def get_weather_data(self):
        """
        Purpose is to get weather data from nearby stations of a fire
        :return: None
        """

        # TODO: Add last_n_days query functionality
        radius = '10'
        last_n_days = 7

        cnt = 0 # Keeps cnt of rows. Used to periodically write rows to file
        stations_data = [] # Stores data for a few stations till it is written to file
        header = True # For dataframe header
        write_rows_threshold = 100

        columns = ['FOD_ID', 'STID', 'distance', 'date_time', 'air_temperature', 'relative_humidity', 'wind_speed',
                   'precipitation']

        for index, row in self.wildfires_df.iterrows():
            if cnt % write_rows_threshold == 0: # Every write_rows_threshold rows, write to file
                df = pd.DataFrame(stations_data, columns=columns)
                with open(self.wildfire_weather_file, 'a') as f:
                    df.to_csv(f, header=header)
                    header = False
                stations_data = []

            #
            start = (row['DISCOVERY_DATE'] - timedelta(days=0)).replace(hour=12, minute=0).strftime("%Y%m%d%H%M")
            lat = row['LATITUDE']
            lon = row['LONGITUDE']
            days = [start]

            station_data = self.get_weather_data_api(lat, lon, radius, row['FOD_ID'], days,
                                                     stids=self.get_5_nearest(lat=float(lat),
                                                                lon=float(lon), n=5))
            print(cnt)
            if station_data:
                stations_data.extend(station_data)
            cnt += 1

    def get_weather_data_api(self, lat, lon, radius, wildfire_ID, days, stids):
        """
        Queries mesowest API for given lat, lon and radius. Given nearby stations
        :param lat: optional, if stid provided
        :param lon: optional, if stid provided
        :param radius: optional, if stid provided
        :param wildfire_ID:
        :param days: TODO: for n days. Currently will only query for a single day
        :param stids: STIDs
        :return: None
        """
        radius_param = str(lat) + ',' + str(lon) + ',' + str(radius)
        vars = ['air_temp', 'relative_humidity', 'wind_speed', 'precip_accum']

        # Get data for a locations with sepcified radius
        data = []
        try:
            for day in days:
                day_data = self.meso.attime(radius=radius_param, attime=day, within=60, stid=stids, vars=vars)
                if not day_data:
                    return None

                data.append(day_data)
        except Exception as e:
            print(e)
            return

        if not data:
            return

        # Query data to generate station_df which would have the following columns:
        # STID, Date_time, Var1, Var2...
        # We could omit all stations which do not have all the values
        # Currently, keeping all stations irrespective of their missing values
        # Each occurence of a date_time val is supposed to be a row.
        # Here since, we are only querying attime, we will have only 1 date_time field
        vars_data = ['date_time', 'air_temp_value_1', 'relative_humidity_value_1', 'wind_speed_value_1',
                     'precip_accum_value_1']

        stations = [station['STID'] for station in data[0]['STATION']]
        station_data = []

        # We are storing station ID from first day's data.
        # Subsequently querying other days data for these stations and storing them
        # TODO: Optimize below for loop
        for station_id in stations:
            for day_idx in range(len(data)):
                if data[day_idx]:
                    temp = []
                    for station in data[day_idx]['STATION']:
                        if station['STID'] == station_id:
                            row = [wildfire_ID, station['STID'], station['DISTANCE']]
                            for var in vars_data:
                                if var == 'date_time':
                                    row.append(days[day_idx])
                                elif var in station['OBSERVATIONS']:
                                    row.append(station['OBSERVATIONS'][var]['value'])
                                else:
                                    row.append('')
                            temp.append(row)
                    station_data.extend(temp)

        return station_data
