# Wildfire Analysis


## Team
Hardik Mahipal Surana
Pranshu Dave
Tanvi Pisat

## Installation
1. Setup a virtual environment
```bash
virtualenv -p python3 venv
source venv/bin/activate
```

2. Install project dependencies
```bash
pip3 install -r requirements.txt
```

3. Run regression models
```bash
python src/regression_models.py data/combined.csv
```
## Dataset Features

| Column Name       | Description                                              |
| ----------------- | -------------------------------------------------------- |
| FOD_ID            | global unique identifier                                 |
| FIRE_NAME         | name of the incident                                     |
| DISCOVERY_DATE    | date when the fire was discovered/confirmed to exist     |
| CONT_DATE         | date when the fire was declared contained/controlled     |
| STAT_CAUSE_DESCR  | Description of the (statistical) cause of the fire       |
| FIRE_SIZE         | Estimate of acres within the final perimeter of the fire |
| FIRE_SIZE_CLASS   | Code for fire size based on the number of acres burned   |
| LATITUDE          | in decimal degrees                                       |
| LONGITUDE         | in decimal degrees                                       |
| STATE             | Two-letter code where fire occured                       |
| distance          | distance of sensors from fire                            |
| date_time         | when the data was collected                              |
| air_temperature   | air temperature (1 day reading)                          |
| relative_humidity | relative humidity (1 day reading)                        |
| wind_speed        | wind speed (1 day reading)                               |
| precipitation     | precipitation (1 day reading)                            |

## Other Features
Air Quality
Air Quality Class/Category
Air pressure

FFMC index
DMC index
DC index
ISI index
Buildup index

vegetation
fuel

altitude/height above sea level

## Potential Problems
1. Heteroscedasticity - When dependent variable's variability is not equal across values of an independent variable
2. Multicollinearity - When the independent variables are highly correlated to each other
3. Underfitting - high bias
4. Overfitting - high variance


## Notes
1. date columns in the database are in julian days
2. FIRE_SIZE_CLASS - 
    A=greater than 0 but less than or equal to 0.25 acres, 
    B=0.26-9.9 acres, 
    C=10.0-99.9 acres, 
    D=100-299 acres, 
    E=300 to 999 acres, 
    F=1000 to 4999 acres,
    G=5000+ acres

## Sources
1. [Kaggle](https://www.kaggle.com/rtatman/188-million-us-wildfires)
2. AirNow API
3. Mesowest API
