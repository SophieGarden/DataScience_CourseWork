
# coding: utf-8

# In[81]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
import datetime
from ggplot import *


# In[9]:


pd.set_option('display.max_columns', 100)


# In[2]:


df_tw = pd.read_csv('turnstile_data_master_with_weather.csv')

y1 = df_tw.loc[(df_tw['rain'] == 0) & (df_tw['ENTRIESn_hourly'] < 6000),['ENTRIESn_hourly']] # your code here to plot a historgram for hourly entries when it is not raining
y2 = df_tw.loc[(df_tw['rain'] == 1) & (df_tw['ENTRIESn_hourly'] < 6000),['ENTRIESn_hourly']] # your code here to plot a historgram for hourly entries when it is not raining
#plt.hist(y1,30, label=['No Rain', 'Rain'])
#plt.title("Histogram of ENTRIESn_hourly")


# In[33]:


df_tw.head()


# In[76]:


df_tw['DATEi'] = pd.to_datetime(df_tw.DATEn)
df_tw['day_of_week'] = df_tw.DATEi.dt.weekday #Monday is 0 and Sunday is 6
df_tw['day_of_week_name'] = df_tw.DATEi.dt.weekday_name


# In[258]:


print(list(df_tw.columns).index('UNIT'),list(df_tw.columns).index('DATEn'),list(df_tw.columns).index('TIMEn'))
print(list(df_tw.columns).index('ENTRIESn_hourly'))
print(list(df_tw.columns).index('fog'),list(df_tw.columns).index('rain'))
print(len(list(df_tw.columns)))


# In[243]:


df_tw.columns


# In[20]:


type(y1.values)


# In[44]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(y1.values,bins = 30)
ax.hist(y2.values,bins = 30)


# In[56]:


plt.figure()
plt.hist([y1.values,y2.values], bins=30, color = ['lightpink','skyblue'],alpha=0.7, ec='None',  stacked=True, label = ['No Rain', 'Rain'])
plt.xlabel('ENTRIESn_hourly')
plt.ylabel('Frenquency')
plt.title('Histogram of ENTRIESn_hourly')
plt.legend(frameon=False)


# In[63]:


y1_all = df_tw.loc[(df_tw['rain'] == 0) ,['ENTRIESn_hourly']] # your code here to plot a historgram for hourly entries when it is not raining
y2_all = df_tw.loc[(df_tw['rain'] == 1) ,['ENTRIESn_hourly']] # your code here to plot a historgram for hourly entries when it is not raining
with_rain_mean = np.mean(y2_all.values)
without_rain_mean = np.mean(y1_all.values)
(U, p) = scipy.stats.mannwhitneyu(y1_all.values,y2_all.values)
print(U,p)


# In[66]:


df_tw.describe()


# In[77]:


plt.subplot(1,3,1)
plt.plot(df_tw.ENTRIESn_hourly, df_tw.day_of_week_name,'.')
plt.subplot(1,3,2)
plt.plot(df_tw.ENTRIESn_hourly, df_tw.Hour,'.')
plt.subplot(1,3,3)
plt.plot(df_tw.ENTRIESn_hourly, df_tw.meantempi,'.')


# In[41]:


plt.figure(figsize=(15,2))
plt.subplot(1,3,1)
plt.plot(df_tw.maxtempi, df_tw.ENTRIESn_hourly,'.')
plt.subplot(1,3,2)
plt.plot(df_tw.mintempi, df_tw.ENTRIESn_hourly,'.')
plt.subplot(1,3,3)
plt.plot(df_tw.meantempi, df_tw.ENTRIESn_hourly,'.')


# In[42]:


plt.figure(figsize=(15,2))
plt.subplot(1,3,1)
plt.plot(df_tw.maxpressurei, df_tw.ENTRIESn_hourly,'.')
plt.subplot(1,3,2)
plt.plot(df_tw.minpressurei, df_tw.ENTRIESn_hourly,'.')
plt.subplot(1,3,3)
plt.plot(df_tw.meanpressurei, df_tw.ENTRIESn_hourly,'.')


# In[43]:


plt.figure(figsize=(15,2))
plt.subplot(1,3,1)
plt.plot(df_tw.maxdewpti, df_tw.ENTRIESn_hourly,'.')
plt.subplot(1,3,2)
plt.plot(df_tw.mindewpti, df_tw.ENTRIESn_hourly,'.')
plt.subplot(1,3,3)
plt.plot(df_tw.meandewpti, df_tw.ENTRIESn_hourly,'.')


# In[45]:


plt.figure(figsize=(15,2))
plt.subplot(1,3,1)
plt.plot(df_tw.rain, df_tw.ENTRIESn_hourly,'.')
plt.subplot(1,3,2)
plt.plot(df_tw.fog, df_tw.ENTRIESn_hourly,'.')
plt.subplot(1,3,3)
plt.plot(df_tw.precipi, df_tw.ENTRIESn_hourly,'.')


# In[ ]:


# predictions of ridership per hour


# In[67]:


import numpy as np
import pandas
from ggplot import *
import scipy
import matplotlib.pyplot as plt

"""
In this question, you need to:
1) implement the compute_cost() and gradient_descent() procedures
2) Select features (in the predictions procedure) and make predictions.

"""

def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    mu = df.mean()
    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " +                          "not be normalized. Please do not include features with only a single value " +                          "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma

def compute_cost(features, values, theta):
    """
    Compute the cost function given a set of features / values, 
    and the values for our thetas.
    """
    
    # your code here
    m = len(values)
    sum_squared_errors = sum(np.square( np.dot(features, theta) - values))
    cost = sum_squared_errors/(2*m)
    
    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """
    
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        # your code here
        pre_y = np.dot(features, theta)
        theta += alpha/m * np.dot((values - pre_y), features)
        cost = compute_cost(features, values, theta)
        cost_history.append(cost)
    return theta, pandas.Series(cost_history)

def predictions(dataframe, alpha=0.1, num_iterations=30):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.
    
    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
    
    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe. 
    We recommend that you don't use the EXITSn_hourly feature as an input to the 
    linear model because we cannot use it as a predictor: we cannot use exits 
    counts as a way to predict entry counts. 
    
    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~15%) of the data contained in 
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with 
    this computer on your own computer, locally. 
    
    
    If you'd like to view a plot of your cost history, uncomment the call to 
    plot_cost_history below. The slowdown from plotting is significant, so if you 
    are timing out, the first thing to do is to comment out the plot command again.
    
    If you receive a "server has encountered an error" message, that means you are 
    hitting the 30-second limit that's placed on running your program. Try using a 
    smaller number for num_iterations if that's the case.
    
    If you are using your own algorithm/models, see if you can optimize your code so 
    that it runs faster.
    '''
    # Select Features (try different features!)
    #features = dataframe[['rain', 'precipi', 'Hour', 'meantempi','fog', 'meandewpti', 'meanpressurei']]
    features = dataframe[['Hour', 'maxpressurei', 'maxdewpti', 'mindewpti',                           'minpressurei', 'meandewpti', 'meanpressurei', 'fog',                          'rain', 'meanwindspdi', 'mintempi', 'meantempi', 'maxtempi', 'precipi']]
    
    dummy_day_of_week = pandas.get_dummies(dataframe['day_of_week'], prefix = 'dow')
    features = features.join(dummy_day_of_week)
    
    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)

    # Set values for alpha, number of iterations.
    #change to be input parameters
    #alpha = 0.1 # please feel free to change this value
    #num_iterations = 50 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    #plot = None
    # -------------------------------------------------
    # Uncomment the next line to see your cost history
    # -------------------------------------------------
    
    plot = plot_cost_history(alpha, cost_history)
    # 
    # Please note, there is a possibility that plotting
    # this in addition to your calculation will exceed 
    # the 30 second limit on the compute servers.
    
    predictions = np.dot(features_array, theta_gradient_descent)
    return predictions, plot

def plot_cost_history(alpha, cost_history):
    """
       This function is for viewing the plot of your cost history.
       You can run it by uncommenting this

           plot_cost_history(alpha, cost_history) 

       call in predictions.

       If you want to run this locally, you should print the return value
       from this function.
    """

    #cost_history = np.array(cost_history)
    #print(type(cost_history))
    
    cost_df = pandas.DataFrame({'Cost_History': cost_history,'Iteration': range(len(cost_history))})
    
    return (ggplot(cost_df, aes('Iteration', 'Cost_History')) +           geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha ))
    

def plot_residuals(values, predictions):
    '''
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).
    Try different binwidths for your histogram.

    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:

    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''
    
    plt.figure()
    plt.hist(values - predictions)
    return plt

def compute_r_squared(values, predictions):
    '''
    In exercise 5, we calculated the R^2 value for you. But why don't you try and
    and calculate the R^2 value yourself.
    
    Given a list of original data points, and also a list of predicted data points,
    write a function that will compute and return the coefficient of determination (R^2)
    for this data.  numpy.mean() and numpy.sum() might both be useful here, but
    not necessary.

    Documentation about numpy.mean() and numpy.sum() below:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    '''
    
    # your code here
    
    r_squared = 1- ((values - predictions)**2).sum()/((values-np.mean(values))**2).sum()
    return r_squared


# In[48]:


values = np.array(df_tw['ENTRIESn_hourly'])


# In[68]:


pre, plot = predictions(df_tw, alpha = .1, num_iterations=40)


# In[69]:


plot


# In[70]:


plot_residuals(values, pre)


# In[71]:


compute_r_squared(values, pre)


# In[73]:


import statsmodels.api as sm
features = df_tw[['Hour', 'day_of_week','maxpressurei', 'maxdewpti', 'mindewpti',                           'minpressurei', 'meandewpti', 'meanpressurei', 'fog',                          'rain', 'meanwindspdi', 'mintempi', 'meantempi', 'maxtempi', 'precipi']]
# features = df_tw[['Hour', 'meandewpti', 'meanpressurei', 'fog',\
#                           'rain', 'meanwindspdi', 'meantempi','maxtempi', 'precipi']]

dummy_day_of_week = pandas.get_dummies(df_tw['day_of_week'], prefix = 'dow')
features = features.join(dummy_day_of_week)
            
    # Add UNIT to features using dummy variables
dummy_units = pandas.get_dummies(df_tw['UNIT'], prefix='unit')
features = features.join(dummy_units)
features = sm.add_constant(features)   
    # Values
values = df_tw['ENTRIESn_hourly']
    
model = sm.OLS(values,features)
results = model.fit()


# In[74]:


results.rsquared


# In[183]:


df_sum = df_tw[['day_of_week','ENTRIESn_hourly']].groupby('day_of_week', as_index=False).agg(np.mean)
#df_sum['day_of_week'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
print(df_sum)


# In[158]:


df_sum.columns


# In[178]:


gg = ggplot(df_sum, aes(x = 'day_of_week', weight = 'ENTRIESn_hourly'))     + geom_bar(facecolor='blue', fill='blue', alpha=0.3)    + ggtitle('Ridership by day of weel') + xlab('Day') + ylab('Total Entries')    + scale_x_discrete(labels=(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))
print(gg)


# In[140]:


df_sum = df_tw[['day_of_week_name','ENTRIESn_hourly']].groupby('day_of_week_name', as_index=False).agg(np.mean)
df_sum


# In[148]:


weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
mapping = {day: i for i, day in enumerate(weekdays)}
key = df_sum['day_of_week_name'].map(mapping)
df_sum = df_sum.iloc[key.argsort()]
df_sum


# In[220]:


df_tw['Increase_hourly'] = df_tw['ENTRIESn_hourly']-df_tw['EXITSn_hourly']
df_sum_h = df_tw[['Hour','ENTRIESn_hourly','EXITSn_hourly']].groupby('Hour', as_index = False).agg(np.mean)
#df_sum_h

gg_h = ggplot(df_sum_h, aes('Hour', weight = 'ENTRIESn_hourly')) +geom_bar(facecolor = 'blue', fill = 'blue', alpha = .5)
print(gg_h)


# In[254]:


df_sum_h = df_tw[['Hour','Increase_hourly']].groupby('Hour', as_index = False).agg(np.mean)
#df_sum_h

gg_h = ggplot(df_sum_h, aes('Hour', weight = 'Increase_hourly')) + geom_bar(facecolor = 'blue', fill = 'blue', alpha = .5)
print(gg_h)


# In[255]:


plot = ggplot(df_sum_h, aes(x='Hour',y='Increase_hourly'))         + geom_point(size=df_sum_h['Increase_hourly'].abs())         + xlim(-0.2,23.2) + ylab('Increase_hourly') + ggtitle('Total Increase_hourly By Hour')
print( plot)


# In[224]:


df_sum_h = df_tw[['UNIT','ENTRIESn_hourly','EXITSn_hourly']].groupby('UNIT', as_index = False).agg(np.mean)
#df_sum_h.describe


# In[225]:


gg_h = ggplot(df_sum_h, aes('UNIT', weight = 'ENTRIESn_hourly')) +geom_bar(facecolor = 'blue', fill = 'blue', alpha = .5)
print(gg_h)
gg_h = ggplot(df_sum_h, aes('UNIT', weight = 'EXITSn_hourly')) +geom_bar(facecolor = 'blue', fill = 'blue', alpha = .5)
print(gg_h)


# In[ ]:


# Mapreduce exercise
## Ridership per station


# In[ ]:


import sys
import string
import logging

from util import mapper_logfile
logging.basicConfig(filename=mapper_logfile, format='%(message)s',
                    level=logging.INFO, filemode='w')

def mapper():
    """
    In this exercise, for each turnstile unit, you will determine the date and time 
    (in the span of this data set) at which the most people entered through the unit.
    
    The input to the mapper will be the final Subway-MTA dataset, the same as
    in the previous exercise. You can check out the csv and its structure below:
    https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/turnstile_data_master_with_weather.csv

    For each line, the mapper should return the UNIT, ENTRIESn_hourly, DATEn, and 
    TIMEn columns, separated by tabs. For example:
    'R001\t100000.0\t2011-05-01\t01:00:00'

    Since you are printing the output of your program, printing a debug 
    statement will interfere with the operation of the grader. Instead, 
    use the logging module, which we've configured to log to a file printed 
    when you click "Test Run". For example:
    logging.info("My debugging message")
    Note that, unlike print, logging.info will take only a single argument.
    So logging.info("my message") will work, but logging.info("my","message") will not.
    """
    
    for line in sys.stdin:
        # your code here
        data = line.strip().split(',')

        if len(data) != 22 or data[1] == 'UNIT':
            continue
            
        print('{}\t{}\t{}\t{}'.format(data[1], data[6], data[2], data[3]))

mapper()

import sys
import logging

from util import reducer_logfile
logging.basicConfig(filename=reducer_logfile, format='%(message)s',
                    level=logging.INFO, filemode='w')

def reducer():
    '''
    Write a reducer that will compute the busiest date and time (that is, the 
    date and time with the most entries) for each turnstile unit. Ties should 
    be broken in favor of datetimes that are later on in the month of May. You 
    may assume that the contents of the reducer will be sorted so that all entries 
    corresponding to a given UNIT will be grouped together.
    
    The reducer should print its output with the UNIT name, the datetime (which 
    is the DATEn followed by the TIMEn column, separated by a single space), and 
    the number of entries at this datetime, separated by tabs.

    For example, the output of the reducer should look like this:
    R001    2011-05-11 17:00:00	   31213.0
    R002	2011-05-12 21:00:00	   4295.0
    R003	2011-05-05 12:00:00	   995.0
    R004	2011-05-12 12:00:00	   2318.0
    R005	2011-05-10 12:00:00	   2705.0
    R006	2011-05-25 12:00:00	   2784.0
    R007	2011-05-10 12:00:00	   1763.0
    R008	2011-05-12 12:00:00	   1724.0
    R009	2011-05-05 12:00:00	   1230.0
    R010	2011-05-09 18:00:00	   30916.0
    ...
    
    
    
    cur_key = None
    max_e = [None, None, 0]

    for line in sys.stdin:
        # your code here
        data = line.strip().split('\t')

        if len(data) != 4:
            continue

        new_key, count, date, time = data
        
        if cur_key == None:
            cur_key = new_key
        elif cur_key != new_key:
            print('{}\t{} {}\t{}'.format(cur_key, max_e[0], max_e[1], max_e[2]))
            cur_key = new_key
            max_e = [None, None, 0.0]
 
        if float(count) >= max_e[2]:
            max_e = [date, time, float(count)]
            
    if cur_key != None:        
        print('{}\t{} {}\t{}'.format(cur_key, max_e[0], max_e[1], max_e[2]))
        
            
reducer()


# In[5]:


## Ridership by weather


# In[ ]:


import sys
import string
import logging

from util import mapper_logfile
logging.basicConfig(filename=mapper_logfile, format='%(message)s',
                    level=logging.INFO, filemode='w')

def mapper():
    '''
    For this exercise, compute the average value of the ENTRIESn_hourly column 
    for different weather types. Weather type will be defined based on the 
    combination of the columns fog and rain (which are boolean values).
    For example, one output of our reducer would be the average hourly entries 
    across all hours when it was raining but not foggy.

    Each line of input will be a row from our final Subway-MTA dataset in csv format.
    You can check out the input csv file and its structure below:
    https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/turnstile_data_master_with_weather.csv
    
    Note that this is a comma-separated file.

    This mapper should PRINT (not return) the weather type as the key (use the 
    given helper function to format the weather type correctly) and the number in 
    the ENTRIESn_hourly column as the value. They should be separated by a tab.
    For example: 'fog-norain\t12345'
    
    '''

    # Takes in variables indicating whether it is foggy and/or rainy and
    # returns a formatted key that you should output.  The variables passed in
    # can be booleans, ints (0 for false and 1 for true) or floats (0.0 for
    # false and 1.0 for true), but the strings '0.0' and '1.0' will not work,
    # so make sure you convert these values to an appropriate type before
    # calling the function.
    def format_key(fog, rain):
        return '{}fog-{}rain'.format(
            '' if fog else 'no',
            '' if rain else 'no'
        )
    
    for line in sys.stdin:
    	# your code here
    	reader = line.strip().split(",")
    	if len(reader) != 22 or reader[1] == 'UNIT':
    	    continue
    	fog, rain, entries_h = reader[14], reader[15], reader[6]
    	key = format_key(float(fog), float(rain))
    	print('{}\t{}'.format(key, entries_h))

mapper()

import sys
import logging

from util import reducer_logfile
logging.basicConfig(filename=reducer_logfile, format='%(message)s',
                    level=logging.INFO, filemode='w')

def reducer():
    '''
    Given the output of the mapper for this assignment, the reducer should
    print one row per weather type, along with the average value of
    ENTRIESn_hourly for that weather type, separated by a tab. You can assume
    that the input to the reducer will be sorted by weather type, such that all
    entries corresponding to a given weather type will be grouped together.

    In order to compute the average value of ENTRIESn_hourly, you'll need to
    keep track of both the total riders per weather type and the number of
    hours with that weather type. That's why we've initialized the variable 
    riders and num_hours below. Feel free to use a different data structure in 
    your solution, though.

    An example output row might look like this:
    'fog-norain\t1105.32467557'

    '''

    riders = 0      # The number of total riders for this key
    num_hours = 0   # The number of hours with this key
    cur_key = None

    for line in sys.stdin:
        # your code here
        reader = line.strip().split("\t")
        if len(reader) != 2:
            continue
        new_key, count = reader
        if cur_key == None:
            cur_key = new_key
        elif cur_key != new_key:
            entries_h_avg = riders/float(num_hours)
            print('{}\t{}'.format(cur_key, entries_h_avg))
            cur_key = new_key
            num_hours = 0
            riders = 0
        num_hours += 1
        riders += float(count)
        
    if cur_key != None:
        entries_h_avg = riders/float(num_hours)
        print('{}\t{}'.format(cur_key, entries_h_avg))
        

reducer()


# In[ ]:


## busiest hours


# In[ ]:


import sys
import string
import logging

from util import mapper_logfile
logging.basicConfig(filename=mapper_logfile, format='%(message)s',
                    level=logging.INFO, filemode='w')

def mapper():
    """
    In this exercise, for each turnstile unit, you will determine the date and time 
    (in the span of this data set) at which the most people entered through the unit.
    
    The input to the mapper will be the final Subway-MTA dataset, the same as
    in the previous exercise. You can check out the csv and its structure below:
    https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/turnstile_data_master_with_weather.csv

    For each line, the mapper should return the UNIT, ENTRIESn_hourly, DATEn, and 
    TIMEn columns, separated by tabs. For example:
    'R001\t100000.0\t2011-05-01\t01:00:00'

    """
    
    for line in sys.stdin:
        # your code here
        data = line.strip().split(',')

        if len(data) != 22 or data[1] == 'UNIT':
            continue
            
        print('{}\t{}\t{}\t{}'.format(data[1], data[6], data[2], data[3]))

mapper()


import sys
import logging

from util import reducer_logfile
logging.basicConfig(filename=reducer_logfile, format='%(message)s',
                    level=logging.INFO, filemode='w')

def reducer():
    '''
    Write a reducer that will compute the busiest date and time (that is, the 
    date and time with the most entries) for each turnstile unit. Ties should 
    be broken in favor of datetimes that are later on in the month of May. You 
    may assume that the contents of the reducer will be sorted so that all entries 
    corresponding to a given UNIT will be grouped together.
    
    The reducer should print its output with the UNIT name, the datetime (which 
    is the DATEn followed by the TIMEn column, separated by a single space), and 
    the number of entries at this datetime, separated by tabs.

    For example, the output of the reducer should look like this:
    R001    2011-05-11 17:00:00	   31213.0
    R002	2011-05-12 21:00:00	   4295.0
    R003	2011-05-05 12:00:00	   995.0
    R004	2011-05-12 12:00:00	   2318.0
    R005	2011-05-10 12:00:00	   2705.0
    R006	2011-05-25 12:00:00	   2784.0
    R007	2011-05-10 12:00:00	   1763.0
    R008	2011-05-12 12:00:00	   1724.0
    R009	2011-05-05 12:00:00	   1230.0
    R010	2011-05-09 18:00:00	   30916.0
    ...
    ...

    '''
    
    
    cur_key = None
    max_e = [None, None, 0]

    for line in sys.stdin:
        # your code here
        data = line.strip().split('\t')

        if len(data) != 4:
            continue

        new_key, count, date, time = data
        
        if cur_key == None:
            cur_key = new_key
        elif cur_key != new_key:
            print('{}\t{} {}\t{}'.format(cur_key, max_e[0], max_e[1], max_e[2]))
            cur_key = new_key
            max_e = [None, None, 0.0]
 
        if float(count) >= max_e[2]:
            max_e = [date, time, float(count)]
            
    if cur_key != None:        
        print('{}\t{} {}\t{}'.format(cur_key, max_e[0], max_e[1], max_e[2]))
        
            
reducer()

