import schedule
from datetime import date, datetime
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import csv
import pandas as pd
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from collections import deque
import random
import re
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import gym
from gym import spaces
import time
from matplotlib import pyplot as plt
import math

# web scraping share price from investing.com and do some calculation for 
def checkSharePrice(x):
    # turn string date to date format
    today = datetime.strptime(str(x), "%m/%d/%Y")
    
    # send request to the website
    link = "https://www.investing.com/equities/airasia-bhd-historical-data" 
    req = Request(link, headers={'User-Agent':'Mozilla/5.0'})
    webpage = urlopen(req).read()
    
    # get background html
    html = BeautifulSoup(webpage, "html.parser")
    
    # find information through tag attribute
    historicalPriceTable = html.find('table', {'class':'datatable_table__D_jso datatable_table--border__B_zW0 datatable_table--mobile-basic__W2ilt datatable_table--freeze-column__7YoIE'})
    try:
        for row in historicalPriceTable.find_all('tr', {'data-test':'historical-data-table-row'}):
            data = row.find_all('td')
            Date = data[0].find('time').text
            Price = data[1].text
            Open = data[2].text
            High = data[3].text
            Low = data[4].text
            Vol = data[5].text
            Change = data[6].text
            Vol = strToFloat(Vol)
            
            # check is the date inside the file already or not
            if(datetime.strptime(Date, "%m/%d/%Y") == today):
                with open('./realtimeData.csv', 'a', newline='') as f:
                    df = pd.read_csv('./realtimeData.csv', header=None ,names=['Date', 'Open', 'High', 'Low', 'Vol.', 'Close', 'KLCI', 'Pos', 'Neu', 'Neg', 'dr', 'f02', 'Vol Log', 'diff', 'diff50', 'roc', 'ma_5', 'ma_200', 'ema_50'])
                    # yes then close
                    if(Date in df['Date'].unique()):
                        f.close()
                    # no then write data into the file
                    else:
                        # feature engineering calculation
                        dr = (float(Price)/float(Open)) - 1
                        f02 = (float(Price)/df['Close'].iloc[-1] - 1)
                        VLog = math.log(Vol)
                        diff = Vol - df['Vol.'].iloc[-1]
                        diff50 = Vol - df['Vol.'].iloc[-49]
                        rocV = Vol/roc([df['Vol.'].iloc[-1], Vol], 2)[-1]
                        ma_5 = moving_average(np.concatenate((df['Vol.'][-4:], [Vol])), 5)[-1]
                        ma_200 = Vol/moving_average(np.concatenate((df['Vol.'][-199:], [Vol])), 200)[-1]
                        ema_50 = float(Price)/moving_average(np.concatenate((df['Close'][-199:], [float(Price)])), 200)[-1]
                        writer = csv.writer(f)
                        writer.writerow([Date, float(Open), float(High), float(Low), Vol, float(Price), 0, 0, 0, 0, dr, float(f02), VLog, float(diff), float(diff50), rocV, float(ma_5), float(ma_200), float(ema_50)])
                        f.close()
                    checkKLCI(x)
                    getNews(x)
                break
            elif(datetime.strptime(Date, "%m/%d/%Y") < today):
                print("No such date data. Maybe Holiday.")
                break
    except Exception as E:
        print(E)

# text preprocessing for volume to remove M or K and multiply them with number. Exp: 1K -> 1000
def strToFloat(x):
    if re.search("M$", x) != None:
        return float(re.sub('M','',x))*1000000
    elif  re.search("K$", x) != None:
        return float(re.sub('K','',x))*1000

# web scraping KLCI index from investing.com
def checkKLCI(x):
    # turn string into date format
    today = datetime.strptime(str(x), "%m/%d/%Y")
    
    # request for the website
    link = "https://www.investing.com/indices/ftse-malaysia-klci-historical-data"
    req = Request(link, headers={'User-Agent':'Mozilla/5.0'})
    webpage = urlopen(req).read()
    # get underlying HTML coding
    html = BeautifulSoup(webpage, "html.parser")
    # find information by tag attribute
    historicalPriceTable = html.find('table', {'class':'datatable_table__D_jso datatable_table--border__B_zW0 datatable_table--mobile-basic__W2ilt datatable_table--freeze-column__7YoIE'})
    try:
        for row in historicalPriceTable.find_all('tr', {'data-test':'historical-data-table-row'}):
            data = row.find_all('td')
            Date = data[0].find('time').text
            Price = data[1].text
            Open = data[2].text
            High = data[3].text
            Low = data[4].text
            Vol = data[5].text
            Change = data[6].text
            # check same date and write the data to the row in csv file
            if(datetime.strptime(Date, "%m/%d/%Y") == today):
                df = pd.read_csv('./realtimeData.csv', header=None ,names=['Date', 'Open', 'High', 'Low', 'Vol.', 'Close', 'KLCI', 'Pos', 'Neu', 'Neg', 'dr', 'f02', 'Vol Log', 'diff', 'diff50', 'roc', 'ma_5', 'ma_200', 'ema_50'])
                if(Date in df['Date'].unique()):
                    df.loc[df['Date']==Date, ['KLCI']] = float(re.sub(',','',Price))
                    df.to_csv('./realtimeData.csv', index=False, header=False)
                    break
                else:
                    print("No such date data. Maybe Holiday.")
                    break
            elif(datetime.strptime(Date, "%m/%d/%Y") < today):
                print("No such date data. Maybe Holiday.")
                break
    except Exception as E:
        print(E)
    
# get currect date
def getCurrentDate():
    today = date.today()
    return today.strftime("%m/%d/%Y")

# text preprocessing for real time news headlines sentiment analysis
def process_text(text):
    lemmatizer = WordNetLemmatizer()
    
    # Remove newline
    text = text.replace('\n',' ').replace('\r','').strip()
    
    # remove spcial character and digits
    text = re.sub('[\W\d]',' ',text)
    text = re.sub(' +', ' ', text)
    
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    
    # lemmatization
    for i in range(len(filtered_sentence)): filtered_sentence[i] = re.sub(filtered_sentence[i], lemmatizer.lemmatize(filtered_sentence[i]), filtered_sentence[i]) 
    
    # combine those words to sentence
    text = ' '.join(filtered_sentence)
    text = re.sub(' +', ' ', text)
    return text

# sentiment analysis for real time news headlines 
def sentimentAnalysis(x):
    generator = pipeline(task="sentiment-analysis",model="ProsusAI/finbert")
    return (generator(process_text(x))[0]['label'], generator(process_text(x))[0]['score'])

# get realtime news headlines from klsescreener.com website
def getNews(x):
    # turn string into date format
    today = datetime.strptime(str(x), "%m/%d/%Y")
    
    # request for the website
    link = "https://www.klsescreener.com/v2/news/stock/5099" 
    req = Request(link, headers={'User-Agent':'Mozilla/5.0'})
    webpage = urlopen(req).read()
    
    # get underlying HTML coding
    html = BeautifulSoup(webpage, "html.parser")
    newsResult = [0,0,0]
    try:
        # get news headlines by tag attribute in HTML code
        for row in html.find_all('div', {'class':'item figure flex-block'}):
            title = row.find('a',{'target':'_blank'})
            orgAndDate = row.find_all('span')
            if(datetime.strptime(orgAndDate[1].get('data-date').split(' ')[0], '%Y-%m-%d') == today):
                # sentiment analysis
                result = sentimentAnalysis(title.text)
                if(result[0]=='positive'):
                    newsResult[0]+=result[1]
                elif(result[0]=='neutral'):
                    newsResult[1]+=result[1]
                elif(result[0]=='negative'):
                    newsResult[2]+=result[1]
            elif(datetime.strptime(orgAndDate[1].get('data-date').split(' ')[0], '%Y-%m-%d') < today):
                break
        # write it to file
        df = pd.read_csv('./realtimeData.csv', header=None ,names=['Date', 'Open', 'High', 'Low', 'Vol.', 'Close', 'KLCI', 'Pos', 'Neu', 'Neg', 'dr', 'f02', 'Vol Log', 'diff', 'diff50', 'roc', 'ma_5', 'ma_200', 'ema_50'])
        Date = today.strftime("%m/%d/%Y")
        if(Date in df['Date'].unique()):
            df.loc[df['Date']==Date, ['Pos','Neu','Neg']] = newsResult
            df.to_csv('./realtimeData.csv', index=False, header=False)
    except Exception as E:
        print(E)

# read value from txt file
def readVar():
    f = open("./variable.txt", "r")
    text = f.read()
    var = text.split('\n')
    key = []
    value = []
    for i in range(len(var)):
        keyValuePair = var[i].split(' = ')
        key.append(keyValuePair[0])
        value.append(float(keyValuePair[1]))
    return key, value

# write value from txt file
def writeVar(key, value):
    result = []
    for i in range(len(key)):
        result.append("{} = {}".format(key[i], value[i]))
    new = '\n'.join(result)
    with open("./variable.txt", "w") as f:
        f.write(new)
        f.close()

# reinforcement learning environment
class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.df = df
        self.MAX_ACCOUNT_BALANCE = 4000000
        self.MAX_NUM_SHARES = 4116080
        self.MAX_SHARE_PRICE = 3
        self.MAX_INDEX = 5000
        self.MAX_NEWS = 10
        self.MAX_OPEN_POSITIONS = 10
        self.MAX_STEPS = 3000
        self.INITIAL_ACCOUNT_BALANCE = 10000
        self.days = 5
        
        self.reward_range = (0, self.MAX_ACCOUNT_BALANCE) 
        
        
        # Buy, sell, hold
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        # low = lowest accepted value
        # high = highest accepted value
        # possible output = [0,0], [1,0], [1,1], [2,0] ......
        # 0-3: action, 0-1: amount 
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        # open , high, low, close, daily volume, balance, curStockPositions, profit
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
          low=0, high=1, shape=(18, self.days+1), dtype=np.float16)


    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'Open'].values / self.MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'High'].values / self.MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'Low'].values / self.MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'Close'].values / self.MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'Volume'].values / self.MAX_NUM_SHARES,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'KLCI'].values / self.MAX_INDEX,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'Pos'].values / self.MAX_NEWS,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'Neu'].values / self.MAX_NEWS,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'dr'].values / self.MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'f02'].values / self.MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'Vol Log'].values / self.MAX_NUM_SHARES,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'diff50'].values / self.MAX_NUM_SHARES,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'diff'].values / self.MAX_NUM_SHARES,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'roc'].values / self.MAX_NUM_SHARES,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'ma_5'].values / self.MAX_NUM_SHARES,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'ma_200'].values / self.MAX_NUM_SHARES,
            self.df.loc[self.current_step: self.current_step +
                        self.days, 'ema_50'].values / self.MAX_SHARE_PRICE,
        ])

        details = [
            self.balance / self.MAX_ACCOUNT_BALANCE,
            self.max_net_worth / self.MAX_ACCOUNT_BALANCE,
            self.shares_held / self.MAX_NUM_SHARES,
            self.cost_basis / self.MAX_SHARE_PRICE,
            self.total_shares_sold / self.MAX_NUM_SHARES,
            self.total_sales_value / (self.MAX_NUM_SHARES * self.MAX_SHARE_PRICE),
        ]
        
        details = np.concatenate((details, [0]*(self.days-5)))
        
        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [details], axis=0)
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "Close"]
        
        action_type = action[0]
        amount = action[1]
        lot = 1
        # Buy amount % of shares held 
        self.buy.append(None)
        self.sell.append(None)
        if action_type < 1:
            total_possible = int(self.balance / current_price)
            shares_bought = int((total_possible * amount)/lot)*lot
            # error checking for buying
            if(shares_bought>0):
                prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * current_price
                extrafee = calculateExtrafee(additional_cost)
                max_available_share = 1
                while((additional_cost + extrafee)>self.balance or (additional_cost<0) or (extrafee<0)):
                    shares_bought = int(((total_possible-(max_available_share*lot)) * amount)/lot)*lot
                    additional_cost = shares_bought * current_price
                    extrafee = calculateExtrafee(additional_cost)
                    max_available_share += 1
                    if(shares_bought<=max_available_share):
                        shares_bought = 0
                        additional_cost = shares_bought * current_price
                        extrafee = calculateExtrafee(additional_cost)
                        break
                self.balance -= (additional_cost + extrafee)
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
                self.shares_held += shares_bought
                self.buy[-1] = self.df.loc[self.current_step, "Close"]
        # Sell amount % of shares held
        elif action_type < 2:
            shares_sold = int((self.shares_held/lot) * amount)*lot
            if(shares_sold>0):
                additional_cost = shares_sold * current_price
                extrafee = calculateExtrafee(additional_cost)
                self.balance += (additional_cost - extrafee)
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * current_price
                self.sell[-1] = self.df.loc[self.current_step, "Close"]

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
            
        if self.shares_held == 0:
            self.cost_basis = 0


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        
        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - (self.days+1):
            self.current_step = 0

        delay_modifier = (self.current_step / self.MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()
        return obs, reward, done, [self.balance, self.net_worth, self.max_net_worth, self.shares_held,
           self.cost_basis, self.total_shares_sold, self.total_sales_value], self.buy, self.sell

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = self.INITIAL_ACCOUNT_BALANCE
        self.net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        #self.current_step = random.randint(
        #   0, len(self.df.loc[:, 'Open'].values) - (days+1))
        self.current_step = 0
        
        self.buy = []
        self.sell = []
        
        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - self.INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Total Cost: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Total Account Value: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Net Profit: {profit}')
    
def calculateExtrafee(additional_cost):
    # Brokerage fee, clearing fee, stamp duty
    # bf - min RM 8, 0.08%
    # sd - RM 1.5 per RM1000 of transaction value
    # cf - 0.03% of transaction value, max RM 1000
    extrafee = 0
    bf = 0.0008
    cf = 0.0003
    sd = 1.5
    if additional_cost * bf > 8:
        extrafee += additional_cost * bf
    else:
        extrafee += 8
    if additional_cost * cf > 1000:
        extrafee += 1000
    else:
        extrafee += additional_cost * cf
    extrafee += int(additional_cost / 1000)*sd
    return extrafee

# rate of change function
def roc(arr, window_size):
    roc_array = []
    i = 0
    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i : i + window_size]
        roc_val = ((window[i + window_size-1]-window[i])/ window[i])* 100

        # Store the average of current
        # window in moving average list
        roc_array.append(roc_val)

        # Shift window to right by one position
        i += 1
    return roc_array

# moving average function
def moving_average(arr, window_size):
    moving_averages = []
    i = 0
    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i : i + window_size]

        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1
    return moving_averages