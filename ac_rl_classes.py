"""
Classes
"""
import talib as ta
import numpy as np
import pandas as pd
import gym
from gym import spaces
class DataProcessing:
    """
    Class for loading and processing databento TBBO data.
    """
    def __init__(self, file_path, ticker):
        data = pd.read_csv(file_path)
        data = data[data['symbol'] == ticker]

        data['price']=data['price']/1e9
        data['bid_px_00']=data['bid_px_00']/1e9
        data['ask_px_00']=data['ask_px_00']/1e9

        data['Close'] = data['price']
        data['Volume'] = data['size']
        data['High'] = data[['bid_px_00', 'ask_px_00']].max(axis=1)
        data['Low'] = data[['bid_px_00', 'ask_px_00']].min(axis=1)
        data['Open'] = data['Close'].shift(1).fillna(data['Close'])
        self.data = data.iloc[32:].dropna()
    
    def add_spread_info(self):
        self.data['mid_price'] = (self.data['bid_px_00'] + self.data['ask_px_00']) / 2
        self.data['spread'] = self.data['ask_px_00'] - self.data['bid_px_00']
        self.data['liquidity'] = self.data['bid_sz_00'] * self.data['bid_px_00'] + self.data['ask_sz_00'] * self.data['ask_px_00']
    
    def add_momentum_indicators(self):
        self.data['RSI'] = ta.RSI(self.data['Close'], timeperiod=14)
        self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = ta.MACD(self.data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['Stoch_k'], self.data['Stoch_d'] = ta.STOCH(self.data['High'], self.data['Low'], self.data['Close'],
                                                              fastk_period=14, slowk_period=3, slowd_period=3)
    def add_volume_indicators(self):
        self.data['OBV'] = ta.OBV(self.data['Close'], self.data['Volume'])

    def add_volatility_indicators(self):
        self.data['Upper_BB'], self.data['Middle_BB'], self.data['Lower_BB'] = ta.BBANDS(self.data['Close'], timeperiod=20)
        self.data['ATR_1'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=1)
        self.data['ATR_2'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=2)
        self.data['ATR_5'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=5)
        self.data['ATR_10'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=10)
        self.data['ATR_20'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=20)

    def add_trend_indicators(self):
        self.data['ADX'] = ta.ADX(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
        self.data['+DI'] = ta.PLUS_DI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
        self.data['-DI'] = ta.MINUS_DI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
        self.data['CCI'] = ta.CCI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=5)

    def add_other_indicators(self):
        self.data['DLR'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['TWAP'] = self.data['Close'].expanding().mean()
        self.data['VWAP'] = (self.data['Volume'] * (self.data['High'] + self.data['Low']) / 2).cumsum() / self.data['Volume'].cumsum()

    def get_state_columns(self):
        all_columns = self.data.columns.to_list()
        strings_to_remove = ['ts_recv', 'ts_event', 'rtype', 'publisher_id', 'instrument_id', 'flags', 'ts_in_delta', 'sequence', 'action', 'side','symbol', 
                     'size', 'ask_sz_00', 'bid_sz_00', 'bid_px_00', 'ask_px_00', 'bid_ct_00', 'ask_ct_00', 'price', 'depth']
        filtered_list = list(filter(lambda item: item not in strings_to_remove, all_columns))
        return filtered_list


    def add_all(self):
        self.add_spread_info()
        self.add_momentum_indicators()
        self.add_volume_indicators()
        self.add_volatility_indicators()
        self.add_trend_indicators()
        self.add_other_indicators()
        self.state_columns = self.get_state_columns()
        return self.data

class MarketImpactModel:
    """
    Base class for market impact models.

    Methods:
        calculate_impact: Abstract method to calculate market impact.
    """
    def calculate_impact(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")

class LinearCostModel:
    """
    Base class for linear cost models.

    Methods:
        calculate_cost: Abstract method to calculate linear costs.
    """
    def calculate_cost(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SimpleLinearCostModel(LinearCostModel):
    """
    Simple Linear Cost Model

    As used in the reference implementation.

    But really we should be taking into account more information like book depth and liquidity. 
    
    Or using the paper #1 formulation.
    """
    def __init__(self):
        pass
    def calculate_cost(self,row, shares):
        actual_price = row['price']
        expected_price = row['ask_px_00']
        return (expected_price- actual_price) * abs(shares)


class AlmgrenChrisModel(MarketImpactModel):
    """
    Almgren-Chris family of market impact models.

    Attributes:
        temporary_impact_params: Parameters for temporary impact.
        permanent_impact_params: Parameters for permanent impact.
    """
    def __init__(self, temporary_impact_params, permanent_impact_params):
        self.temporary_impact_params = temporary_impact_params
        self.permanent_impact_params = permanent_impact_params

    def calculate_temporary_impact(self, shares):
        """
        Calculate temporary market impact.
        
        Args:
            shares: The number of shares traded.

        Returns:
            Temporary impact cost.
        """
        shares = abs(shares)
        return self.temporary_impact_params['gamma'] * (shares ** self.temporary_impact_params['alpha'])

    def calculate_permanent_impact(self, shares):
        """
        Calculate permanent market impact.
        
        Args:
            shares: The number of shares traded.

        Returns:
            Permanent impact cost.
        """
        shares = abs(shares)
        return self.permanent_impact_params['eta'] * (shares ** self.permanent_impact_params['beta'])

    def calculate_impact(self, shares):
        """
        Calculate total market impact.
        
        Args:
            shares: The number of shares traded.

        Returns:
            Total impact cost.
        """
        temporary_impact = self.calculate_temporary_impact(shares)
        permanent_impact = self.calculate_permanent_impact(shares)
        return temporary_impact + permanent_impact # + brownian motion integral
    
    def calculate_variance_of_impact(self, shares):
        pass
    
class UtilityFunction:
    """
    Utility function for evaluating trading strategies.

    Attributes:
        risk_aversion: Parameter controlling the trade-off between expected shortfall and variance.
    """
    def __init__(self, risk_aversion):
        self.risk_aversion = risk_aversion

    def calculate_expected_cost(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def calculate_variance_of_cost(self, *args, **kwargs):

        raise NotImplementedError("This method should be overridden by subclasses.")

    def calculate_utility(self, *args, **kwargs):
        """
        Calculate the utility of a trading strategy.

        Returns:
            Utility value.
        """
        expected_cost = self.calculate_expected_cost(*args, **kwargs)
        variance_of_cost = self.calculate_variance_of_cost(*args, **kwargs)
        return -(expected_cost + self.risk_aversion * variance_of_cost) # negative to since we penalize cost
    
class AC_UtilityFunction(UtilityFunction):
    """
    Since we are learning the point-wise execution rather than a full trajectory plan all at once (like VWAP etc.)
    , this differs from the linked github implementation
    
    """
    def __init__(self, market_impact_model, linear_cost_model, risk_aversion=0.1):
        self.risk_aversion= risk_aversion
        self.market_impact_model = market_impact_model
        self.linear_cost_model = linear_cost_model
    
    def calculate_expected_cost(self, shares, latency_cost, row):
        return self.market_impact_model.calculate_impact(shares) + self.linear_cost_model.calculate_cost(row, shares) + latency_cost
    
    def calculate_variance_of_cost(self, shares, latency_cost, row):
        """
        For the reference implementation and paper #1, they assume we have use some trajectory rather than a pointwise execution. 

        We should use a measure of the actual variance of our AC model versus the realized empirical results.
        """
        return 0
    

class TradingEnvironment(gym.Env):
    """
    Environment for simulating trading strategies.    
    """
    def __init__(self, data_processing, market_impact_model, linear_cost_model, 
                 daily_trading_limit):
        self.data_processing = data_processing
        self.data = self.data_processing.add_all().dropna()
        self.state_columns = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Stoch_k', 'Stoch_d',
                              'OBV', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'ATR_1', 'ADX', '+DI', '-DI', 'CCI']
        
        self.market_impact_model = market_impact_model
        self.linear_cost_model = linear_cost_model
        self.utility_function = AC_UtilityFunction(self.market_impact_model, self.linear_cost_model)
        
        self.daily_trading_limit = daily_trading_limit
        self.current_step = 0
        self.balance = 10_000_000.0  # $10 million
        self.shares_held = 0
        self.total_shares_traded = 0
        self.action_space = spaces.Box(low = -1, high=1, shape=(1,), dtype=np.float32)
        # if this is negative, it represents the proportion of shares_held to (market) sell
        # if this is positive, it represents the proportion of total_shares_traded to (market) buy
    
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.state_columns),), dtype=np.float32
        )
    def reset(self):
        self.current_step = 0
        self.balance = 10_000_000.0  # $10 million
        self.shares_held = 0
        self.total_shares_traded = 0
        self.cumulative_reward = 0
        self.trades = []
        observation = self._next_observation()
        if np.any(np.isnan(observation)):
            raise ValueError('Data contains NaN values at reset.')
        return observation
    def _next_observation(self):        
        observation =self.data[self.state_columns].iloc[self.current_step].values
        if np.any(np.isnan(observation)):
            raise ValueError(f'Data contains NaN values at step {self.current_step}.')
        return observation
    
    def step(self, action):
        row = self.data.iloc[self.current_step]
        actual_price = row['price']
        reward = 0
        current_price = row['Close']
        if self.current_step >= len(self.data) - 1:
            self.current_step = 0

        if action > 0: # Buy
            shares_to_trade = np.around(self.balance / current_price * action)
        if action < 0: # Sell
            shares_to_trade = np.around(self.shares_held * action)

        current_time = pd.to_datetime(row['ts_event'])
        
        trade_info = {'step': self.current_step, 'timestamp': current_time, 'action': action, 'price': current_price}
        trade_info['shares'] = shares_to_trade
        
        transaction_time = row['ts_in_delta']
        time_penalty = 100*transaction_time/1e9
        if shares_to_trade!=0:
            self.balance -= shares_to_trade * current_price
            self.shares_held += shares_to_trade
            self.total_shares_traded += shares_to_trade
            reward = self.utility_function.calculate_utility(shares_to_trade, time_penalty, row)

            self.cumulative_reward += reward

            if self.trades:
                self.trades[-1]['reward'] = reward
            self.trades.append(trade_info)

        done = self.current_step == len(self.data) - 1
        obs = self._next_observation()
        info = {
        'step': self.current_step,
        'action': action,
        'price': actual_price,
        'shares': self.trades[-1]['shares'] if self.trades else 0
        }
        self.current_step += 1

        print(f"Step: {self.current_step}, Action: {action}, Shares to trade: {shares_to_trade}, Reward: {reward}")

        return obs, reward, done, info
    
    


    def run(self):
        self.reset()
        for _ in range(len(self.data)):
            self.step()
        return self.cumulative_reward, self.trades
    
    def print_trades(self):
        # download all trades in a pandas dataframe using .csv
        trades_df = pd.DataFrame(self.trades)
        # Save a csv
        trades_df.to_csv('trades_ppo.csv', index=False)
        for trade in self.trades:
            print(f"Step: {trade['step']}, Timestamp: {trade['timestamp']}, Action: {trade['action']}, Price: {trade['price']}, Shares: {trade['shares']}, Reward: {trade['reward']}, Transaction Cost: {trade['transaction_cost']}, Slippage: {trade['slippage']}, Time Penalty: {trade['time_penalty']}")

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total shares traded: {self.total_shares_traded}')
        print(f'Total portfolio value: {self.balance + self.shares_held * self.data.iloc[self.current_step]["Close"]}')
        print(f'Cumulative reward: {self.cumulative_reward}')
        self.print_trades()
    






