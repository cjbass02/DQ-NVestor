import gym
import numpy as np
import pandas as pd
import yfinance as yf
from gym import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, tickers, months=12, start_date=None, end_date=None,
                 buy_sell_percentages=None, initial_cash=100000,
                 break_by_month=True, episode_period="month"):
        super(PortfolioEnv, self).__init__()
        self.tickers = tickers
        self.months = months
        self.initial_cash = initial_cash
        self.break_by_month = break_by_month
        self.episode_period = episode_period  # month or year

        if buy_sell_percentages is None:
            self.buy_sell_percentages = [0.05, 0.1, 0.2, 1.0]
        else:
            self.buy_sell_percentages = buy_sell_percentages

        actions_per_stock = 1 + 2 * len(self.buy_sell_percentages)
        self.action_space = spaces.MultiDiscrete([actions_per_stock] * len(tickers))

        # per stock: gap, norm price, pos val, ret5
        obs_features_per_stock = 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(tickers) * obs_features_per_stock + 2,),
            dtype=np.float32,
        )

        self._load_data(start_date, end_date)
        if self.break_by_month:
            self._prepare_episode_indices()
            self.current_episode = 0
        else:
            self.current_episode = None

    def _load_data(self, start_date, end_date):
        # date range
        if end_date is None:
            end_date = pd.Timestamp.today().normalize()
        else:
            end_date = pd.to_datetime(end_date)

        if start_date is None:
            start_date = end_date - pd.DateOffset(months=self.months)
        else:
            start_date = pd.to_datetime(start_date)

        # grab stock data
        data = yf.download(self.tickers, start=start_date, end=end_date, interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [f"{col}_{ticker}" for col, ticker in data.columns]

        data.dropna(inplace=True)
        data.reset_index(inplace=True)

        # dates as 1D
        if "Date" in data.columns:
            self.dates = pd.to_datetime(data["Date"]).to_numpy()
        else:
            self.dates = None

        self.open_prices = {tic: data[f"Open_{tic}"].to_numpy() for tic in self.tickers}
        self.close_prices = {tic: data[f"Close_{tic}"].to_numpy() for tic in self.tickers}
        self.data_len = len(data)

        # 5-day returns
        self.ret5 = {}
        for tic in self.tickers:
            close_series = pd.Series(self.close_prices[tic])
            r5 = close_series.pct_change(5).fillna(0.0).to_numpy()
            self.ret5[tic] = r5

        # SPY context
        spy = yf.download("SPY", start=start_date, end=end_date, interval="1d", progress=False)
        spy.dropna(inplace=True)
        spy.reset_index(inplace=True)

        # flatten to 1D
        self.spy_close = spy["Close"].to_numpy().reshape(-1)

        # SPY 1 day return
        spy_ret = pd.Series(self.spy_close).pct_change().fillna(0.0).to_numpy()
        self.spy_ret_1d = spy_ret

        # align lengths
        min_len = min(self.data_len, len(self.spy_close))
        if min_len < self.data_len:
            # trim stocks to SPY
            self.data_len = min_len
            if self.dates is not None:
                self.dates = self.dates[:min_len]
            for tic in self.tickers:
                self.open_prices[tic] = self.open_prices[tic][:min_len]
                self.close_prices[tic] = self.close_prices[tic][:min_len]
                self.ret5[tic] = self.ret5[tic][:min_len]

        if min_len < len(self.spy_close):
            self.spy_close = self.spy_close[:min_len]
            self.spy_ret_1d = self.spy_ret_1d[:min_len]

        # final length
        self.data_len = min_len

    def _prepare_episode_indices(self):
        """Make episode start indices (by month or year)."""
        dates = self.dates 
        self.episode_start_idx = []
        last_key = None

        if dates is None:
            self.total_episodes = 0
            return

        for i, dt in enumerate(dates):
            dt = pd.Timestamp(dt)
            if self.episode_period == "year":
                key = dt.year
            else:  # default: month
                key = (dt.year, dt.month)

            if key != last_key:
                self.episode_start_idx.append(i)
                last_key = key

        self.total_episodes = len(self.episode_start_idx)

    def reset(self):
        if self.break_by_month:
            if self.current_episode is not None and self.current_episode >= len(self.episode_start_idx):
                self.current_episode = 0
            start_idx = self.episode_start_idx[self.current_episode]
            self.current_idx = start_idx
            self.current_episode += 1
        else:
            self.current_idx = 0

        self.positions = np.zeros(len(self.tickers))
        self.cash = self.initial_cash
        return self._get_observation()

    def _get_observation(self):
        obs = []
        idx = self.current_idx

        # per-stock
        for i, tic in enumerate(self.tickers):
            price_open = self.open_prices[tic][idx]

            if idx == 0:
                gap_pct = 0.0
            else:
                prev_close = self.close_prices[tic][idx - 1]
                gap_pct = (price_open - prev_close) / prev_close

            norm_price = price_open / self.open_prices[tic][0]
            pos_value = self.positions[i] * price_open / self.initial_cash
            ret5 = self.ret5[tic][idx]  # 5-day return

            obs.extend([gap_pct, norm_price, pos_value, ret5])

        # cash feature
        cash_frac = self.cash / self.initial_cash

        # SPY ret
        spy_ret_1d = self.spy_ret_1d[idx]

        obs.append(cash_frac)
        obs.append(spy_ret_1d)

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action = np.array(action, dtype=int)
        actions_per_stock = 1 + 2 * len(self.buy_sell_percentages)
        action = np.clip(action, 0, actions_per_stock - 1)

        idx = self.current_idx
        reward = 0.0

        # trade at open
        for i, tic in enumerate(self.tickers):
            act = action[i]
            price = self.open_prices[tic][idx]

            if act == 0:
                continue

            if act % 2 == 1:
                # buy
                perc_index = act // 2
                perc = self.buy_sell_percentages[perc_index]
                buy_amount = (self.initial_cash / len(self.tickers)) * perc
                shares_to_buy = buy_amount / price
                cost = shares_to_buy * price

                if self.cash >= cost:
                    self.positions[i] += shares_to_buy
                    self.cash -= cost
                else:
                    reward -= 0.001  # penalty for failed buy
            else:
                # sell
                perc_index = (act // 2) - 1
                perc = self.buy_sell_percentages[perc_index]
                sell_amount = (self.initial_cash / len(self.tickers)) * perc
                shares_to_sell = sell_amount / price
                shares_to_sell = min(self.positions[i], shares_to_sell)

                self.positions[i] -= shares_to_sell
                self.cash += shares_to_sell * price

        # open->close reward
        current_val = sum(
            self.positions[i] * self.open_prices[tic][idx]
            for i, tic in enumerate(self.tickers)
        ) + self.cash

        next_val = sum(
            self.positions[i] * self.close_prices[tic][idx]
            for i, tic in enumerate(self.tickers)
        ) + self.cash

        if current_val > 0:
            reward += (next_val - current_val) / current_val

        # diversification bonus
        equity_vals = np.array([
            self.positions[i] * self.open_prices[tic][idx]
            for i, tic in enumerate(self.tickers)
        ], dtype=float)

        equity = equity_vals.sum()
        if equity > 0:
            weights = equity_vals / equity
            hhi = np.sum(weights ** 2)
            n = len(self.tickers)
            hhi_min = 1.0 / n

            diversity_score = (1.0 - hhi) / (1.0 - hhi_min)

            diversity_lambda = 0.001
            reward += diversity_lambda * diversity_score

        # next step
        self.current_idx += 1
        done = False
        if self.current_idx >= self.data_len:
            done = True
        elif self.break_by_month and self.current_idx in self.episode_start_idx[1:]:
            # stop at next month/year
            done = True

        obs = self._get_observation() if not done else None
        info = {"portfolio_value": next_val, "cash": self.cash}
        return obs, reward, done, info

    def get_hhi(self):

        if self.current_idx >= self.data_len:
            idx = self.data_len - 1
        else:
            idx = self.current_idx

        equity_vals = np.array([
            self.positions[i] * self.open_prices[tic][idx]
            for i, tic in enumerate(self.tickers)
        ], dtype=float)

        equity = equity_vals.sum()
        if equity <= 0:
            return None, 0.0

        weights = equity_vals / equity
        hhi = float(np.sum(weights ** 2))

        n = len(self.tickers)
        hhi_min = 1.0 / n
        diversity_score = float((1.0 - hhi) / (1.0 - hhi_min))

        return hhi, diversity_score