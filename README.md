Multi-Head DQN Portfolio Agent
==============================

Research repo for a Deep Q-Network (DQN) agent that trades a basket of US stocks using a custom Gym environment, real historical data from Yahoo Finance, and a multi-head Q-network (one head per stock).

The agent:

*   Starts with a fixed cash budget (e.g. $100,000).
    
*   Trades **once per day at the market open** for each stock.
    
*   Optimizes **daily open→close portfolio return**, with a small bonus for **diversification** (HHI-based).
    
*   Is trained on **many years of history**, then evaluated on a **held-out 12-month test window** against SPY.
    

1\. Motivation
--------------

*   Real-world portfolios are **multi-asset** and decisions are **joint**: you decide how to handle _each_ stock given the same market state.
    
*   Classic DQN examples (CartPole, Atari) focus on **single-action** agents and toy states.
    
*   This project explores:
    
    *   A **MultiDiscrete** action space (one discrete action per stock).
        
    *   A **multi-head DQN**: shared market context, specialized decision head per stock.
        
    *   Using **real equity data** (via yfinance) and simple market features in a clean, reproducible RL setup.
        

2\. Project Structure
---------------------

Adjust filenames as needed to match your repo; conceptually it looks like:

*   environment.pyCustom Gym environment PortfolioEnv:
    
    *   Downloads OHLC data for a list of tickers.
        
    *   Builds daily state features.
        
    *   Executes trades and computes rewards.
        
*   agent.py (or dqn\_agent.py)
    
    *   MultiStockQNetwork: shared trunk + per-stock heads.
        
    *   DQNAgent: replay buffer, epsilon-greedy, target network, train step.
        
*   replay\_buffer.pySimple experience replay implementation.
    
*   notebook.ipynb / notebook\_train\_test\_split.ipynb
    
    *   Train/test split (150 months train, last 12 months test).
        
    *   Training loop.
        
    *   Evaluation loop vs SPY.
        
    *   Plots of portfolio value, per-stock exposure, shares over time, etc.
        
*   requirements.txt(gym, torch, numpy, pandas, matplotlib, yfinance, etc.)
    

3\. Environment: PortfolioEnv
-----------------------------

### 3.1 Data & APIs

Uses **Yahoo Finance** via yfinance:

*   yf.download(self.tickers, start=..., end=..., interval="1d")
    
*   yf.download("SPY", ...) for benchmark and SPY features.
    

Data is daily candles. The environment:

*   Cleans NA rows.
    
*   Aligns stock data and SPY by truncating both to the same length.
    
*   Stores:
    
    *   open\_prices\[tic\]\[t\], close\_prices\[tic\]\[t\]
        
    *   spy\_close\[t\]
        
    *   Precomputed 5-day stock returns and 1-day SPY returns.
        

### 3.2 Episodes: month vs year

Config:
```
PortfolioEnv(      tickers=...,      start_date=...,      end_date=...,      initial_cash=100000,      break_by_month=True,      # split into episodes or not      episode_period="month",   # "month" or "year"  )
```

*   break\_by\_month=False → one long episode from start\_date to end\_date.
    
*   break\_by\_month=True and:
    
    *   episode\_period="month" → each calendar month is an episode.
        
    *   episode\_period="year" → each calendar year is an episode.
        

Episode boundaries are built from the date series: when (year, month) or year changes, a new episode starts.

### 3.3 Action space (MultiDiscrete)

For each stock you define a set of buy/sell percentages:

```
buy_sell_percentages = [0.05, 0.1, 0.2, 1.0]  # 5%, 10%, 20%, 100%
```

This yields 1 + 2 \* len(buy\_sell\_percentages) actions per stock:

*   0 = HOLD
    
*   1, 3, 5, 7, ... = BUY at those % levels (of a per-stock budget).
    
*   2, 4, 6, 8, ... = SELL at those % levels.
    

The overall action is:

```
self.action_space = spaces.MultiDiscrete([actions_per_stock] * n_stocks)
```

So the agent outputs a **vector** of choices, one discrete action per ticker, every day.

### 3.4 State representation (post-engineering)

On each trading day idx, the environment builds:

**Per stock (4 features per ticker):**

1.  gapi=openi(t)−closei(t−1)closei(t−1)\\text{gap}\_i = \\frac{\\text{open}\_i(t) - \\text{close}\_i(t-1)}{\\text{close}\_i(t-1)}gapi​=closei​(t−1)openi​(t)−closei​(t−1)​
    
2.  normPricei=openi(t)openi(0)\\text{normPrice}\_i = \\frac{\\text{open}\_i(t)}{\\text{open}\_i(0)}normPricei​=openi​(0)openi​(t)​
    
3.  posVali=sharesi(t)⋅openi(t)initial\_cash\\text{posVal}\_i = \\frac{\\text{shares}\_i(t)\\cdot\\text{open}\_i(t)}{\\text{initial\\\_cash}}posVali​=initial\_cashsharesi​(t)⋅openi​(t)​
    
4.  **5-day return** of that stock (close-to-close).
    

**Global features (2 features total):**

1.  **Cash fraction**: cash / initial\_cash.
    
2.  **SPY 1-day return** (close-to-close).
    

Total state dimension:

```
state_dim = 4 * n_stocks + 2
```

### 3.5 Reward function

Trades execute at **today’s open**.

Reward is the **portfolio’s open→close return for that day**, plus a diversification bonus.

Core P&L term:
```
current_val = Σ_i (positions[i] * open_prices[tic][idx]) + cash  next_val    = Σ_i (positions[i] * close_prices[tic][idx]) + cash  reward += (next_val - current_val) / current_val
```

**Diversification (HHI) bonus:**

*   equity\_vals = positions\[i\] \* open\_prices\[tic\]\[idx\]equity = equity\_vals.sum()weights = equity\_vals / equity # if equity > 0
    
*   HHI=∑iwi2HHI = \\sum\_i w\_i^2HHI=i∑​wi2​(1/N = best diversification, 1 = fully concentrated in one stock)
    
*   diversity\_score=1−HHI1−1/N\\text{diversity\\\_score} = \\frac{1 - HHI}{1 - 1/N}diversity\_score=1−1/N1−HHI​
    
*   reward += diversity\_lambda \* diversity\_score # small coefficient, e.g. 0.001
    

This gently nudges the agent away from one-stock bets without dominating the return term.

4\. Multi-Head DQN Agent
------------------------

### 4.1 Network structure: shared trunk + per-stock heads

Implemented in PyTorch as MultiStockQNetwork:

*   **Input**: full state vector \[batch\_size, state\_dim\].
    
*   ```self.shared = nn.Sequential( nn.Linear(state\_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),)Produces a shared embedding h (\[batch, 256\]) encoding full market + portfolio context.```
    
*   ```self.heads = nn.ModuleList( \[nn.Linear(256, actions\_per\_stock) for \_ in range(n\_stocks)\])In forward:h = self.shared(x) # \[batch, 256\]q\_list = \[\]for head in self.heads: q\_i = head(h) # \[batch, actions\_per\_stock\] q\_list.append(q\_i.unsqueeze(1))q\_all = torch.cat(q\_list, dim=1) # \[batch, n\_stocks, actions\_per\_stock\]```
    

Each head sees the same h, but has its own weights → separate Q-values per stock, shared context.

### 4.2 DQNAgent

DQNAgent wraps:

*   Online and target networks:
    
    *   q\_network = MultiStockQNetwork(...)
        
    *   target\_network initialized with same weights, updated periodically.
        
*   Optimizer:
    
    *   optim.Adam(q\_network.parameters(), lr=...).
        
*   Replay buffer:
    
    *   Stores (state, action\_vector, reward, next\_state, done) transitions.
        
*   ε-greedy policy:
    
    *   epsilon\_start, epsilon\_min, epsilon\_decay over episodes.
        
    *   For exploration: random MultiDiscrete action; otherwise argmax Q per stock.
        
*   Training step:
    
    *   Sample batch from replay.
        
    *   Compute Q(s,a) for all stocks, gather chosen actions.
        
    *   Compute targets using target network and max over actions.
        
    *   MSE loss over all stocks and batch; backprop through shared + heads.
        

5\. Train / Test Setup
----------------------

### 5.1 Date split

In the notebook:
```
end         = pd.Timestamp.today().normalize()  train_start = end - pd.DateOffset(months=150)  # 150 months ago  train_end   = end - pd.DateOffset(months=12)   # 12 months ago  test_start  = train_end                        # last 12 months  test_end    = end
```

*   **Training window**: 150 months ago → 12 months ago.
    
*   **Test window**: last 12 months.
    

### 5.2 Environments

**Training environment (older data only):**

```
train_env = PortfolioEnv(      tickers=tickers,      start_date=train_start,      end_date=train_end,      break_by_month=True,      episode_period="month",  # or "year"      initial_cash=100000,      buy_sell_percentages=[0.05, 0.1, 0.2, 1.0],  )
```

**Evaluation environment (last 12 months):**
```
eval_env = PortfolioEnv(      tickers=tickers,      start_date=test_start,      end_date=test_end,      break_by_month=False,  # one long test episode      initial_cash=100000,      buy_sell_percentages=[0.05, 0.1, 0.2, 1.0],  )
```

*   With episode\_period="month" and break\_by\_month=True, each training episode ≈ one calendar month.
    
*   Training “epochs” just cycle over these time episodes multiple times.
    

6\. Running the Experiment
--------------------------

### 6.1 Install dependencies
```
pip install -r requirements.txt
```

Example requirements.txt:
```
gym  torch  numpy  pandas  matplotlib  yfinance
```

### 6.2 Launch the notebook
```
jupyter lab  # or  jupyter notebook
```

Open notebook.ipynb / notebook\_train\_test\_split.ipynb and run cells in order:

1.  Imports and config (tickers, hyperparameters).
    
2.  Train/test split + create train\_env and eval\_env.
    
3.  Create agent (DQNAgent).
    
4.  Training loop over episodes.
    
5.  Evaluation loop on eval\_env (policy-only, no exploration).
    
6.  Plots:
    
    *   Portfolio value vs SPY (in dollars).
        
    *   Per-stock position value.
        
    *   Per-stock shares through time.
        
    *   Training returns per episode.
        

7\. Key knobs to tweak
----------------------

*   **Tickers**: universe composition and size.
    
*   **Buy/sell percentages**: discrete action granularity per stock.
    
*   **Episode granularity**: episode\_period="month" vs "year", or break\_by\_month=False.
    
*   **Initial cash**: portfolio starting size.
    
*   **Reward shaping**:
    
    *   diversity\_lambda for HHI bonus.
        
*   **Network size**:
    
    *   Hidden units in shared trunk.
        
    *   Extra layers in per-stock heads.
        
*   **Training hyperparameters**:
    
    *   Learning rate, batch size, replay capacity.
        
    *   Epsilon schedule.
        
    *   Target update frequency.
        
    *   Number of epochs over the train window.
        

8\. Limitations / Future Work
-----------------------------

*   Uses **daily open→close** only (no intraday dynamics).
    
*   No transaction costs, slippage, or liquidity modeling.
    
*   Simple technical features (gaps, 5-day returns, SPY 1-day returns); richer indicators could be added.
    
*   DQN on non-stationary financial data is inherently noisy; this is a research/teaching setup, not a live trading system.
    

Ideas to explore:

*   More features (volatility, volume, cross-asset signals).
    
*   Alternative rewards (risk-adjusted metrics, drawdown penalties).
    
*   Different RL algorithms (DDPG/SAC for continuous sizing, distributional RL, etc.).
