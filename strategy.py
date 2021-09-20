# Functions to implement our trading strategy.
import numpy as np
import trading_solution.process as proc
import trading_solution.indicators as ind

def random(stock_prices, period=7, amount=5000, fees=20, ledger='ledger_random.txt'):
    '''
    Randomly decide, every period, which stocks to purchase,
    do nothing, or sell (with equal probability).
    Spend a maximum of amount on every purchase.

    Input:
        stock_prices (ndarray): the stock price data
        period (int, default 7): how often we buy/sell (days)
        amount (float, default 5000): how much we spend on each purchase
            (must cover fees)
        fees (float, default 20): transaction fees
        ledger (str): path to the ledger file

    Output: None
    '''
    # Create an empty ledger writing file
    open(ledger, 'w').close()

    # Handle possibility of stock_prices being a single column
    ndims = len(stock_prices.shape)
    if ndims == 1:
        # Make it a 2D array
        stock_prices = stock_prices.reshape((stock_prices.shape[0], 1))

    # Create the portfolio
    N = stock_prices.shape[1]
    portfolio = proc.create_portfolio([amount] * N, stock_prices, fees, ledger)

    # Every period, randomly decide whether to do nothing, buy, or sell all shares
    rng = np.random.default_rng()
    for day in range(0, stock_prices.shape[0], period):
        # Make the decision for each stock
        for stock in range(N):
            decision = rng.choice([-1, 0, 1], p=[1/3, 1/3, 1/3])

            if decision == -1:
                # Sell shares
                proc.sell(day, stock, stock_prices, fees, portfolio, ledger)
            elif decision == 1:
                # Buy shares
                proc.buy(day, stock, amount, stock_prices, fees, portfolio, ledger)

    # Sell everything on the last day
    for stock in range(N):
        proc.sell(stock_prices.shape[0]-1, stock, stock_prices, fees, portfolio, ledger)


def crossing_averages(stock_prices, period_sma=200, period_fma=50, amount=5000, fees=20, ledger='ledger_av.txt'):
    '''
    Trade using the crossing averages strategy.

    Input:
        stock_prices (ndarray): the stock price data
        period_sma (int, default 200): period for the slow moving average
        period_fma (int, default 50): period for the fast moving average
        amount (float, default 5000): how much we spend on each purchase
            (must cover fees)
        fees (float, default 20): transaction fees
        ledger (str): path to the ledger file

    Output: None
    '''
    # Create an empty ledger file (if it exists, erase it first)
    open(ledger, 'w').close()

    # Handle possibility of stock_prices being a single column
    ndims = len(stock_prices.shape)     # or ndims = stock_prices.ndim
    if ndims == 1:
        # Make it a 2D array
        stock_prices = stock_prices.reshape((stock_prices.shape[0], 1))

    # Create the portfolio
    N = stock_prices.shape[1]
    portfolio = proc.create_portfolio([amount] * N, stock_prices, fees, ledger)

    for stock in range(N):
        # Get the slow and fast moving averages for each stock
        sma = ind.moving_average(stock_prices[:, stock], n=period_sma)
        fma = ind.moving_average(stock_prices[:, stock], n=period_fma)

        # Make decisions each day, starting from the first day where we have both indicators
        for day in range(period_sma, stock_prices.shape[0]):
            if (fma[day-1] < sma[day-1]) and (fma[day] >= sma[day]):
                # FMA crossing from below: buy
                proc.buy(day, stock, amount, stock_prices, fees, portfolio, ledger)
            elif (fma[day-1] > sma[day-1]) and (fma[day] <= sma[day]):
                # FMA crossing from above: sell
                proc.sell(day, stock, stock_prices, fees, portfolio, ledger)

        # Sell everything on the last day
        proc.sell(stock_prices.shape[0]-1, stock, stock_prices, fees, portfolio, ledger)


def momentum(stock_prices, period=7, osc_type='stochastic', cool_down=10, amount=5000, fees=20, ledger='ledger_mom.txt'):
    '''
    Trade using the momentum strategy, with either the stochastic or RSI oscillator.

    Input:
        stock_prices (ndarray): the stock price data
        period (int, default 7): period for the oscillator
        osc_type (str, default 'stochastic'): either 'stochastic' or 'RSI' to choose an oscillator
        cool_down (int): number of days to wait after crossing the threshold before buying
        amount (float, default 5000): how much we spend on each purchase
            (must cover fees)
        fees (float, default 20): transaction fees
        ledger (str): path to the ledger file

    Output: None
    '''
    # Create an empty ledger file (if it exists, erase it first)
    open(ledger, 'w').close()

    # Handle possibility of stock_prices being a single column
    ndims = len(stock_prices.shape)     # or ndims = stock_prices.ndim
    if ndims == 1:
        # Make it a 2D array
        stock_prices = stock_prices.reshape((stock_prices.shape[0], 1))

    # Create the portfolio
    N = stock_prices.shape[1]
    portfolio = proc.create_portfolio([amount] * N, stock_prices, fees, ledger)

    for stock in range(N):
        # Get the slow and fast moving averages for each stock
        osc = ind.oscillator(stock_prices[:, stock], n=period, osc_type=osc_type)

        # Make decisions each day, starting from the first day where we have the oscillator value
        low_t = 0.25
        high_t = 0.75

        day = period
        while day < stock_prices.shape[0]:
            if (osc[day-1] > low_t) and (osc[day] <= low_t):
                # Crossed the low threshold, buy shares after cool_down days
                day += cool_down
                proc.buy(day, stock, amount, stock_prices, fees, portfolio, ledger)
            elif (osc[day-1] < high_t) and (osc[day] >= high_t):
                # Crossed the high threshold, sell shares
                day += cool_down
                proc.sell(day, stock, stock_prices, fees, portfolio, ledger)
            else:
                day += 1

        # Sell everything on the last day
        proc.sell(stock_prices.shape[0]-1, stock, stock_prices, fees, portfolio, ledger)
