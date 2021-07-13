import numpy as np

def moving_average(stock_price, n=7, weights=[]):
    '''
    Calculates the n-day (possibly weighted) moving average for a given stock over time.

    Input:
        stock_price (ndarray): single column with the share prices over time for one stock,
            up to the current day.
        n (int, default 7): period of the moving average (in days).
        weights (list, default []): must be of length n if specified. Indicates the weights
            to use for the weighted average. If empty, return a non-weighted average.

    Output:
        ma (ndarray): the n-day (possibly weighted) moving average of the share price over time.
    '''
    # Initialise the array
    days = stock_price.shape[0]
    ma = np.zeros(stock_price.shape)

    if not weights:
        weights = None

    # Loop over the days, starting at day n-1
    ma[:n-1] = np.nan
    for d in range(n-1, days):
        ma[d] = np.average(stock_price[d-n+1:d+1], weights=weights)

    return ma

def oscillator(stock_price, n=7, osc_type='stochastic'):
    '''
    Calculates the level of the stochastic or RSI oscillator with a period of n days.

    Input:
        stock_price (ndarray): single column with the share prices over time for one stock,
            up to the current day.
        n (int, default 7): period of the moving average (in days).
        osc_type (str, default 'stochastic'): either 'stochastic' or 'RSI' to choose an oscillator.

    Output:
        osc (ndarray): the oscillator level with period $n$ for the stock over time.
    '''
    # Initialise oscillator array
    days = stock_price.shape[0]
    osc = np.zeros(days)

    for d in range(n-1, days):
        if osc_type == 'stochastic':
            # Find price range over past n days
            highest = np.max(stock_price[d-n+1:d+1])
            lowest = np.min(stock_price[d-n+1:d+1])
            delta = stock_price[d] - lowest
            deltamax = highest - lowest

            # Make sure the stochastic oscillator is defined
            if deltamax == 0:
                osc[d] = np.nan
            else:
                osc[d] = delta / deltamax

        elif osc_type == 'RSI':
            # Find all price differences
            diffs = np.diff(stock_price[d-n+1:d+1])
            #  diffs = stock_price[d-n+1:d+1] - stock_price[d-n:d]

            # Find average positive and negative differences
            if np.any(diffs > 0):
                pos_average = np.mean(diffs[diffs > 0])
            else:
                pos_average = 0

            if np.any(diffs < 0):
                neg_average = np.abs(np.mean(diffs[diffs < 0]))
            else:
                neg_average = 0

            # Calculate RSI
            if pos_average == 0 and neg_average == 0:
                osc[d] = np.nan
            elif neg_average == 0:
                osc[d] = 1
            else:
                osc[d] = 1 - 1 / (1 + pos_average / neg_average)

    # Set undefined values to nan
    osc[:n-1] = np.nan
    return osc
