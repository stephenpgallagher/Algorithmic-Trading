# Module to generate some simulation data.
import numpy as np
import matplotlib.pyplot as plt

def news(chance, volatility, rng):
    '''
    Simulate the positive or negative impact of the news
    over the share price of a stock.

    Input:
        chance (float): probability (in %) that there are important news on a given day
        volatility (float): volatility (std) of the affected stock
        rng (generator): random number generator

    Output:
        impact (ndarray): array of drift values over duration of event
        or None: if there are no important news at all
    '''
    # Is there news today?
    news_today = rng.choice([0, 1], p=[1 - 0.01*chance, 0.01*chance])

    if news_today:
        # Generate a random duration and m for the news
        duration = rng.integers(3, 15)
        m = rng.normal(0, 2)

        # Calculate the array of drift values
        drift = m * volatility
        impact = [drift] * duration

        return impact


def generate_stock_prices(number_of_days, initial_price, volatility):
    '''
    Generates daily adjusted closing prices for N different stocks,
    for a given number of days.

    Stock prices are assumed to follow a random walk pattern, with normally
    distributed daily share price changes, with mean 0 and standard deviation representing
    the volatility.

    If a share price reaches 0, the company fails, and no further share prices
    are simulated.

    Input:
        number_of_days (int): number of days for which to simulate data
        initial_price (list): the N stock prices at the start
        volatility (list): standard deviations of the daily share price increments

    Output:
        stock_prices (ndarray): a Numpy array with "duration" rows and N columns,
            each column representing the adjusted closing price of a stock over time.
    '''

    # Initialise the arrays
    N = len(initial_price)
    stock_prices = np.zeros([number_of_days, N], dtype=float)
    stock_prices[0, :] = initial_price
    drift = np.zeros([number_of_days, N], dtype=float)

    # Initialise the random number generator
    rng = np.random.default_rng()

    # Loop over the days
    for day in range(1, number_of_days):
        # Get the prices before news
        stock_prices[day, :] = stock_prices[day-1, :] + rng.normal(0, volatility, size=N)

        # Take news into account for each stock
        for n in range(N):
            current_drift = news(1, volatility[n], rng)
            if current_drift is not None:
                # New piece of news today
                duration = min(len(current_drift), number_of_days-day)
                drift[day:day+duration, n] += current_drift[:duration]

            stock_prices[day, n] += drift[day, n]

    # Check if companies have failed
    if np.any(stock_prices<=0):
        stock_prices[np.argmax(stock_prices<=0):, :] = np.nan

    return stock_prices


def get_data(method='read', initial_price=None, volatility=None):
    '''
    Generates or reads simulation data for one or more stocks over 5 years,
    given their initial share price and volatility.

    Input:
        method (str): either 'generate' or 'read' (default 'read').
            If method is 'generate', use generate_stock_price() to generate
                the data from scratch.
            If method is 'read', use Numpy's loadtxt() to read the data
                from the file stock_data_5y.txt.

        initial_price (list): list of initial prices for each stock (default None)
            If method is 'generate', use these initial prices to generate the data.
            If method is 'read', choose the column in stock_data_5y.txt with the closest
                starting price to each value in the list, and display an appropriate message.

        volatility (list): list of volatilities for each stock (default None).
            If method is 'generate', use these volatilities to generate the data.
            If method is 'read', choose the column in stock_data_5y.txt with the closest
                volatility to each value in the list, and display an appropriate message.

        If no arguments are specified, read price data from the whole file.

    Output:
        sim_data (ndarray): NumPy array with N columns, containing the price data
            for the required N stocks each day over 5 years.

    Examples:
    Returns an array with 2 columns:
    >>> sim_data = get_data(method='generate', initial_price=[150, 250], volatility=[1.8, 3.2])
    >>> print(sim_data[0, :])
    [150. 250.]

    Displays a message and returns None:
    >>> get_data(method='generate', initial_price=[150, 200])
    Please specify the volatility for each stock.

    Displays a message and returns None:
    >>> get_data(method='generate', volatility=[3])
    Please specify the initial price for each stock.

    Returns an array with 2 columns and displays a message:
    >>> get_data(method='read', initial_price=[210, 58])
    Found data with initial prices [200, 50] and volatilities [1.5, 0.7].

    Returns an array with 1 column and displays a message:
    >>> get_data(volatility=[5.1])
    Found data with initial prices [850] and volatilities [5].

    If method is 'read' and both initial_price and volatility are specified,
    volatility will be ignored (a message is displayed to indicate this):
    >>> get_data(initial_price=[210, 58], volatility=[5, 7])
    Found data with initial prices [200, 50] and volatilities [1.5, 0.7].
    Input argument volatility ignored.

    No arguments specified, all default values, returns price data for all stocks in the file:
    >>> get_data()
    '''
    # Read data from file
    if method == 'read':
        all_sim_data = np.loadtxt('stock_data_5y.txt')

        if initial_price is None:
            if volatility is None:
                # Default values: read all data
                return all_sim_data[1:, :]

            else:
                # Get the column numbers to extract
                columns = []
                for v in volatility:
                    # Find the closest volatility
                    columns.append(np.argmin(np.abs(v - all_sim_data[0, :])))
                print(f'Found data with initial prices {all_sim_data[1, columns]} and volatilities {all_sim_data[0, columns]}.')
                return all_sim_data[1:, columns]
        else:
            # initial_price is specified
            columns = []
            for p in initial_price:
                # Find the closest initial price
                columns.append(np.argmin(np.abs(p - all_sim_data[1, :])))
            print(f'Found data with initial prices {all_sim_data[1, columns]} and volatilities {all_sim_data[0, columns]}.')

            if volatility is not None:
                print('Input argument volatility ignored.')

            return all_sim_data[1:, columns]

    # Generate data from scratch
    elif method == 'generate':
        # Check input arguments
        if initial_price is None:
            print('Please specify the initial price for each stock.')
            return
        if volatility is None:
            print('Please specify the volatility for each stock.')
            return

        # Use generate_stock_prices() to generate the data
        return generate_stock_prices(5*365, initial_price, volatility)
