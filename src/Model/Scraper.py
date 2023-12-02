import yfinance as yf

class Scraper:
    #Class to scrape stock data using yfinance, with customizable stock symbol, time frame, and interval.

    def __init__(self, stock_name="aapl", time_frame="max", search_intensity="1d"):
        #Initializes the scraper with the stock symbol, time period, and data point interval.
        self.stock_name = stock_name
        self.time_frame = time_frame
        self.search_intensity = search_intensity
        stock = yf.Ticker(stock_name)
        self.stock_historical = stock.history(period=time_frame, interval=search_intensity)
    
    def get_all(self):
        #Returns the entire historical data as a DataFrame.
        return self.stock_historical
    
    def get_close(self):
        #Returns a Series of the historical closing prices.
        return self.stock_historical['Close']
    
    def get_range(self, start_date, end_date, search_intensity="1d"):
        #Fetches and returns historical data for a specified date range.
        new_stock = yf.Ticker(self.stock_name)
        self.new_stock_historical = new_stock.history(start=start_date, end=end_date, interval=search_intensity)
        return self.new_stock_historical
