import yfinance as yf
import pandas as pd

class Scraper:
    def __init__(self, stock_name = "aapl", time_frame = "max", search_intesity = "1d"):
        self.stock_name = stock_name
        self.time_frame = time_frame
        self.search_intesity = search_intesity
        stock = yf.Ticker(stock_name)
        self.stock_historical = stock.history(period=time_frame, interval=search_intesity)
    
    def getAll(self):
        return self.stock_historical
    
    def getClose(self):
        return self.stock_historical['Close']