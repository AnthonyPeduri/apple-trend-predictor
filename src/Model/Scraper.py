import yfinance as yf
import pandas as pd
import sys 


class Scraper:
    def __init__(self, stock_name = "aapl", time_frame = "max", search_intesity = "1d"):
        self.stock_name = stock_name
        self.time_frame = time_frame
        self.search_intesity = search_intesity
        stock = yf.Ticker(stock_name)
        self.stock_historical = stock.history(period=time_frame, interval=search_intesity)
    
    def get_all(self):
        return self.stock_historical
    
    def get_close(self):
        return self.stock_historical['Close']
    
    def get_range(self, start_date, end_date, search_intesity = "1d"):
        new_stock = yf.Ticker(self.stock_name)
        self.new_stock_historical = new_stock.history(start = start_date, end = end_date, interval= search_intesity)
        return self.new_stock_historical