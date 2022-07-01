import pandas as pd

def variance_calculator:
  pass

def time_analysis:
  pass


if __name__ == '__main__':
  test = "src\\datasets\\Glucose\\596\\596-ws-testing.csv"
  test = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose\\596\\596-ws-testing.csv"
  df = pd.read_csv(test, skiprows=1, header=None)
  print(df)