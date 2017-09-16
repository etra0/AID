import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# primero leemos la data
main_df = pd.read_csv('data/student-mat.csv', sep=';')
main_df.info()
