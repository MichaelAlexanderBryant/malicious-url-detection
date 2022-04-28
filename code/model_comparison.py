import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_no_sampling = pd.read_csv('../output/modeling/no_sampling/df_scores.csv')
df_upsampling = pd.read_csv('../output/modeling/upsampling/df_scores.csv')
df_downsampling = pd.read_csv('../output/modeling/downsampling/df_scores.csv')

df = pd.concat([df_no_sampling, df_upsampling, df_downsampling], axis=0)

