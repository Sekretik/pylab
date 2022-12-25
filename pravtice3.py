import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bar = pd.read_csv('lessons\\pr\\flavors_of_cacao.csv',
                       sep=',', header=0, names=['company', 'bar_name','ref', 'rew_date', 'percent', 'company_loc', 'rate', 'bean_dtype', 'bean_orig'])

rates = bar['rate']

# plt.hist(rates, bins=5)
# plt.show()

bean_percent = bar['percent'].str.strip('%').astype(float) / 100

# plt.hist(bean_percent, bins=10)
# plt.show()

# plt.scatter(rates,bean_percent*100)
# plt.show()

# plt.boxplot(rates)
# plt.show()

# plt.boxplot(bean_percent)
# plt.show()

# bean_origin = bar['bean_orig']

# bean_origin = bar.filter(['bean_orig'])

# bean_origin['bean_orig'] = bean_origin['bean_orig'].replace('', np.nan).replace('\xa0', np.nan)

# bean_origin.dropna(subset=['bean_orig'], inplace=True)

# rarity = bean_origin['bean_orig'].value_counts().reset_index(level=0)

# count = rarity.where(rarity['bean_orig'] < 6).dropna().count()['index']

# rarity = rarity.drop(rarity[rarity['bean_orig'] < 6].index)

# rarity.loc[len(rarity)] = ['Other',count]

# plt.pie(rarity['bean_orig'],labels=rarity['index'])
# plt.show()


# mean_rates = bar.groupby('bean_orig')['rate'].mean()
# median_rates = bar.groupby('bean_orig')['rate'].median()

# top3 = mean_rates.sort_values(ascending=False).head(3)
# print(top3)

mean_bar_rates = bar.groupby('bar_name')['rate'].median()
mean_bar_rates = bar.groupby('bar_name')['rate'].mean()

top3bars = mean_bar_rates.sort_values(ascending=False).head(3)

print(top3bars)