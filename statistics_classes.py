import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/hand_data.csv')


cols_x = [c for c in df.columns if '_x' in c]
cols_y = [c for c in df.columns if '_y' in c]


centres = pd.DataFrame()
centres['mean_x'] = df.groupby('label')[cols_x].mean().mean(axis=1)
centres['mean_y'] = df.groupby('label')[cols_y].mean().mean(axis=1)

print("Coordonnées du 'Centre' moyen pour chaque posture :")
print(centres)


dispersion = df.groupby('label')[cols_x + cols_y].std().mean(axis=1)
print("\nDispersion moyenne des points (étalement de la main) :")
print(dispersion)


plt.figure(figsize=(10, 6))
sns.scatterplot(data=centres, x='mean_x', y='mean_y', hue='label', s=200)
plt.title("Position du centre de gravité moyen par posture")
plt.show()


