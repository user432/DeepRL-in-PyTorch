import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import glob


data_1 = glob.glob("runs_JSON/SAC/*.json")
prep_data = []

for name in data_1:
    with open(name) as json_file:
        data_i = json.load(json_file)
    data_i = np.array(data_i)
    prep_data.append(list(data_i[:,2]))

fig = plt.figure(figsize=(10,7))
#xdata = np.arange(len(prep_data[0]))
data = prep_data
linestyle = '-.'
color = 'r'
label = 'SAC'
sns.set(style="darkgrid", font_scale=1.5)
sns.tsplot( data=data, color=color, linestyle=linestyle, condition=label)

plt.ylabel("Reward", fontsize=12)
plt.xlabel("Episodes", fontsize=12)
plt.title("Pendulum Env", fontsize=14)

plt.legend(loc='lower right')
#plt.savefig("GA_with_5_seeds_run.jpg", dpi=300)
plt.show()
