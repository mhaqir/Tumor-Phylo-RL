import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

##################################### importing the values and calculating the deviation

test_10x10_004 = pd.read_csv("/u/mhaghir/neural-combinatorial-optimization-rl-tensorflow/ResultsCarbonate/test_10x10_NLL_notequal/test_10x10_0.004.csv",\
                        sep = ",")     # set the directory of file here
f_1_0_o = test_10x10_004["f_1_0_o"].values
f_0_1_o = test_10x10_004["f_0_1_o"].values

f_1_0_rl = test_10x10_004["f_1_0_rl"].values
f_0_1_rl = test_10x10_004["f_0_1_rl"].values

diff_1_to_0 = np.abs(f_1_0_rl - f_1_0_o)
diff_0_to_1 = np.abs(f_0_1_rl - f_0_1_o)


df1 = pd.DataFrame(index = ["test" + str(i) for i in range(len(diff_1_to_0))], columns = ["diff_1_to_0", "diff_0_to_1"])
df1["diff_1_to_0"] = diff_1_to_0
df1["diff_0_to_1"] = diff_0_to_1

float_col1 = df1.select_dtypes(include = ['float64']) # This will select float columns only
for col in float_col1.columns.values:
    df1[col] = df1[col].astype('int64')

#################################### ploting the results

f, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=False)  # set the number of subplots here
plt.subplots_adjust(hspace = 0.35)


sns.countplot(x = "variable", hue = "value", data = pd.melt(df1), ax = axes[0,0])
axes[0,0].set_title("10x10, " + "betta=0.004, Solved: " + str(len(df1)))
axes[0,0].set(xlabel = "Deviation from optimal number of flips" , ylabel = "Frequency (out of 100 instances)")
axes[0,0].legend(loc='upper right')
