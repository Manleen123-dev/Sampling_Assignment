import pandas as pd
df = pd.read_csv("../data/Creditcard_data.csv")

print(df.shape)
print(df.head())
print(df['Class'].value_counts())
X = df.drop('Class', axis=1)
y = df['Class']
print(X.shape)
print(y.shape)
####################################################################################################
from imblearn.over_sampling import RandomOverSampler

print("Starting Random Over Sampling...")

ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y)

print(y_ros.value_counts())
#######################################################################################################
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)

print(y_rus.value_counts())
#####################################################################################################
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)

print(y_sm.value_counts())
###########################################################################################################
from imblearn.under_sampling import NearMiss

nearmiss = NearMiss()
X_nm, y_nm = nearmiss.fit_resample(X, y)

print(y_nm.value_counts())
############################################################################################################
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for train_index, _ in sss.split(X, y):
    X_ss = X.iloc[train_index]
    y_ss = y.iloc[train_index]

print(y_ss.value_counts())
##############################################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

models = {
    "M1_LogisticRegression": LogisticRegression(max_iter=1000),
    "M2_DecisionTree": DecisionTreeClassifier(random_state=42),
     "M3_RandomForest": RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
),
    "M4_SVM": SVC(),
    "M5_KNN": KNeighborsClassifier()
}
samples = {
    "Sampling1_ROS": (X_ros, y_ros),
    "Sampling2_RUS": (X_rus, y_rus),
    "Sampling3_SMOTE": (X_sm, y_sm),
    "Sampling4_NearMiss": (X_nm, y_nm),
    "Sampling5_Stratified": (X_ss, y_ss)
}
results = []
for model_name, model in models.items():
    for sample_name, (X_s, y_s) in samples.items():

        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s, test_size=0.3, random_state=42, stratify=y_s
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append({
            "Model": model_name,
            "Sampling": sample_name,
            "Accuracy": acc
        })

        print(model_name, sample_name, acc)

        
     
results_df = pd.DataFrame(results)
results_df.to_csv("../results/accuracy_table.csv", index=False)

print("Accuracy table saved to results/accuracy_table.csv")
################################################################################################################

import matplotlib.pyplot as plt
plt.figure()
df['Class'].value_counts().plot(kind='bar')
plt.title("Class Distribution Before Sampling")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../results/class_distribution_before.png")
plt.close()
plt.figure()
y_ros.value_counts().plot(kind='bar')
plt.title("Class Distribution After Random Over Sampling")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../results/class_distribution_after_ros.png")
plt.close()
plt.figure()
y_sm.value_counts().plot(kind='bar')
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../results/class_distribution_after_smote.png")
plt.close()
############################################################################################################

import numpy as np

acc_df = pd.read_csv("../results/accuracy_table.csv")

pivot_df = acc_df.pivot(
    index="Model",
    columns="Sampling",
    values="Accuracy"
)
plt.figure()
plt.imshow(pivot_df.values)
plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45)
plt.yticks(range(len(pivot_df.index)), pivot_df.index)
plt.colorbar(label="Accuracy")
plt.title("Accuracy Heatmap (Models vs Sampling Techniques)")
plt.tight_layout()
plt.savefig("../results/accuracy_heatmap.png")
plt.close()
plt.figure()
plt.axis('off')
###############################################################################################################
table = plt.table(
    cellText=np.round(pivot_df.values, 3),
    rowLabels=pivot_df.index,
    colLabels=pivot_df.columns,
    loc='center'
)

table.scale(1, 1.5)
plt.title("Accuracy Matrix Table")
plt.tight_layout()
plt.savefig("../results/accuracy_matrix_table.png")
plt.close()
##############################################################
sampling_mean = acc_df.groupby("Sampling")["Accuracy"].mean()
sampling_mean_df = sampling_mean.reset_index()

sampling_mean_df.to_csv("../results/sampling_mean_accuracy.csv", index=False)
plt.figure()
sampling_mean.plot(kind='bar')
plt.title("Mean Accuracy per Sampling Technique")
plt.xlabel("Sampling Technique")
plt.ylabel("Mean Accuracy")
plt.tight_layout()
plt.savefig("../results/sampling_mean_accuracy.png")
plt.close()
model_mean = acc_df.groupby("Model")["Accuracy"].mean()

plt.figure()
model_mean.plot(kind='bar')
plt.title("Mean Accuracy per Model")
plt.xlabel("Model")
plt.ylabel("Mean Accuracy")
plt.tight_layout()
plt.savefig("../results/model_mean_accuracy.png")
plt.close()
################################################################################
model_wise = acc_df.pivot(index="Model", columns="Sampling", values="Accuracy")

plt.figure()
plt.axis('off')

table = plt.table(
    cellText=model_wise.round(3).values,
    rowLabels=model_wise.index,
    colLabels=model_wise.columns,
    loc='center'
)

table.scale(1, 1.5)
plt.title("Model-wise Accuracy Comparison")
plt.tight_layout()
plt.savefig("../results/model_wise_accuracy_table.png")
plt.close()
########################################################################################
plt.figure()

for model in acc_df["Model"].unique():
    subset = acc_df[acc_df["Model"] == model]
    plt.plot(
        subset["Sampling"],
        subset["Accuracy"],
        marker='o',
        label=model
    )

plt.title("Accuracy Trend Across Sampling Techniques")
plt.xlabel("Sampling Technique")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("../results/accuracy_trend_across_sampling.png")
plt.close()
############################################################################################
plt.figure()
acc_df.boxplot(column="Accuracy", by="Sampling")
plt.title("Accuracy Distribution per Sampling Technique")
plt.suptitle("")
plt.xlabel("Sampling Technique")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/accuracy_boxplot_sampling.png")
plt.close()
#############################################################################################
plt.figure()
acc_df.boxplot(column="Accuracy", by="Model")
plt.title("Accuracy Distribution per Model")
plt.suptitle("")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/accuracy_boxplot_model.png")
plt.close()
# ============================================================
# Best Sampling Technique per Model
# ============================================================

best_sampling = acc_df.loc[
    acc_df.groupby("Model")["Accuracy"].idxmax()
]

best_sampling.to_csv("../results/best_sampling_per_model.csv", index=False)

print("best_sampling_per_model.csv generated")
