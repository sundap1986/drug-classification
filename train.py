# Read the Dataset
import pandas as pd

df = pd.read_csv("./data/raw/drug200.csv", delimiter="\t")

# Information about the dataset
df.info()
# Shape of the Dataset
df.shape
# Display the first 5 Rows
df.head()
# Display the Last 5 Rows
df.tail()
# Check the Null values in the dataset
df.isnull().sum()
# Desccribe the Dataset for Numerical Columns
df.describe().T
# Desccribe the Dataset for Categorical Columns
df.describe(include="object").T
# Divide the Categorical and Numerical Columns
cat_cols = df.select_dtypes(include="object").columns
print(f"Categorical columns are: {cat_cols}")

num_cols = df.select_dtypes(exclude="object").columns
print(f"Numerical columns are: {num_cols}")
# Label Encoding the Categorical Columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["BP"] = le.fit_transform(df["BP"])
df["Cholesterol"] = le.fit_transform(df["Cholesterol"])
df["Drug"] = le.fit_transform(df["Drug"])

# Display the values
df.head()
# Normalize the Dataset using MinMaxscaler.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()
# Split the Data into Training and Testing
from sklearn.model_selection import train_test_split

X = df.drop("Drug", axis=1)
y = df["Drug"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the Model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

model = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
}

for name, clf in model.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy for {name} model: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report for {name} model:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for {name} model:\n{confusion_matrix(y_test, y_pred)}")
    print(f"F1 Score for {name} model: {f1_score(y_test, y_pred, average='weighted')}")
    print(
        f"Precision Score for {name} model: {precision_score(y_test, y_pred, average='weighted')}"
    )
    print(f"Recall Score for {name} model: {recall_score(y_test, y_pred, average='weighted')}")
    print("\n")
    # Select the Model : RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# Write the Metrics
with open("./reports/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
    f.write(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
    f.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}\n")
    f.write(f"Precision Score: {precision_score(y_test, y_pred, average='weighted')}\n")
    f.write(f"Recall Score: {recall_score(y_test, y_pred, average='weighted')}\n")
    f.write(f"Model Used: {model}\n")
    # Confusion Matrix Display using Plot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Save this plot image
plt.savefig("./reports/confusion_matrix.png")
# Save the MOdel using SKOPS
import skops.io as sio

sio.dump(model, "./models/drug-classification.skops")
