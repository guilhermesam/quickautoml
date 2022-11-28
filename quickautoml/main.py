from quickautoml.estimators import make_classifier
import pandas as pd


if __name__ == "__main__":
    model = make_classifier()
    df = pd.read_csv("../datasets/drebin215dataset5560malware9476benign.csv")
    model.fit(df.drop(["TelephonyManager.getSimCountryIso", "class"], axis=1), df["class"])
    print(model.best_model)
