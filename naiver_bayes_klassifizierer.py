import pandas as pd
import math
from collections import Counter, defaultdict

class NaiveBayes:
    def __init__(self, continuous, feature_names):
        # Store which features are continuous (like temperature) and their names
        self.continuous = list(continuous)
        self.feature_names = list(feature_names)
        self.prior = {}
        self.cont_stats = defaultdict(dict)   #will keep the mean and variance for cont features like temperature for each class
        self.disc_counts = defaultdict(lambda: defaultdict(Counter))    #will store count of categorical values like how many yes and no for each feature in each class
        self.class_counts = Counter()   #will count how many samples belong to each class
        self.unique_values = defaultdict(set)   #will keep track of unique possible values like yes or no for every feature, which helps later when calculating probabilty

    def fit(self, df, target_name):
        y = df[target_name].astype(str)
        total = len(y)
            #Calculate prior probabilities (P(yes), P(no))
        for cls, cnt in y.value_counts().items():        
            self.prior[cls] = cnt / total
            self.class_counts[cls] = cnt
            #Group data by class (yes/no)
        grouped = df.groupby(target_name)             
        for cls, group in grouped:
            for i, fname in enumerate(self.feature_names):
                if self.continuous[i]:
                    col = pd.to_numeric(group[fname], errors='coerce').dropna()
                    mu = col.mean() if len(col) > 0 else 0.0    #mean
                    var = col.var(ddof=0) if len(col) > 0 else 1e-9    #variance
                    if var == 0 or pd.isna(var):    #avoid divide by 0
                        var = 1e-9
                    self.cont_stats[cls][fname] = (mu, var)
                else:
                     #For yes/no type features — count values
                    vals = group[fname].astype(str).str.strip().str.lower().tolist()
                    self.disc_counts[cls][fname].update(vals)
                    self.unique_values[fname].update(vals)

    def _log_gauss(self, x, mu, var):
        eps = 1e-9
        var = var if var > 0 else eps
        return -0.5 * math.log(2 * math.pi * var) - ((x - mu) ** 2) / (2 * var)

    def predict_probability(self, df):
        # Predict probabilities for each row
        X = df[self.feature_names]
        classes = list(self.prior.keys())
        rows = []
        #For each patient
        for _, row in X.iterrows():  
            log_probs = []
            for cls in classes:
                #Start with prior (P(class))
                lp = math.log(self.prior.get(cls, 1e-9))    
                for i, fname in enumerate(self.feature_names):
                    if self.continuous[i]:  #Continuous feature
                        try:
                            x = float(row[fname])
                        except:
                            continue
                        mu, var = self.cont_stats[cls].get(fname, (0.0, 1e-9))
                        lp += self._log_gauss(x, mu, var)
                    else:
                        sval = str(row[fname]).strip().lower()  #Categorical feature (yes/no)
                        counts = self.disc_counts[cls][fname]
                        total = sum(counts.values())
                        n_unique = len(self.unique_values[fname]) if len(self.unique_values[fname]) > 0 else 1
                        prob = (counts.get(sval, 0) + 1) / (total + n_unique)
                        lp += math.log(prob)
                log_probs.append(lp)
            #Convert log-probabilities into normal probabilities
            m = max(log_probs)
            exps = [math.exp(lp - m) for lp in log_probs]
            s = sum(exps)
            probs = [e / s for e in exps]
            # Store probabilities and prediction
            prob_map = {classes[i]: probs[i] for i in range(len(classes))}
            pred = max(prob_map, key=prob_map.get)
            rowres = {"predicted_label": pred}
            rowres.update({f"prob_{c}": prob_map[c] for c in classes})
            rows.append(rowres)
        return pd.DataFrame(rows)

    def evaluate_on_data(self, df, target_name):
            #Check how well model predicts on test data
        preds = self.predict_probability(df)
        predicted = preds["predicted_label"].tolist()
        actual = df[target_name].astype(str).tolist()
            #Count how many predictions are correct
        correct = sum(1 for a, p in zip(actual, predicted) if a == p)
        acc = correct / len(actual) if len(actual) > 0 else 0.0
            #Create confusion matrix (true vs predicted)
        classes = sorted(list(set(actual) | set(predicted)))
        cm = pd.DataFrame(0, index=classes, columns=classes)
        for a, p in zip(actual, predicted):
            cm.loc[a, p] += 1
        return acc, cm, preds
