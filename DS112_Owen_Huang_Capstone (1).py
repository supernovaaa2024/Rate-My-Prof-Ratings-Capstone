#%%
# Capstone Project
# Owen Huang 
# NetID: yh4842
# N number: N17864220
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import random
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
random.seed(17864220)
np.random.seed(17864220)
seed = 17864220


#%%

#1 Read files and label the columns so we know the meaning of each column.
rmpCapstoneNum = pd.read_csv("rmpCapstoneNum.csv", header=None, names=["Average Rating","Average Difficulty","Number of Ratings","Received a pepper?", "The proportion of students that said they would take the class again", "The number of ratings coming from online classes", "Male Gender", "Female Gender"])
rmpCapstoneQual = pd.read_csv("rmpCapstoneQual.csv", header=None, names=["Major/Field", "University", "US State (2 letter abbreviation)"])
rmpCapstonedata_combined = pd.concat([rmpCapstoneNum, rmpCapstoneQual], axis=1) # Combined dataset for more interesting analyis later - Extra Credit

# Preprocessing - Handling Missing Values & Evaluating the Distribution of Data
missingPercentage = (rmpCapstoneNum.isnull().sum() / len(rmpCapstoneNum)) * 100
print(f"Missing Percentages (%):\n{missingPercentage}")
# Since there is missing data, we should not include the rows with missing data and if we impute any data then the result will be skewed and not accurate.
# More ratings the more accurate the average ratings is, and since some professors only has a few ratings, we will need to exclude them. However, there is a tradeoff. If we only include professors with a high number of ratings, the number of professors decrease dramatically, decreasing power when conducting hypothesis testing.
# Therefore, a threshold must be determined. Below I plotted a graph of the number of data points that remains when the threshold increases.
thresholds = []
for threshold in range(1, int(max(rmpCapstoneNum['Number of Ratings']))):
    filtered = rmpCapstoneNum[rmpCapstoneNum['Number of Ratings'] >= threshold]
    length = len(filtered)
    thresholds.append(length)
plt.figure(figsize=(8, 6))
plt.plot(range(1, int(max(rmpCapstoneNum["Number of Ratings"]))), thresholds)
plt.xticks(np.arange(0, int(max(rmpCapstoneNum["Number of Ratings"])) + 1, 20))
plt.xlabel("Threshold")
plt.ylabel("Number of data points")
plt.show()
# Here, we see that the number of data points we have drops significantly (exponentially) when we increase the threshold. Therefore, we still have to set a relatively low threshold. We will set it to 3 because it will still give quite a large number of data points of 25368.
filtered_dataframe = rmpCapstoneNum[rmpCapstoneNum['Number of Ratings'] >= 3]
print(f"There are {len(filtered_dataframe)} datapoints")

print(f"After Filtering: Missing Values:\n{filtered_dataframe.isnull().sum()}")  # Check if there are any missing values after filtering

# Plot the distribution of average ratings to see if they are normally distributed.
plt.figure(figsize=(10,6))
sns.histplot(filtered_dataframe["Average Rating"], bins=90, kde=True)
plt.title('Distribution of Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()
print("From the distritution of the average ratings, we see that the avgerage ratings are not normally distributed. Violating the assumption of normality of data required for parametric tests.")

# Check for multicollinearity - Needed when performing Regression
plt.figure(figsize=(6,4))
sns.heatmap(filtered_dataframe.dropna().corr(), annot=True, cmap="coolwarm", fmt=".3f")
plt.title("Pearson Correlation Heatmap for Features")
plt.show()

# Further Inspection - Ambiguous Genders
# Further examination of the gender of the professors reveals that there are professors who are marked as both male and female or there is no gender assigned. 
# They will be dropped when performing multiple regression and classification but will be included when gender is not a dependent variable used in regression or classification.
filtered_no_ambiguous_genders = filtered_dataframe[((filtered_dataframe["Male Gender"] == 1) & (filtered_dataframe["Female Gender"] == 0)) | ((filtered_dataframe["Male Gender"] == 0) & (filtered_dataframe["Female Gender"] == 1))] # Here we filter out the ratings where both male and female are denoted as 1 because it is ambiguous so we don't know their actual gender.

#%%
# Question 1
# We can either use parametric test or non-parametric test. There are a few assumptions for parametric tests: 1. Normality, 2. Homogeneity of variance 3. The mean is meaningful
# Since ratings are ordinal data not cardinal, the mean is not meaninful. Hence, we will use median instead of the mean and perform non-parametric test.
# Independent samples using Mann-Whitney U test
# Plot the distribution of average ratings

# Filter out the two Groups - male and female ratings
male_ratings = filtered_no_ambiguous_genders[filtered_no_ambiguous_genders["Male Gender"] == 1]["Average Rating"]
female_ratings = filtered_no_ambiguous_genders[filtered_no_ambiguous_genders["Female Gender"] == 1]["Average Rating"]
print("Median rating for male professors:", np.median(male_ratings))
print("Median rating for female professors:", np.median(female_ratings))

# Left-tailed Mann-Whitney U test
u_stat1, p_value1 = mannwhitneyu(male_ratings, female_ratings, alternative="greater")
print("Mann-Whitney U Test Statistic:", u_stat1)
print(f"The p-value is {p_value1}, which is less than 0.005 so the test-statistic is statistically significant and so we can drop the null hypothesis that there is no pro-male gender bias and conclude that male Professors receive higher ratings than female Professors.")

# two-sample KS test on male and female ratings to ensure that the two distributions are not the same.
from scipy.stats import ks_2samp
ks_stat, ks_p_value = ks_2samp(male_ratings, female_ratings)
print("KS Test statistic:", ks_stat)
print("p-value:", ks_p_value)
print("The p_value is less than 0.005. Therefore, there is a statistically significant difference between the two rating distributions.")

# Plot of the two distribution to visualize the differences
plt.figure(figsize=(10,6))
sns.histplot(male_ratings, bins=90, kde=True, label="Male Ratings")
sns.histplot(female_ratings, bins=90, kde=True, label="Female Ratings")
plt.title("Distribution of Male and Female Ratings")
plt.legend()
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.show()

# However, we should also address the concern of confounders. For example, it might be that female professors have less teaching experience. So, it is actually the less teaching experience that leads to the lower ratings for female professors.
# First model dropping Female Gender
df1 = filtered_no_ambiguous_genders.dropna()
X_all_1 = df1.drop(columns=["Average Rating", "Female Gender"])
Y = df1["Average Rating"]

# Standardize the Features
scaler = StandardScaler()
X_train, X_test, Y_train, Y_test = train_test_split(X_all_1, Y, train_size=0.8, random_state=17864220)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Multiple regression model
Model1_Multi = LinearRegression().fit(X_train_scaled, Y_train)
predictions_multi = Model1_Multi.predict(X_test_scaled)
rmse_multi = np.sqrt(mean_squared_error(Y_test, predictions_multi))
r2_multi = r2_score(Y_test, predictions_multi)

print(f"RMSE of this Multiple Regression Model: {rmse_multi:.4f}")
print(f"R-squared of this Multiple Regression Mode: {r2_multi:.4f}")
print("Coefficients:")
print(pd.Series(Model1_Multi.coef_, index=X_all_1.columns))
print(f"Intercept: {Model1_Multi.intercept_:.4f}")

# Second model dropping "Male Gender"
df1 = filtered_no_ambiguous_genders.dropna()
X_all_1 = df1.drop(columns=["Average Rating", "Male Gender"])
Y = df1["Average Rating"]

# Standardize the Features
scaler = StandardScaler()
X_train, X_test, Y_train, Y_test = train_test_split(X_all_1, Y, train_size=0.8, random_state=17864220)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Multiple regression model
Model1_Multi = LinearRegression().fit(X_train_scaled, Y_train)
predictions_multi = Model1_Multi.predict(X_test_scaled)
rmse_multi = np.sqrt(mean_squared_error(Y_test, predictions_multi))
r2_multi = r2_score(Y_test, predictions_multi)

print(f"RMSE of this Multiple Regression Model: {rmse_multi:.4f}")
print(f"R-squared of this Multiple Regression Mode: {r2_multi:.4f}")
print("Coefficients:")
print(pd.Series(Model1_Multi.coef_, index=X_all_1.columns))
print(f"Intercept: {Model1_Multi.intercept_:.4f}")




#%%
# Question 2. 

# First, we examine any missing values.
experience = filtered_dataframe["Number of Ratings"]
quality = filtered_dataframe["Average Rating"]
missing_experience = experience.isnull().sum()
missing_quality = quality.isnull().sum()
print(f"Missing values in experience: {missing_experience}")
print(f"Missing values in quality: {missing_quality}")
exp_quality = pd.DataFrame({'Experience': experience, 'quality':quality})

# We plot the quality of the teaching (rating) against the experience of the Professor so we better understand it. We see that when the number of ratings is below about 70, there is a large variability and inconsistency of their ratings, and only whne the number of ratings exceed 70 that we start to see a trend of high ratings.
plt.scatter(x=experience, y=quality, s=7, alpha=0.5)
plt.xlabel("Experience (number of ratings)")
plt.ylabel("Quality")
plt.show()

print(f"Median: {np.nanmedian(experience)}")
print(f"Mean: {np.nanmean(experience)}")
# Plotting it and separating the groups
filtered_exp_quality_greaterthan70 = exp_quality[exp_quality["Experience"] >= 70] # First Group - More Experienced
filtered_exp_quality_lessthan70 = exp_quality[exp_quality["Experience"] < 70] # Second Group - Less Experienced
plt.scatter(filtered_exp_quality_greaterthan70["Experience"], filtered_exp_quality_greaterthan70["quality"], alpha=0.8, label="More Experienced (More than 70 ratings)", s=7)
plt.scatter(filtered_exp_quality_lessthan70["Experience"], filtered_exp_quality_lessthan70["quality"], alpha=0.8, label="Less Experienced (Less than 70 ratings)", s=7)
plt.xlabel("Experience (number of ratings)")
plt.ylabel("Quality")
plt.legend()
plt.show()

# U Test on the two groups
u_stat70, p_value70 = mannwhitneyu(filtered_exp_quality_greaterthan70["quality"], filtered_exp_quality_lessthan70["quality"], alternative="greater")
print(f"u_stat_70:{u_stat70}")
print(f"p_value_70:{p_value70}")

# NOW USE MEDIAN - Result: No Longer Statistically Significant
filtered_exp_quality_greaterthanMedian = exp_quality[exp_quality["Experience"] >= np.nanmedian(experience)] # First Group - "More Experienced"
filtered_exp_quality_lessthanMedian = exp_quality[exp_quality["Experience"] < np.nanmedian(experience)] # Second Group - "Less Experienced"
u_statMedian, p_valueMedian = mannwhitneyu(filtered_exp_quality_greaterthanMedian["quality"], filtered_exp_quality_lessthanMedian["quality"], alternative="greater")
print(f"u_stat_median:{u_statMedian}")
print(f"p_value_median:{p_valueMedian}")


numRepeats = int(1e4)
bootstrap_mean_differences = []
np.random.seed(17864220)

for ii in range(numRepeats):
    greater70 = np.random.choice(filtered_exp_quality_greaterthan70["quality"], size=len(male_ratings), replace=True)
    lessthan70 = np.random.choice(filtered_exp_quality_lessthan70["quality"], size=len(female_ratings), replace=True)
    median_difference = np.nanmedian(greater70) - np.nanmedian(lessthan70)
    bootstrap_mean_differences.append(median_difference)

alpha = 0.005
lower_bound = np.percentile(bootstrap_mean_differences, 100 * (alpha / 2))
upper_bound = np.percentile(bootstrap_mean_differences, 100 * (1 - alpha / 2))

print(f"99.5% Confidence Interval for Difference in Medians: ({lower_bound:.5f}, {upper_bound:.5f}) And since the lower bound does not touch 0, it is statistically significant.")
print(f"Width of Confidence Interval= {upper_bound - lower_bound:.5f}")
plt.figure(figsize=(6, 4))
plt.hist(bootstrap_mean_differences, bins=50, alpha=0.8, label="Distribution of Median Differences")
plt.axvline(lower_bound, color="red", label=f"99.5% CI Lower: {lower_bound:.5f}")
plt.axvline(upper_bound, color="red", label=f"99.5% CI Upper: {upper_bound:.5f}")
plt.title("Distribution of Difference in more and less experienced Professors' Ratings - Bootstrap)")
plt.xlabel("Difference of Median Rating")
plt.ylabel("Frequency")
plt.legend()
plt.grid(linestyle="-", alpha=0.5)
plt.show()

#%%

# 3. 
# Here, we would use spearman correlation ninstead of the pearson correation because the data is ordinal and so not onn a ratio scale.
# There is a moderate negative monotonic relationship between the average rating and average difficulty.
rating_and_difficulty = filtered_dataframe[["Average Rating", "Average Difficulty"]].dropna()
correlation, p_value = spearmanr(rating_and_difficulty["Average Difficulty"], rating_and_difficulty["Average Rating"])
print(f"Q3. spearman correlation: {correlation}")
print(f"p_value: {p_value}")
print("The p-value is less than 0.005 so the test-statistic is statistically significant and we can drop the null hypothesis and conclude that there is a negative yet moderate monotonic relationship between average rating and average difficulty.")

plt.scatter(rating_and_difficulty["Average Difficulty"], rating_and_difficulty["Average Rating"], label="Avg Difficulty Against Avg Rating", alpha=0.3, s=10)
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.title("Avg Difficulty Against Avg Rating")
plt.legend()
plt.show()

# Mean rating at each difficulty rating 
mean_ratings = rating_and_difficulty.groupby("Average Difficulty")["Average Rating"].mean().reset_index()
plt.figure(figsize=(12, 9))
plt.plot(mean_ratings["Average Difficulty"], mean_ratings["Average Rating"], marker="o", linestyle="-")
plt.title("Mean Average Rating at Each Difficulty Level (Using Difficulty Score)")
plt.xlabel("Average Difficulty")
plt.ylabel("Average of the Average Ratings Grouped By Each Difficulty Level")
plt.grid()
plt.show()


#%%

# 4. 
# Here, we will define a lot of classes in the online modality as when the number of ratings from online classes is greater than 60% of the total number of ratings.
# We use the u test because the groups are independent and the data is ordinal and not normally diostributed. 
# Here, we will use two one_tailed test to see if professors who teach a large proportion of classes in the online modality.
# The null hypothesis is that teaching a lot of classes in the online modality has no effect on the ratings that Professor receives.
# The alternative hypothesis is that teaching a lot of classes in the online modality leads to lower ratings for that Professor.

online_df = filtered_dataframe[["Average Rating", "The number of ratings coming from online classes","Number of Ratings"]].dropna()
proportion_of_ratings = online_df["The number of ratings coming from online classes"]/online_df["Number of Ratings"] # Created a new variable.
online_df["Proportion of Ratings from Online Classes"] = proportion_of_ratings
high_online = online_df[online_df["Proportion of Ratings from Online Classes"]>=0.5]["Average Rating"]
low_online = online_df[online_df["Proportion of Ratings from Online Classes"]<0.5]["Average Rating"]

u_stat_less_onetail, p_value = mannwhitneyu(high_online,low_online, alternative="less")
print(f"Q4. One-tailed (Left tailed) u-statistic: {u_stat_less_onetail}")
print(f"p_value:{p_value}")


# Plot of medians of the two groups on separate graphs
plt.figure(figsize=(10,6))
plt.subplot(1, 2, 1)
plt.hist(high_online, bins=100, color="blue", alpha=0.8)
plt.axvline(np.nanmedian(high_online), color="red", linestyle="--", label="Median of High Proportion of Online Ratings")
plt.title("High Online Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.legend()
plt.subplot(1, 2, 2)
plt.hist(low_online, bins=100, color="orange", alpha=0.8)
plt.axvline(np.nanmedian(low_online), color="red", linestyle="--", label="Median of Low Proportion of Online Ratings")
plt.title("Low Online Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.legend()
plt.show()


#%%

# 5. What is the relationship between the average rating and the proportion of people who would take
# the class the professor teaches again?
from scipy.stats import spearmanr
# Here, we use spearman correlation instead of pearson correlation because the data is not on a ratio/interval scale hence the relationship is not linear.
df5 = filtered_dataframe[["Average Rating", "The proportion of students that said they would take the class again"]].dropna()
correlation5, p_value5 = spearmanr(df5["Average Rating"], df5["The proportion of students that said they would take the class again"])
print(f"Q5. Spearman Correlation: {correlation5}")
print(f"p-value: {p_value5}")

# Graph
plt.scatter(df5["The proportion of students that said they would take the class again"], df5["Average Rating"], label="Proportion of Students Who Would Take Class Again Against Average Rating", alpha=0.3, s=10)
plt.xlabel("Proportion of Students Who Would Take Class Again")
plt.ylabel("Average Rating")
plt.title("Proportion of Students Who Would Take Class Again Against Average Rating")
plt.legend()
plt.show()



#%%
# 6. Do professors who are “hot” receive higher ratings than those who are not? Again, a significance
# test is indicated.
df6 = filtered_dataframe[["Average Rating", "Received a pepper?"]].dropna()
hot_ratings = df6[df6["Received a pepper?"] == 1]["Average Rating"]
not_hot_ratings = df6[df6["Received a pepper?"] == 0]["Average Rating"]
u_stat6, p_value6 = mannwhitneyu(hot_ratings, not_hot_ratings, alternative="greater")
print(f"Q6. u-test statistic: {u_stat6}")
print(f"p-value: {p_value6}")

# Graph of the two distributions
plt.figure(figsize=(10,6))
sns.histplot(hot_ratings, bins=90, label="Received Pepper Ratings")
sns.histplot(not_hot_ratings, bins=90, label="No Pepper Ratings")
plt.title("Distribution of Hot and Not Hot Ratings")
plt.legend()
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.show()



#%%
# 7. Build a regression model predicting average rating from difficulty (only). Make sure to include the R2
# and RMSE of this model.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
seed = 17864220

df7 = filtered_dataframe[["Average Rating", "Average Difficulty"]].dropna()
X7 = df7[["Average Difficulty"]]
Y7 = df7["Average Rating"]
X_train, X_test, Y_train, Y_test = train_test_split(X7, Y7, train_size=0.8, random_state=seed)
model = LinearRegression().fit(X_train, Y_train)
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
r2 = r2_score(Y_test, predictions)
coefficients = model.coef_
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")
print(f"coefficient of Average Difficulty: {coefficients}")

plt.scatter(X_test, Y_test, label="Ratings from the test set", alpha=0.2, s=10)
plt.plot(X_test, predictions, color="red", label="Best Fit Line of Linear Regression Model")
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.title("Average Rating against Average Difficulty")
plt.legend()
plt.show()


#%%

# 8. Build a regression model predicting average rating from all available factors. Make sure to include
# the R2 and RMSE of this model. Comment on how this model compares to the “difficulty only” model
# and on individual betas. Hint: Make sure to address collinearity concerns.
# To determine the train test split, 
# Drop rows with NaN values in X or Y
# To address collinearity concerns, we could use ridge or lasso regression to address the problem of collinearity.
# Ridge regression works 
# We will use both.
# feature selection
# have to z score 
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
seed = 17864220
df8 = filtered_no_ambiguous_genders.dropna()
X_all_8 = df8.drop(columns=["Average Rating"])
Y = df8["Average Rating"]

scaler = StandardScaler()
X_train, X_test, Y_train, Y_test = train_test_split(X_all_8, Y, train_size=0.8, random_state=seed) # Note that I set the random satte to 17864220 and this makes the coefficient of number of ratings negative - I ran it without specifying it and sometimes it yields a positive coefficient.
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train) 
X_test_scaled = scaler.transform(X_test)

plt.figure(figsize=(6,4))
sns.heatmap(pd.DataFrame(X_train, columns=X_all_8.columns).corr(), annot=True, cmap="coolwarm", fmt=".2f") 
plt.title("Correlations of Scaled Selected Features (Training Set) - Linear Regression")
plt.show()

#------------------Lasso Regression with Cross-Validation------------------ I did not drop the dummy variable and decided to let the lasso regression do the feature selection.
model_all_factors_lasso_cv = LassoCV(cv=10, max_iter=100000).fit(X_train_scaled, Y_train)
predictions_all_lasso = model_all_factors_lasso_cv.predict(X_test_scaled)
rmse_all_lasso = np.sqrt(mean_squared_error(Y_test, predictions_all_lasso))
r2_all_lasso = r2_score(Y_test, predictions_all_lasso)

print(f"\nBest alpha for Lasso: {model_all_factors_lasso_cv.alpha_:.4f}") 
print(f"RMSE with all factors - LassoCV: {rmse_all_lasso:.4f}")
print(f"R-squared with all factors - LassoCV: {r2_all_lasso:.4f}")
lasso_coefficients = pd.Series(model_all_factors_lasso_cv.coef_, index=X_all_8.columns)
print("LassoCV Coefficients:")
print(lasso_coefficients)
print(f"LassoCV Intercept: {model_all_factors_lasso_cv.intercept_:.4f}")

#------------------Ridge Regression with Cross-Validation------------------
model_all_factors_ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=10).fit(X_train_scaled, Y_train)
predictions_all_ridge = model_all_factors_ridge_cv.predict(X_test_scaled)
rmse_all_ridge = np.sqrt(mean_squared_error(Y_test, predictions_all_ridge))
r2_all_ridge = r2_score(Y_test, predictions_all_ridge)
print(f"\nBest alpha for Ridge: {model_all_factors_ridge_cv.alpha_:.4f}") # Print the best alpha
print(f"RMSE with all factors - RidgeCV: {rmse_all_ridge:.4f}")
print(f"R-squared with all factors - RidgeCV: {r2_all_ridge:.4f}")
ridge_coefficients = pd.Series(model_all_factors_ridge_cv.coef_, index=X_all_8.columns)
print("RidgeCV Coefficients (best alpha):")
print(ridge_coefficients)
print(f"RidgeCV Intercept: {model_all_factors_ridge_cv.intercept_:.4f}")


#------------------Linear Regression------------------ I dropped the Female Gender to avoid the dummy variable trap.
df8 = filtered_no_ambiguous_genders.dropna()
X_all_8 = df8.drop(columns=["Average Rating", "Female Gender"])
Y = df8["Average Rating"]

scaler = StandardScaler()
X_train, X_test, Y_train, Y_test = train_test_split(X_all_8, Y, train_size=0.8, random_state=seed) # Note that I set the random satte to 17864220 and this makes the coefficient of number of ratings negative - I ran it without specifying it and sometimes it yields a positive coefficient.
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train) 
X_test_scaled = scaler.transform(X_test)

model_all_factors_linear = LinearRegression().fit(X_train_scaled, Y_train)
predictions_all_linear = model_all_factors_linear.predict(X_test_scaled)
rmse_all_linear = np.sqrt(mean_squared_error(Y_test, predictions_all_linear))
r2_all2_linear = r2_score(Y_test, predictions_all_linear)

print(f"\nRMSE with all factors - Linear Regression: {rmse_all_linear:.4f}")
print(f"R-squared with all factors - Linear Regression: {r2_all2_linear:.4f}")

linear_coefficients = pd.Series(model_all_factors_linear.coef_, index=X_all_8.columns)
print("Linear Regression Coefficients:")
print(linear_coefficients)
print(f"Linear Regression Intercept: {model_all_factors_linear.intercept_:.4f}")


#%%
# 9. Build a classification model that predicts whether a professor receives a “pepper” from average
# rating only. Make sure to include quality metrics such as AU(RO)C and also address class imbalances.
# The class (0, 1) is split into 60% and 40% respectively. This means that the classes are roughly balanced. However, there is still some imbalance and therefore we should use AUC to ecvaluate instead of focusing on accuracy.
# We could also use precision recall curve to further address class imbalance because it focuses on the positive (1) class only.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve

# In this question we are not using the gender to perform classification. Therefore, we will still keep the data/rows where the gender is ambiguous, since they are still a Professor. 
df9 = filtered_dataframe[["Average Rating", "Received a pepper?"]].dropna() 
X_9 = df9[["Average Rating"]]
Y_9 = df9["Received a pepper?"]
X_train, X_test, Y_train, Y_test = train_test_split(X_9, Y_9, train_size=0.8, random_state=17864220) 
modelLogistic = LogisticRegression(random_state=17864220, class_weight="balanced", max_iter=10000).fit(X_train, Y_train)
predicted_classifications = modelLogistic.predict(X_test)
predicted_probabilities = modelLogistic.predict_proba(X_test)[:, 1]

withOrwithout_pepper = df9["Received a pepper?"].value_counts()
print(withOrwithout_pepper)
aucroc_score = roc_auc_score(Y_test, predicted_probabilities)
print(f"auc score: {aucroc_score}")

print("Classification Report:\n", classification_report(Y_test, predicted_classifications))
print("To address class imbalance, we can use the precision-recall curve.")

# ROC Curve to address class imbalance - Not using accuracy
falsePositiveR, truePositiveR, thresholds = roc_curve(Y_test, predicted_probabilities)
plt.figure(figsize=(8, 4))
plt.plot(falsePositiveR, truePositiveR, label="Single Predictor ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="True Positve Rate = False Positive Rate")
plt.xlabel("False Positive Rate - 1-Specificity")
plt.ylabel("True Positive Rate - Recall/Sensitivity")
plt.title(f"ROC Curve AUROC = {aucroc_score:.4f}")
plt.legend()
plt.show()

#%%
# 10.
# To address class imbalances. PCA to scale data.
# Collinearity concern.
# Preprocessing: PCA Analysis: "extract underlying factors that account for the observed data." New syhtetic feature - linear combination of some  variables
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Unlike in question 9, here we don't include the rows where there is ambiguous gender because we will also use gender to predict whether the Professor receiveds a pepper.
df10 = filtered_no_ambiguous_genders.dropna()
X_all_10 = df10.drop(columns=["Received a pepper?", "Female Gender"])
Y_10 = df10["Received a pepper?"]

withOrwithout_pepper2 = df10["Received a pepper?"].value_counts()
print(withOrwithout_pepper2) # To check if the class is imbalanced. The class is roughly balanced.

# ------- Without PCA ------
X_train, X_test, Y_train, Y_test = train_test_split(X_all_10, Y_10, train_size=0.8, random_state=17864220)
modelLogistic = LogisticRegression(random_state=17864220, class_weight="balanced", max_iter=100000).fit(X_train, Y_train)
predicted_classifications = modelLogistic.predict(X_test)
predicted_probabilities = modelLogistic.predict_proba(X_test)[:, 1]

aucroc_score = roc_auc_score(Y_test, predicted_probabilities)
print(f"auc score: {aucroc_score}")

print("Classification Report without PCA:\n", classification_report(Y_test, predicted_classifications))
print("To address class imbalance, we can use the precision-recall curve.")

# --- ROC Curve to address class imbalance ---
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# ROC Curve to address class imbalance
falsePositiveR, truePositiveR, thresholds = roc_curve(Y_test, predicted_probabilities)
plt.figure(figsize=(8, 4))
plt.plot(falsePositiveR, truePositiveR, label=f"ROC Curve (No PCA) ROC Score = {aucroc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="True Positive Rate = False Positive Rate")
plt.xlabel("False Positive Rate - 1-Specificity")
plt.ylabel("True Positive Rate - Recall/Sensitivity")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ------ With PCA -------
X_train, X_test, Y_train, Y_test = train_test_split(X_all_10,Y_10,train_size=0.8,random_state=17864220)
scaler = StandardScaler() # This ensures that the data is mean callibrated so that the mean is 0.
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

sns.heatmap(pd.DataFrame(X_train_scaled, columns=X_all_10.columns).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlations of Scaled Selected Features (Training Set)")
plt.show()

# PCA
pca = PCA().fit(X_train_scaled)
rotated_data = pca.transform(X_train_scaled) # 6 factors instead of the original 6 variables
eigenValues = pca.explained_variance_
loadings = pca.components_
varExplained = eigenValues/sum(eigenValues)*100
for ii in range(len(varExplained)):
    print(f"Variance Explained: {varExplained[ii].round(3)}") # Prints out the percentage of the variance explained by each factor to decide which components to use.

# Scree plot - To decide which compnents to use - we will use the Kaiser Criterion. We will use components 1,2,3
kaiser_threshold = 1
num_components = len(eigenValues)
x = np.linspace(1, num_components, num_components)
plt.figure(figsize=(10, 6))
plt.bar(x, eigenValues, color="blue", alpha=0.8, label="Eigenvalues")
plt.plot([0, num_components + 1], [1, 1], color="orange", linestyle="--", label="Kaiser Criterion (Eigenvalue=1)") 
plt.xlabel("Principal Components")
plt.ylabel("Eigenvalues")
plt.title("Scree Plot")
plt.xticks(x)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

pca_final = PCA(n_components=6) # Only keep the first three components - Kairser Criteria
X_trained_scaled_pcaFinal = pca_final.fit_transform(X_train_scaled)
X_test_scaled_pcaFinal = pca_final.transform(X_test_scaled)
print(f"\nShape of data after final PCA transformation (Train): {X_trained_scaled_pcaFinal.shape}")
print(f"Shape of data after final PCA transformation (Test): {X_test_scaled_pcaFinal.shape}")

logistic_model_pca = LogisticRegression(random_state=17864220, class_weight="balanced", max_iter=10000)
logistic_model_pca.fit(X_trained_scaled_pcaFinal, Y_train)

# Predict probabilities and classes on the test set
Y_predicted_probability_pca = logistic_model_pca.predict_proba(X_test_scaled_pcaFinal)[:, 1] # Probabilities for 1
Y_pred_class_pca = logistic_model_pca.predict(X_test_scaled_pcaFinal)

# AUC
print("\nAUC ROC for PCA")
auroc_pca = roc_auc_score(Y_test, Y_predicted_probability_pca) # Probabilities for AUROC
print(f"AUROC Score (PCA): {auroc_pca}")

# Calssification Report
print("Classification Report (PCA):\n", classification_report(Y_test, Y_pred_class_pca, target_names=["No Pepper", "Pepper"]))

# Confusion Matrix
confusion_matrix_pca = confusion_matrix(Y_test, Y_pred_class_pca)
print("Confusion Matrix (PCA):\n", confusion_matrix_pca)

# Visualization of Confusion Matrix
sns.heatmap(confusion_matrix_pca, annot=True, cmap="Blues", xticklabels=["No Pepper", "Pepper"], yticklabels=["No Pepper", "Pepper"])
plt.ylabel("Actual")
plt.xlabel("Predictions")
plt.title("Confusion Matrix (PCA)")
plt.show()

# Loading Matrix
loadingValues1 = loadings[0, :]
plt.figure(figsize=(10, 4))
plt.bar(X_all_10.columns, loadingValues1)
plt.ylabel("Loadings")
plt.title("Loadings for PC1")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

loadingValues2 = loadings[1, :]*-1
plt.figure(figsize=(10, 4))
plt.bar(X_all_10.columns, loadingValues2)
plt.ylabel("Loadings")
plt.title("Loadings for PC2")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

loadingValues3 = loadings[2, :]
plt.figure(figsize=(10, 4))
plt.bar(X_all_10.columns, loadingValues3)
plt.ylabel("Loadings")
plt.title("Loadings for PC3")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

loadingValues4 = loadings[3, :]
plt.figure(figsize=(10, 4))
plt.bar(X_all_10.columns, loadingValues3)
plt.ylabel("Loadings")
plt.title("Loadings for PC4")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# ROC Curve
falsePositiveRate, truePositiveRate, thresholds = roc_curve(Y_test, Y_predicted_probability_pca)
plt.figure(figsize=(8, 4))
plt.plot(falsePositiveRate, truePositiveRate, label=f"ROC Curve (PCA 6 Components) ROC Score = {auroc_pca:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="True Positive Rate =False Positive Rate")
plt.xlabel("False Positive Rate - 1-Specificity")
plt.ylabel("True Positive Rate - Recall/Sensitivity")
plt.title("ROC Curve (PCA 6 Components)")
plt.legend()
plt.show()



#%%
# Extra Credit: Find the states with the highest average ratings for Economics Professors and Math Professors
filtered_combined_df = rmpCapstonedata_combined[rmpCapstonedata_combined["Number of Ratings"] >= 30].copy()
econ_df = filtered_combined_df[filtered_combined_df["Major/Field"].str.lower().str.contains("economics", na=False)]
math_df = filtered_combined_df[filtered_combined_df["Major/Field"].str.lower().str.contains("math", na=False)]
dataScience_df = filtered_combined_df[filtered_combined_df["Major/Field"].str.lower().str.contains("data science", na=False)]
econ_ratingsByState = econ_df.groupby("US State (2 letter abbreviation)")["Average Rating"].mean().sort_values(ascending=False)
math_ratingsByState = math_df.groupby("US State (2 letter abbreviation)")["Average Rating"].mean().sort_values(ascending=False)
print("States with the highest average ratings of Math professors:")
print(math_ratingsByState.head(5))
print("States with the highest average ratings of Economics professors:")
print(econ_ratingsByState.head(5))
