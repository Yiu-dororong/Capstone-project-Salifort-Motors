# Capstone project: Salifort Motors

This is the capstone project of my Google advanced data analytics course. I want to leave my work here to present my understanding and idea. 

As usual, I will leave the full code notebook [here](https://colab.research.google.com/drive/1ZFJWdcpXAL0blaiomfiWGrDV7UJ9uC8d?usp=sharing). The dataset can be found in [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction/data), or in my repositories. There are trained model files(.pickle) inside as well.

# Situation

The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, and they referred to me as a data analytics professional and ask me to provide data-driven suggestions based on my understanding of the data. They have the following question: *what’s likely to make the employee leave the company?*

The goals in this project are to analyze the data collected by the HR department and to *build a model that predicts whether or not an employee will leave the company.*

If the model can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

# Prepare

To begin with, let's take a glimpse on the data dictionary,

Variable  |Description |
-----|-----|
satisfaction_level|Employee-reported job satisfaction level [0&ndash;1]|
last_evaluation|Score of employee's last performance review [0&ndash;1]|
number_project|Number of projects employee contributes to|
average_monthly_hours|Average number of hours employee worked per month|
time_spend_company|How long the employee has been with the company (years)
Work_accident|Whether or not the employee experienced an accident while at work
left|Whether or not the employee left the company
promotion_last_5years|Whether or not the employee was promoted in the last 5 years
Department|The employee's department
salary|The employee's salary (U.S. dollars)

The `left` column will play a key role on this project, as it would be the outcome variable.

Now, I shall load the dataset (named as `df0`) and take a look on it by using `.info()` and `.describe()`.

Luckily, there are no missing values, but I found out that there 3008 out of 14999 are duplicates from `df0.duplicated().sum()`. I dropped them as save it to a new dataframe called `df`, the size became 11991 rows x 10 columns.

I checked if there are outliers, for `time_spend_company`, 5628 of them fall into the definition of outliers. I did not check for other features, because this can already tell us that this is a highly skewed dataset.

# Analyze

**EDA**

I created a lot of graphs to visualise the distribution of each feature.

<img width="363" height="273" alt="image" src="https://github.com/user-attachments/assets/32cefbfa-666b-49c9-8442-a033eea67ee5" />

We can tell that the turnover rate is about one-sixth, 16.6%.

<img width="463" height="316" alt="image" src="https://github.com/user-attachments/assets/1b9cbe75-ddb9-4bd1-80bd-c1bf1d24b5f9" />

The satisfaction level in general is pretty high, but there is a high amount of zero scores.

I also created a new feature, which gives me some interesting findings.

```
# Create a new feature
df['change_in_satisfaction'] = df['satisfaction_level'] - df['last_evaluation']
```

<img width="463" height="316" alt="image" src="https://github.com/user-attachments/assets/d4530c56-2ad3-49f6-8a36-f29c3fac0c5f" />

By observing the graph, we can see that the employees who left can be divided into two clusters. One is located in the centre, having similar score to the last year, while the another the one is located on the left hand side (-0.7,-0.9) region, representing a drastic decrese compared to last year. Considering that the maximum score is 1.0, haveing a decreasing of 0.8 is like drop from (0.8,1.0) to (0.0,0.2).

Since the distribution projected on the graph above is bimodel, there is a clear spilt on -0.5, this portion has a turnover rate of 50%, much higher then in general, 16.6%.

```
# Filter out the employees rated 0.5 than less year
df[df['change_in_satisfaction']<-0.5]['left'].value_counts(normalize=True)

Output:
        proportion
left	
1	0.509766
0	0.490234
```

Move on to other features.

<img width="704" height="371" alt="image" src="https://github.com/user-attachments/assets/3ef71ed6-7bcf-498f-9de5-5f89a8adc89f" />

Although there are cases that employees having `number_project` being 2 and still leave, but one can say the more project they got, the more likely they are going to leave, especially seen in a 100% left for having 7 projects.

Correlation heatmap is also used to gain insights on what correlates with `left` most.

<img width="930" height="686" alt="image" src="https://github.com/user-attachments/assets/4d625c3d-a378-436a-8e4c-8fcc240b9c52" />

We can see that `left` is mostly correlated with `satisfaction_level`, `change_in_satisfaction` and `project_per_year`.

```
# I am not sure if this feature means total projects conducted throughout career or just this year so I created this
df['project_per_year'] = df['number_project']/df['time_spend_company']
```

For `average_montly_hours`,

<img width="1001" height="679" alt="image" src="https://github.com/user-attachments/assets/037b1b90-d282-4a8d-9c29-466afda3d1a9" />

Notice that employees who left can be roughly divided into 3 groups. The first one located in the topright corner, having high working hours but still reported high satisfaction level. The second one is in the middle, having a slightly low satisfaction level but relatively low working hours. Perhaps the jobs does not match their interest. The last one is in the left bottom corner, extremely high working hours and satisfaction level near 0.

<img width="493" height="352" alt="image" src="https://github.com/user-attachments/assets/bbc14880-4009-461c-8edc-5d48f94c2620" />

15% of the employee had suffered from work accident, but data shows that those who have never had a work accident, turns our to have a higher proportion to leave the company. Perhaps there compensation for it make employee feel the company cares them.

<img width="548" height="383" alt="image" src="https://github.com/user-attachments/assets/47942c57-3953-4485-abfe-447407c9b7dc" />

A extremely low percent (1.69%) of people who did get a promotion, but a promotion did have its value, lowering the turnover rate.

<img width="471" height="467" alt="image" src="https://github.com/user-attachments/assets/7da5c7bc-a4ae-4a3b-99b9-3e3b4da8fd74" />

No strong correlation between department and left from the graph.

<img width="471" height="393" alt="image" src="https://github.com/user-attachments/assets/86b60059-dcf6-46e1-8d16-7b0f4b87c08b" />

Salary being low and medium take up the most part of the whole population.

## Insights 

Based on the `change_in_satisfaction histogram` and `average_monthly_hours` scatter plot, we can see that there are at least two groups of people chose to leave the company. Employees can leave with different satisfaction.

Too much project, too long working hours seem to be high risk for employee to leave.

Salary, promotion and accident seems to have negative correlation with left.

department does not give a high clear correlation.

# Construct

As mentioned, this dataset has a large portion of outliers, so this will violate the assumption of logistic regression. I do not want to drop such large amount of data neither, so I chose to use machine learn models instead.

These are the hyperparameters I tried for the models using `GridSearchCV`, I then use `.fit(X_train, y_train)` to train the models.

```
# 1. Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [10,15,20,25],
        'max_features': [1.0],
        'max_samples': [0.5,0.75],
        'min_samples_leaf': [5],
        'min_samples_split': [2],
        'n_estimators': [300,400,500],
        }

# 3. Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='recall')
```

```
# 1. Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5,10],
        'min_child_weight': [2],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [300,400,500]
        }

# 3. Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=4, refit='recall')
```

As a result, they give a exceptionally high score. I am worried of having overfitting problem. Let's move on to the validation part to see how well they perform.

<img width="407" height="105" alt="image" src="https://github.com/user-attachments/assets/2b0c0909-6fe8-42cd-a5b9-a95ce54d8476" />

They also perform very well on the validation set, and XGB model won by a tiny amount.

<img width="407" height="206" alt="image" src="https://github.com/user-attachments/assets/a0ac200b-71b7-46a2-96da-08062c5553ae" />

We can see that even for unseen test dataset, XGB still perform extremely well, so it does not seem to have overfitting problem.

Here are some more results for this champion model:

<img width="565" height="432" alt="image" src="https://github.com/user-attachments/assets/1f13216d-3b46-4636-9b28-272efa3a1c44" />

<img width="731" height="455" alt="image" src="https://github.com/user-attachments/assets/06847901-b655-459b-9cde-8147c031a528" />

To summarize, our model has an over 90% of recall, which means that when the model predicts the employees is leaving, there is more than 90% to be true.

From the model, it told us that the most important features are:

`working hours`, `satisfaction`, `number of projects`, `time_spend_company`.

This matches what we expected in EDA. These can be direction for us to prevent employees to leave.

# Exceute

**Key insights:**

We can clearly see there are two main groups of employees who chose to leave. The first one being short tenure and having moderate working conditions, as reflected by few projects, acceptable satisfaction level and low working hours. While the another being the complete opposite, high workload, low satisfaction, long working hours. (The third group is someone has high satisfaction and long working hour, yet hard to explain as if the first group.)

For the first one, it is likely to be there is a mismatch between employee's favour and our company's environment, similar to the cause of frictional unemployment. And for the second one, it is clearly that the employees are not satisfied with the working conditions.

**Conclusions:**

It is hard to retain the first group of people, as they claimed to have a 0.8 out of 1.0 satisfaction level. The reason behind maybe they just want to change their job.

The main focus should be the second group. It includes some employees who have stayed in the company for around 5 years, which are valuable human resources. Based on the statistics, higher salary and having promotion are likely to make people stay.

**Recommendations:**

Allow projects to be shared accross wider manpower. As observed in the `number_project` graph, people who took up 7 projects all left. This also linked to the next problem, long working hours to complete. Longer working hours up to 300 hours a month is a serious problem to work-life balance, very probably results in lower satisfaction level. By sharing projects, it should reduce all these symptoms and make people more willing to stay.

Provide more compensation for being hard-working. Although not proven, my theory to the fraction of people who had encountered work accident turns out to be less likely to leave, is due to there are compensation provided to them, make them feel valued by the company. There should go the same to ovettime. Consider that salary and promotion also affect turnover rate, a better review system is needed to reward people being diligent.

**Further studies:**

The reasons behind first group chose to leave are still unclear, we shall keep track on this and find ways to collect more relevant data if possible. I also need information on what `number_project` really means.

Although the model gives a more than 90% accuracy, it is still worth noting that the features we used are some very objective metrics. We may want to find out what factors determine satisfaction level, can we include some intrisic feature, such as the working environment, team morale, colleagues' friendliness.


