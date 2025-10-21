# Capstone project: Salifort Motors

This is the capstone project of my Google advanced data analytics course. I want to leave my work here to present my understanding and idea. Unlike the previous one (Waze's study), I completed this one with very little guidance. Therefore, I am glad to share my thought here.

As usual, I will leave the full code notebook [here]().

# Situation

The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, and they referred to me as a data analytics professional and ask you to provide data-driven suggestions based on my understanding of the data. They have the following question: what’s likely to make the employee leave the company?

The goals in this project are to analyze the data collected by the HR department and to **build a model that predicts whether or not an employee will leave the company.**

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

Luckily, there are no missing values, but I found out that there 3008 out of 14999 are duplicates from `df0.duplicated().sum()`. I dropped them as save it to a new dataframe called `df`, the size became 11992 rows x 10 columns.

I checked if there are outliers, for `time_spend_company`, 5628 of them fall into the definition of outliers. I did not check for other features, because this can already tell us that this is a highly skewed dataset.

# Analyze

**EDA**

I created a lot of graph to visualise the distribution of each feature.

<img width="363" height="273" alt="image" src="https://github.com/user-attachments/assets/32cefbfa-666b-49c9-8442-a033eea67ee5" />

We can tell that the turnover rate is about one-sixth, 16.6%.

<img width="463" height="316" alt="image" src="https://github.com/user-attachments/assets/1b9cbe75-ddb9-4bd1-80bd-c1bf1d24b5f9" />

The satisfaction level in general is pretty high, but there is a high amount of 0 scores.

I also created a new feature, which gives me some interesting finding.

```
# Create a new feature
df['change_in_satisfaction'] = df['satisfaction_level'] - df['last_evaluation']
```

<img width="463" height="316" alt="image" src="https://github.com/user-attachments/assets/d4530c56-2ad3-49f6-8a36-f29c3fac0c5f" />

By observing the graph, we can see that the employees who left can be divided into two clusters. One is located in the centre, having similar score to the last year, while the another the one is located on the left hand side (-0.7,0.9) region, representing a drastic decrese compared to last year. Considering that the maximum score is 1.0, haveing a decreasing of 0.8 is like drop from (0.8,1.0) to (0.0,0.2).

Since the distribution projected on the graph above is bimodel, there is a clear spilt on -0.5, this portion has a turnover rate of ~0.5, much higher then in general (0.16).

```
# Filter out the employees rated 0.5 than less year
df[df['change_in_satisfaction']<-0.5]['left'].value_counts(normalize=True)
```

Move on to other features.

<img width="704" height="371" alt="image" src="https://github.com/user-attachments/assets/3ef71ed6-7bcf-498f-9de5-5f89a8adc89f" />

Although there are cases that employees having number_project being 2 and still leave, but one can say the more project they got, the more likely they are going to leave.

