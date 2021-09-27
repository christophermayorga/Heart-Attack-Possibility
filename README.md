# Heart_Attack_Possibility


Hello,

Welcome to the README file for my "Heart Attack Possibility Project".


___________________________
## Abstract
This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no/less chance of heart attack and 1 = more chance of heart attack


____________________________
## Goal
Create a classification model to predict whether or not a patient has a greater chance of having a heart attack.



____________________________

Data Dictionary:


|   Feature      |  Data Type   | Description    |
| :------------- | :----------: | -----------: |
| age | int64 | age of patient|
| is_male | int64 | 0 = Female, 1 = Male |
| chest_pain | int64 | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic | 
| resting_blood_pressure | int64 | resting blood pressure |
| cholesterol | int64 | serum cholesteral in mg/dl |
| fasting_blood_sugar | int64 | fasting blood sugar |
| rest_elect_results | int64 | resting electrocardiographic results |
| max_heart_rate | int64 | maximum heart rate achieved |
| exang | int64 | exercise induced angina (1 = yes; 0 = no) |
| oldpeak | float | ST depression induced by exercise relative to rest |
| slope | int64 | slope of the peak exercise ST segment: 1 = upsloping, 2 = flat, 3 = downsloping |
| ca | int64 | number of major vessels (0-3) colored by flourosopy |
| thal | int64 | 3 = normal; 6 = fixed defect; 7 = reversable defect |
| target | int64 | diagnosis of heart disease (angiographic disease status), 0: no/less chance of heart attack and 1: greater chance of having heart attack |




Stats:

- $H_o$: There is no association between age and risk of heart attack.
- $H_a$: There is an association between age and risk of heart attack.



- $H_o$: There is no association between is_male and risk of heart attack.
- $H_a$: There is an association between is_male and risk of heart attack.


- $H_o$: There is no association between chest_pain and risk of heart attack.
- $H_a$: There is an association between chest_pain and risk of heart attack.