import json
import pandas as pd
from ludwig.utils import data_utils

all_users = {}
all_problems = {}
all_concepts = {}

df = data_utils.read_csv("data.csv")

# removing_duplicate_users = [i for n, i in enumerate(df['Anon Student Id'].tolist()) if i not in df['Anon Student Id'].tolist()[:n]]
removing_duplicate_users = df['Anon Student Id'].unique().tolist()
count_user = 1
for i in removing_duplicate_users:
    all_users[i] = count_user
    count_user += 1
json_users = json.dumps(all_users, indent = 2)
#---------------------------------------------------------------------------------------

# removing_duplicate_problems = [i for n, i in enumerate(df['Problem Name'].tolist()) if i not in df['Problem Name'].tolist()[:n]]
removing_duplicate_problems = df['Problem Name'].unique().tolist()
count_problem = 1
for i in removing_duplicate_problems:
    all_problems[i] = count_problem
    count_problem += 1
json_problems = json.dumps(all_problems, indent = 2)
#---------------------------------------------------------------------------------------

cleanedList1 = [x for x in df['KC (WPI-Apr-2005)'].tolist() if str(x) != 'nan']
cleanedList2 = [x for x in df['KC (WPI-Apr-2005)_1'].tolist() if str(x) != 'nan']
cleanedList3 = [x for x in df['KC (skills_from_dataframe)_11'].tolist() if str(x) != 'nan']
cleanedList4 = [x for x in df['KC (skills_from_dataframe)_13'].tolist() if str(x) != 'nan']
cleanedList5 = [x for x in df['KC (skills_from_dataframe)_17'].tolist() if str(x) != 'nan']
cleanedList6 = [x for x in df['KC (skills_from_dataframe)_15'].tolist() if str(x) != 'nan']
cleanedList7 = [x for x in df['KC (MCAS39-State_WPI-Simple)'].tolist() if str(x) != 'nan']
cleanedList8 = [x for x in df['KC (MCAS39-State_WPI-Simple)_21'].tolist() if str(x) != 'nan']
cleanedList9 = [x for x in df['KC (MCAS39-State_WPI-Simple)_23'].tolist() if str(x) != 'nan']
cleanedList10 = [x for x in df['KC (MCAS39-State_WPI-Simple)_25'].tolist() if str(x) != 'nan']
cleanedList11 = [x for x in df['KC (MCAS5-State_WPI-Simple)'].tolist() if str(x) != 'nan']

combining_lists = cleanedList1 + cleanedList2 + cleanedList3 + cleanedList4 + cleanedList5 + cleanedList6 + cleanedList7 + cleanedList8 + cleanedList9 + cleanedList10 + cleanedList11
# removing_duplicate_concepts = [i for n, i in enumerate(combining_lists) if i not in combining_lists[:n]]
removing_duplicate_concepts = list(set(combining_lists))

count_concept = 1
for i in removing_duplicate_concepts:
    all_concepts[i] = count_concept
    count_concept += 1
json_concepts = json.dumps(all_concepts, indent = 2)
dataset_statistics = {

        'Dataset': ['Assistments Math 2005-2006'],
        'Learners': [str(len(all_users))],
        'Concepts': [str(len(all_concepts))],
        'Questions': [str(len(all_problems))],
        'Responses': [str(len(df['Outcome']))]

        }
statistics_df = pd.DataFrame(dataset_statistics, columns = ['Dataset', 'Learners','Concepts','Questions','Responses'])
print(statistics_df)