import pandas as pd

pd.options.mode.chained_assignment = None

df = pd.read_csv("LLM_dataset_2.csv",
                 on_bad_lines='skip',
                 sep='⍼')

df.drop(columns=['index'], inplace=True)
df.dropna(inplace=True)

print(df.head())
print(df.shape)

txt_set = ""
for user_id, user_txt in zip(df['user_id'], df['text']):
    if user_id == 123123:
        txt_set = txt_set + "\n" + user_txt

plane_txt = open('LLM_dataset_2_daniel_plain.txt', 'x')
a = plane_txt.write(txt_set)
plane_txt.close()
print(a)
