#Python Version: 3.7.9
import pandas as pd
import numpy as np

RANDOM_SAMPLE_SIZE = 20000

def data_cleaning(df):
  new_df = df.copy()

  return new_df

def init_data(df):
  df.dropna(inplace=True)
  df.drop_duplicates(inplace=True, subset=['review_body'])
  df['star_rating'] = df['star_rating'].astype('int')
  return df


if __name__ == '__main__':
  df = pd.read_pickle("./data.pkl")

  df = init_data(df)
  cleaned_df = data_cleaning(df)


  class1_df = cleaned_df[cleaned_df['star_rating'] <= 2].sample(RANDOM_SAMPLE_SIZE)
  class2_df = cleaned_df[cleaned_df['star_rating'] == 3].sample(RANDOM_SAMPLE_SIZE)
  class3_df = cleaned_df[cleaned_df['star_rating'] >= 4].sample(RANDOM_SAMPLE_SIZE)

 # print(df.info(verbose=True, show_counts= True))
  #print(cleaned_df.info(verbose=True, show_counts= True))

  print(class1_df)
  print(class2_df)
  print(class3_df)




