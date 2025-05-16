"""
ⓒ Debug.Ai 2025 Eshan Jayasundara

This is the main file which combines all other files and do the expected job.
"""

from api_call import *
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import login
from datasets import Dataset
from decouple import config

stackapps_token = config("STACKAPPS_TOKEN")
stackapps_key = config("STACKAPPS_KEY")
login(token=config("TOKEN"))

date_ranges = [
    (config("FROM_DATE"), config("TO_DATE"))
]

def fetch_data(from_date, to_date):
    data = get_question_answer_pairs(from_date=from_date, to_date=to_date, tags=['python'], stackapps_token=stackapps_token, stackapps_key=stackapps_key)
    return pd.DataFrame(data)

with ThreadPoolExecutor() as executor:
    dataframes = list(executor.map(lambda dates: fetch_data(*dates), date_ranges))

df = pd.concat(dataframes, ignore_index=True)

update_huggingface_dataset(
    new_data_df=df, 
    username="eshangj", 
    dataset_name="stackoverflow_q_and_a_sample", 
    split="train"
    )
