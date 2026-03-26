import pandas as pd
df=pd.read_csv('data/rag_chunks.csv')
print(df[df['source_filename']=='02_sindh_criminal_prosecution_service_act_2009.txt'].iloc[8]['chunk_text'])
