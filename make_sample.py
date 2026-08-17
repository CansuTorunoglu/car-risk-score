import pandas as pd

cols = [
    'price', 'year', 'manufacturer', 'model',
    'condition', 'odometer', 'fuel', 'transmission',
    'drive', 'title_status', 'VIN',
    'description', 'image_url', 'state', 'posting_date'
]

df = pd.read_csv('data/cardata.csv/vehicles.csv', usecols=cols, nrows=10000)
df.to_csv('data/vehicles_filtered.csv', index=False, encoding='utf-8-sig')
print(f'Kaydedildi: {len(df)} satir, {len(df.columns)} sutun')
print('Sutunlar:', df.columns.tolist())
