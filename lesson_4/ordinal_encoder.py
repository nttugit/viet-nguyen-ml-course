from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder

data = DataFrame(["XL","L","S","M","XS","XS","L","M"])
values = ["XS","S","M","L","XL"]

scaler = OrdinalEncoder(categories=[values])

encoded_data = scaler.fit_transform(data)
print("values", data.values)
for origin, encoded in zip(data.values,encoded_data):
    print(f"Before: {origin}, after: {encoded}")