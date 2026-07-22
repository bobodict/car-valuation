# Training Data Contract

The training pipeline normalizes the public `car details v4.csv` dataset into these 14 columns:

```text
price,mileage,displacement,seats,owner_count,year,brand,model,city,transmission,fuel_type,vehicle_type,color,accident_history
```

The source is the Indian used-car market and prices are INR. Mileage is km and displacement is liters. The current source URL is:

```text
https://raw.githubusercontent.com/chandanverma07/DataSets/master/car%20details%20v4.csv
```

The source mapping is deterministic: `Price` to `price`, `Kilometer` to `mileage`, `Engine` text to liters, `Seating Capacity` to `seats`, ordinal `Owner` labels to `owner_count`, and the source brand/model/location/transmission/fuel/color fields to their normalized names. `vehicle_type` is `car` because the source has no reliable body-type field. `accident_history` is `unknown` because the source does not provide it.

Raw files are downloaded into `backend/data/raw/` and normalized files into `backend/data/processed/`; both are ignored by Git. The downloader records the URL, UTC retrieval time, byte count, SHA-256, and path in a manifest. Run the reproducible flow from `backend`:

```powershell
D:/car-valuation/venv/Scripts/python.exe -m scripts.download_public_dataset
D:/car-valuation/venv/Scripts/python.exe -m scripts.train_model --download
D:/car-valuation/venv/Scripts/python.exe -m scripts.evaluate_model data/processed/normalized_training.csv
```

Do not publish an accuracy claim without the recorded source, split, currency, target unit, and mean-price baseline.
