# Training Dataset Contract

The reproducible evaluation script expects a CSV with these 14 columns:

```text
price,mileage,displacement,seats,owner_count,year,brand,model,city,transmission,fuel_type,vehicle_type,color,accident_history
```

`price` and `mileage` use the model training units stored in the supplied artifact. The current public API accepts mileage in万公里 and converts it to公里 in the model adapter. The CSV used for retraining must document its price unit, source, collection date, deduplication rules, and train/validation/test split.

Run the evaluator as a module from `backend`:

```powershell
python -m scripts.evaluate_model path/to/used_cars.csv
```

The evaluator reports RMSE, MAE, R2, 10% error accuracy, and a mean-price baseline. Do not publish a model accuracy claim without comparing against that baseline and documenting the data split.
