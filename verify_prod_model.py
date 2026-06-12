import mlflow
import mlflow.sklearn
import pandas as pd


experiment_name = "NIDS_XGBoost"
experiment = mlflow.get_experiment_by_name(experiment_name)

# Find the run by its tag
# MLflow search syntax allows us to filter directly on tags
production_runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.stage = 'production'"
)

# Check if we actually found a run
if not production_runs.empty:
    
    # Grab the run_id of the most recent production run (index 0)
    production_run_id = production_runs.iloc[0]["run_id"]
    print(f"Found production run! Run ID: {production_run_id}")

    # Isolate the metric columns directly from the DataFrame
    metric_cols = [c for c in production_runs.columns if c.startswith("metrics.")]
        
    print("\n--- Per-Class Metrics ---")
    # Convert to dictionary for cleaner printing and easier API returning
    metrics_dict = production_runs.iloc[0][metric_cols].to_dict()
    for metric, value in metrics_dict.items():
        # Strip the 'metrics.' prefix just for cleaner console output
        clean_name = metric.replace("metrics.", "")
        print(f"{clean_name:<45}: {value:.4f}")
    
    # Construct the Model URI
    # Format: runs:/<run_id>/<artifact_path_you_used_when_saving>
    model_uri = f"runs:/{production_run_id}/production_pipeline"
    
    # Load the actual model back into memory
    print(f"Loading model from: {model_uri}...")
    loaded_production_model = mlflow.sklearn.load_model(model_uri)

    X_cv = pd.read_csv('X_test.csv')
    sample = X_cv.iloc[[0]]
    print("Sanity prediction:", loaded_production_model.predict(sample))
    
    print("Model successfully loaded! Ready for predictions.")
    
else:
    print("Error: No runs found with the tag 'stage=production'.")