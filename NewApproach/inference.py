# Updated inference.py with CloudWatch alert integration for 5 unique node IDs

"""
Run inference using final models on PCA features from featex.py.
Now sends alerts to AWS CloudWatch for predicted failures.

Usage:
    python inference.py \
      --records  outputs/stream1_records.csv \
      --features outputs/pca50.csv \
      --models   models/ \
      --threshold 0.2
"""

import argparse
import pandas as pd
import joblib
import pathlib
import boto3
import time

# CloudWatch setup
LOG_GROUP_NAME = 'BGLAlerts'
LOG_STREAM_NAME = 'FrequentPatternWarnings'

# Initialize CloudWatch Logs client
client = boto3.client('logs', region_name='us-east-1')  # Change to your region

# Create log group and log stream if they don't exist
try:
    client.create_log_group(logGroupName=LOG_GROUP_NAME)
except client.exceptions.ResourceAlreadyExistsException:
    pass

try:
    client.create_log_stream(logGroupName=LOG_GROUP_NAME, logStreamName=LOG_STREAM_NAME)
except client.exceptions.ResourceAlreadyExistsException:
    pass

def send_log_event(message):
    """Send a log event to AWS CloudWatch."""
    streams = client.describe_log_streams(logGroupName=LOG_GROUP_NAME, logStreamNamePrefix=LOG_STREAM_NAME)
    upload_sequence_token = streams['logStreams'][0].get('uploadSequenceToken')

    kwargs = {
        'logGroupName': LOG_GROUP_NAME,
        'logStreamName': LOG_STREAM_NAME,
        'logEvents': [{
            'timestamp': int(time.time() * 1000),
            'message': message
        }]
    }

    if upload_sequence_token:
        kwargs['sequenceToken'] = upload_sequence_token

    response = client.put_log_events(**kwargs)
    print("✅ Alert sent to CloudWatch:", response)

def main(records_csv, features_csv, models_dir, threshold):
    # 1) Load metadata and PCA features
    df_meta = pd.read_csv(records_csv, parse_dates=['window_start'])
    df_feat = pd.read_csv(features_csv)
    
    if len(df_meta) != len(df_feat):
        raise ValueError("Row count mismatch between records and features")
    
    X = df_feat.values
    
    # 2) Load trained pipelines
    models_path = pathlib.Path(models_dir)
    cls_pipe = joblib.load(models_path / 'cls_final.joblib')
    reg_pipe = joblib.load(models_path / 'reg_final.joblib')
    
    # 3) Predict
    p_fail = cls_pipe.predict_proba(X)[:, 1]
    eta_min = reg_pipe.predict(X)
    alert = (p_fail >= threshold).astype(int)
    
    # 4) Build output DataFrame
    out = pd.DataFrame({
        'window_start': df_meta['window_start'],
        'node_id'     : df_meta['node_id'],
        'p_fail'      : p_fail,
        'eta_min'     : eta_min,
        'alert'       : alert
    })
    
    # 5) Write CSV
    out.to_csv('predictions.csv', index=False)
    
    # 6) Print preview
    print("First rows of predictions.csv:")
    print(out.head())

    # 7) Send alerts to CloudWatch for 5 unique node IDs with p_fail > 0.65
    filtered = out[(out['alert'] == 1) & (out['p_fail'] > 0.65)]
    unique_nodes = filtered['node_id'].unique()[:5]  # Get up to 5 unique node IDs

    for node_id in unique_nodes:
        row = filtered[filtered['node_id'] == node_id].iloc[0]  # Get the first row for this node
        message = (
            f"ALERT: Predicted failure detected!\n"
            f"Node ID: {row['node_id']}\n"
            f"Window Start: {row['window_start']}\n"
            f"Failure Probability: {row['p_fail']:.2f}\n"
            f"Estimated Time to Failure (minutes): {row['eta_min']:.2f}"
        )
        send_log_event(message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records',  required=True,
                        help='stream1_records.csv from featex.py')
    parser.add_argument('--features', required=True,
                        help='pca50.csv from featex.py')
    parser.add_argument('--models',   required=True,
                        help='Directory containing cls_final.joblib & reg_final.joblib')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability cutoff for alert')
    args = parser.parse_args()
    main(args.records, args.features, args.models, args.threshold)