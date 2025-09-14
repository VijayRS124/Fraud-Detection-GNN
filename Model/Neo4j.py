# File: load_data.py
# Purpose: Reads a specific list of CSV files, samples records from each at a 2:1
# non-fraud to fraud ratio, engineers features (including transaction distance),
# and loads the combined data into a Neo4j database.
# RUN THIS SCRIPT FIRST.

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import os

# ==============================================================================
#  1. CONFIGURATION
# ==============================================================================
class Config:
    """Configuration for data loading from a specific list of files."""
    # Define the base folder where all CSVs are located
    BASE_DATA_FOLDER = r"dataset.csv"

    # List the exact 10 CSV filenames you want to process
    CSV_FILE_LIST = [
        "adults_2550_female_rural_00-19.csv",
        "adults_2550_female_rural_200-399.csv",
        "adults_2550_female_urban_00-19.csv",
        "adults_2550_female_urban_200-399.csv",
        "adults_2550_male_rural_00-19.csv",
        "adults_2550_male_rural_200-399.csv",
        "adults_50up_female_rural_40-59.csv",
        "adults_50up_female_rural_400-599.csv",
        "adults_50up_male_urban_60-79.csv",
        "adults_50up_male_urban_600-799.csv",
    ]
    
    # Neo4j Credentials
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "vijay@123" 

# ==============================================================================
#  2. HELPER FUNCTIONS
# ==============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth in kilometers."""
    R = 6371  # Radius of Earth
    
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def classify_job_category(job_title):
    if not isinstance(job_title, str): return "Other"
    job_title = job_title.lower()
    if any(x in job_title for x in ['accountant', 'auditor', 'financial', 'tax']): return 'Finance'
    if any(x in job_title for x in ['doctor', 'nurse', 'therapist', 'medical']): return 'Healthcare'
    if any(x in job_title for x in ['teacher', 'educator', 'lecturer']): return 'Education'
    if any(x in job_title for x in ['scientist', 'research']): return 'Science & Research'
    if any(x in job_title for x in ['it', 'developer', 'software']): return 'IT'
    else: return 'Other'

def engineer_features(df):
    """Creates new, informative features from the raw data."""
    print("✨ Engineering new features for the combined dataset...")
    df_sorted = df.sort_values(['acct_num', 'unix_time']).copy()
    
    df_sorted['time_since_last_txn'] = df_sorted.groupby('acct_num')['unix_time'].diff().fillna(0)
    
    stats = df_sorted.groupby('acct_num')['amt'].agg(['mean', 'std']).reset_index()
    stats.columns = ['acct_num', 'acct_mean_amt', 'acct_std_amt']
    df_sorted = pd.merge(df_sorted, stats, on='acct_num', how='left')
    df_sorted['amt_zscore'] = (df_sorted['amt'] - df_sorted['acct_mean_amt']) / (df_sorted['acct_std_amt'] + 1e-6)
    
    print("   -> Calculating distance between account holder and merchant...")
    for col in ['lat', 'long', 'merch_lat', 'merch_long']:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce').fillna(0)
    
    df_sorted['txn_distance'] = haversine_distance(
        df_sorted['lat'], df_sorted['long'],
        df_sorted['merch_lat'], df_sorted['merch_long']
    )
    
    df_sorted.fillna({'amt_zscore': 0, 'txn_distance': 0}, inplace=True)
    return df_sorted

# ==============================================================================
#  3. MAIN DATA LOADING LOGIC
# ==============================================================================
def main():
    config = Config()
    all_sampled_dfs = []
    
    print(f"📂 Processing a specific list of {len(config.CSV_FILE_LIST)} CSV files...")

    for filename in config.CSV_FILE_LIST:
        try:
            file_path = os.path.join(config.BASE_DATA_FOLDER, filename)
            
            if not os.path.exists(file_path):
                print(f"   ⚠️ Warning: File not found at '{file_path}'. Skipping.")
                continue

            print(f"   -> Processing file: {filename}")
            df = pd.read_csv(file_path, delimiter='|')

            if 'is_fraud' not in df.columns:
                print(f"   ⚠️ Warning: Skipping file {filename} (missing 'is_fraud' column).")
                continue
            
            fraud_df = df[df['is_fraud'] == 1]
            non_fraud_df = df[df['is_fraud'] == 0]
            num_fraud_in_file = len(fraud_df)

            if num_fraud_in_file == 0:
                print(f"   -> No fraud cases found in {filename}. Skipping file.")
                continue
            
            num_non_fraud_to_sample = num_fraud_in_file * 2
            n_non_fraud_samples = min(len(non_fraud_df), num_non_fraud_to_sample)
            
            print(f"      - Sampling: {num_fraud_in_file} fraud and {n_non_fraud_samples} non-fraud records.")
            
            sampled_fraud = fraud_df
            sampled_non_fraud = non_fraud_df.sample(n=n_non_fraud_samples, random_state=42)
            all_sampled_dfs.append(sampled_fraud)
            all_sampled_dfs.append(sampled_non_fraud)

        except Exception as e:
            print(f"   ⚠️ Warning: Could not process file {filename}. Error: {e}")

    if not all_sampled_dfs:
        print("❌ No data could be sampled. Exiting.")
        return

    master_df = pd.concat(all_sampled_dfs, ignore_index=True)
    print(f"\n✅ Combined all files. Total records to load: {len(master_df)}")
    print(f"   - Fraud records: {master_df['is_fraud'].sum()}")
    print(f"   - Non-Fraud records: {len(master_df) - master_df['is_fraud'].sum()}")

    master_df = master_df.replace({np.nan: None})
    master_df['job_category'] = master_df['job'].apply(classify_job_category)
    master_df.drop_duplicates(subset=['trans_num'], inplace=True)
    master_df = engineer_features(master_df)
    
    driver = None
    try:
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("\n✅ Successfully connected to Neo4j")
        
        with driver.session(database="neo4j") as session:
            print("🧹 Clearing existing data from Neo4j...")
            session.run("MATCH (n) DETACH DELETE n")
            print("🛠️ Creating graph nodes and relationships in Neo4j...")
            query = """
            UNWIND $rows AS row
            MERGE (a:Account {acct_num: row.acct_num})
            MERGE (l:Location {zip: row.zip})
                SET l.lat = coalesce(l.lat, toFloat(row.lat)),
                    l.long = coalesce(l.long, toFloat(row.long)),
                    l.city_pop = coalesce(l.city_pop, toInteger(row.city_pop))
            MERGE (t:Transaction {trans_num: row.trans_num})
                ON CREATE SET t.amount = toFloat(row.amt), t.timestamp = toInteger(row.unix_time),
                              t.time_since_last_txn = toInteger(row.time_since_last_txn), 
                              t.amt_zscore = toFloat(row.amt_zscore),
                              t.txn_distance = toFloat(row.txn_distance)
            MERGE (m:Merchant {name: row.merchant})
                 SET m.lat = coalesce(m.lat, toFloat(row.merch_lat)),
                     m.long = coalesce(m.long, toFloat(row.merch_long))
            MERGE (c:Category {name: row.category})
            MERGE (jc:JobCategory {name: row.job_category})
            MERGE (a)-[:LIVES_IN]->(l)
            MERGE (a)-[:HAS_JOB_CATEGORY]->(jc)
            MERGE (a)-[:MADE]->(t)
            MERGE (t)-[:AT]->(m)
            MERGE (m)-[:BELONGS_TO]->(c)
            MERGE (m)-[:LOCATED_IN]->(l)
            FOREACH (fraud IN CASE WHEN row.is_fraud = 1 THEN [1] ELSE [] END |
                MERGE (f:Fraud {id: 'fraud_' + row.trans_num})
                MERGE (t)-[:FLAGGED_AS]->(f)
            )
            """
            records = master_df.to_dict('records')
            session.run(query, parameters={'rows': records})
            
        print("\n🎉 Neo4j database loaded successfully! You can now run the train_model.py script.")
        
    except Exception as e:
        print(f"❌ An error occurred during Neo4j operation: {e}")
    finally:
        if driver:
            driver.close()
            print("\n🔗 Neo4j connection closed.")

if __name__ == "__main__":
    main()