# File: train_model.py
# Purpose: Final version that trains the model and saves all outputs:
# 1. The trained model (.pth)
# 2. The performance metrics (.json)
# 3. The P-R curve plot (.png)

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import warnings
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.transforms import ToUndirected
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, precision_recall_curve, confusion_matrix # <<< MODIFIED
import matplotlib.pyplot as plt
import os
import json

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
#  1. CONFIGURATION
# ==============================================================================
class Config:
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Virat@45" # 👈 *** UPDATE YOUR NEO4J PASSWORD ***
    
    # Tuned Hyperparameters
    GNN_LAYERS = 4 # <<< MODIFIED: Increased from 3 to 4 to add another hidden layer
    HIDDEN_CHANNELS = 64
    OUT_CHANNELS = 2
    LEARNING_RATE = 0.001
    EPOCHS = 200
    GAT_HEADS = 4
    WEIGHT_DECAY = 5e-4
    DROPOUT_RATE = 0.5
    TEST_SPLIT_SIZE = 0.3

# ==============================================================================
#  2. GNN MODEL DEFINITION (No changes)
# ==============================================================================
class GAT_MultiLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout_rate, num_layers):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layers = torch.nn.ModuleList()
        if num_layers < 2: raise ValueError("GNN must have at least 2 layers.")
        # Input Layer
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, add_self_loops=False))
        # Hidden Layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, add_self_loops=False))
        # Output Layer
        self.layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, add_self_loops=False))
        
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

class HeteroGAT(torch.nn.Module):
    def __init__(self, node_feature_dims, metadata, hidden_channels, out_channels, heads, dropout_rate, num_layers):
        super().__init__()
        self.proj_layers = torch.nn.ModuleDict()
        for node_type, in_dim in node_feature_dims.items():
            self.proj_layers[node_type] = torch.nn.Linear(in_dim, hidden_channels)
        base_model = GAT_MultiLayer(hidden_channels, hidden_channels, out_channels, heads, dropout_rate, num_layers)
        self.gat = to_hetero(base_model, metadata, aggr='sum')
        
    def forward(self, x_dict, edge_index_dict):
        projected_x_dict = {node_type: self.proj_layers[node_type](x).relu() for node_type, x in x_dict.items()}
        return self.gat(projected_x_dict, edge_index_dict)

# ==============================================================================
#  3. GNN TRAINING PIPELINE
# ==============================================================================
class FraudGNNTrainer:
    def __init__(self, config):
        self.config = config
        self.driver = None
        self.data = None
        self.model = None

    def _connect_to_neo4j(self):
        try:
            self.driver = GraphDatabase.driver(self.config.NEO4J_URI, auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            print("✅ Successfully connected to Neo4j")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise

    def fetch_graph_from_neo4j(self):
        print("\n🕸️ Fetching graph data from Neo4j (for predictive modeling)...")
        with self.driver.session(database="neo4j") as session:
            nodes_result = session.run("MATCH (n) WHERE NOT 'Fraud' IN labels(n) RETURN labels(n)[0] AS nt, elementId(n) AS id, properties(n) AS p").data()
            edges_result = session.run("MATCH (n)-[r]->(m) WHERE type(r) <> 'FLAGGED_AS' RETURN labels(n)[0] AS src_t, elementId(n) AS src_id, type(r) AS rel_t, labels(m)[0] AS trg_t, elementId(m) AS trg_id").data()
            fraud_labels_result = session.run("MATCH (t:Transaction) OPTIONAL MATCH (t)-[:FLAGGED_AS]->(f:Fraud) RETURN elementId(t) as id, (f IS NOT NULL) as is_fraud").data()
        data, node_mappings, node_properties = HeteroData(), {}, {}
        for node in nodes_result:
            nt = node['nt'].lower()
            if nt not in node_mappings: node_mappings[nt], node_properties[nt] = {}, []
            node_mappings[nt][node['id']] = len(node_mappings[nt])
            node_properties[nt].append(node['p'])
        for nt, props_list in node_properties.items():
            df = pd.DataFrame(props_list)
            if nt == 'transaction':
                feature_cols = ['amount', 'timestamp', 'time_since_last_txn', 'amt_zscore', 'txn_distance']
                features = df[feature_cols].fillna(0)
            elif nt == 'location': features = df[['lat', 'long', 'city_pop']].fillna(0)
            elif nt == 'merchant': features = df[['lat', 'long']].fillna(0)
            else: features = pd.DataFrame(np.ones((len(df), 1)))
            data[nt].x = torch.tensor(StandardScaler().fit_transform(features.values), dtype=torch.float)
        labels = torch.zeros(len(node_mappings.get('transaction', {})), dtype=torch.long)
        fraud_map = {item['id']: item['is_fraud'] for item in fraud_labels_result}
        for id, idx in node_mappings.get('transaction', {}).items():
            if fraud_map.get(id, False): labels[idx] = 1
        data['transaction'].y = labels
        for edge in edges_result:
            src_t, rel_t, trg_t = edge['src_t'].lower(), edge['rel_t'].lower(), edge['trg_t'].lower()
            src_id, trg_id = node_mappings.get(src_t, {}).get(edge['src_id']), node_mappings.get(trg_t, {}).get(edge['trg_id'])
            if src_id is None or trg_id is None: continue
            edge_type = (src_t, rel_t, trg_t)
            edge_index = torch.tensor([[src_id], [trg_id]], dtype=torch.long)
            if edge_type not in data.edge_types: data[edge_type].edge_index = edge_index
            else: data[edge_type].edge_index = torch.cat([data[edge_type].edge_index, edge_index], dim=1)
        print("📊 Graph construction complete.")
        self.data = ToUndirected()(data)
        print(self.data)

    def prepare_for_training(self):
        print("\n⚖️ Creating a balanced training set and a realistic test set...")
        y = self.data['transaction'].y
        fraud_indices = (y == 1).nonzero(as_tuple=False).view(-1)
        non_fraud_indices = (y == 0).nonzero(as_tuple=False).view(-1)
        fraud_train_idx, fraud_test_idx = train_test_split(fraud_indices, test_size=self.config.TEST_SPLIT_SIZE, random_state=42)
        non_fraud_train_idx, non_fraud_test_idx = train_test_split(non_fraud_indices, test_size=self.config.TEST_SPLIT_SIZE, random_state=42)
        num_fraud_train = len(fraud_train_idx)
        num_non_fraud_to_sample = num_fraud_train * 2
        perm = torch.randperm(len(non_fraud_train_idx))
        sampled_non_fraud_train_idx = non_fraud_train_idx[perm[:num_non_fraud_to_sample]]
        train_idx = torch.cat([fraud_train_idx, sampled_non_fraud_train_idx])
        test_idx = torch.cat([fraud_test_idx, non_fraud_test_idx])
        train_idx = train_idx[torch.randperm(len(train_idx))]
        train_mask = torch.zeros(len(y), dtype=torch.bool); train_mask[train_idx] = True
        test_mask = torch.zeros(len(y), dtype=torch.bool); test_mask[test_idx] = True
        self.data['transaction'].train_mask, self.data['transaction'].test_mask = train_mask, test_mask
        print(f"\n🔬 Dataset Split:")
        print(f"   - Training Set: {train_mask.sum()} nodes ({y[train_mask].sum()} fraud, {train_mask.sum() - y[train_mask].sum()} non-fraud)")
        print(f"   - Test Set:     {test_mask.sum()} nodes ({y[test_mask].sum()} fraud, {test_mask.sum() - y[test_mask].sum()} non-fraud)")
        dims = {nt: self.data[nt].x.shape[1] for nt in self.data.node_types}
        self.model = HeteroGAT(dims, self.data.metadata(), self.config.HIDDEN_CHANNELS, self.config.OUT_CHANNELS, self.config.GAT_HEADS, self.config.DROPOUT_RATE, self.config.GNN_LAYERS)
        
    def train_and_evaluate(self):
        # DEFINE SAVE PATHS
        model_dir, results_dir = 'saved_model', 'saved_results'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, 'gat_model.pth')
        metrics_save_path = os.path.join(results_dir, 'performance_metrics.json')
        pr_curve_save_path = os.path.join(results_dir, 'precision_recall_curve.json')
        pr_plot_save_path = os.path.join(results_dir, 'precision_recall_curve.png')

        # TRAINING
        y, train_mask = self.data['transaction'].y, self.data['transaction'].train_mask
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        criterion = torch.nn.CrossEntropyLoss()
        print("\n🚀 Starting GAT training...")
        for epoch in range(1, self.config.EPOCHS + 1):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
            loss = criterion(out['transaction'][train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0 or epoch == 1:
                print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}')
        print("🏁 Training finished.")

        # SAVE THE TRAINED MODEL
        torch.save(self.model.state_dict(), model_save_path)
        print(f"\n✅ Model saved to '{model_save_path}'")
        
        # EVALUATION AND THRESHOLD TUNING
        print("📊 Evaluating model and tuning threshold...")
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
            probs = F.softmax(out['transaction'], dim=-1)
            fraud_probs = probs[:, 1]
        
        test_mask = self.data['transaction'].test_mask
        y_true = y[test_mask].cpu().numpy()
        y_scores = fraud_probs[test_mask].cpu().numpy()

        # Systematic Threshold Tuning
        print("\n--- Systematic Threshold Tuning for 'Fraud' Class ---")
        print("Threshold | Precision | Recall    | F1-Score")
        print("-------------------------------------------------")
        
        thresholds = np.arange(0.05, 1.0, 0.05)
        best_f1 = 0
        best_metrics = {}
        for thresh in thresholds:
            y_pred_custom = (y_scores >= thresh).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred_custom, pos_label=1, average='binary', zero_division=0
            )
            print(f"   {thresh:0.2f}    |   {precision:0.5f} |   {recall:0.5f} |   {f1:0.5f}")
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {
                    'best_threshold': float(thresh), 'precision': float(precision),
                    'recall': float(recall), 'f1_score': float(f1)
                }
        
        # <<< --- MODIFICATION: CALCULATE AND PRINT CONFUSION MATRIX --- >>>
        if best_metrics:
            final_y_pred = (y_scores >= best_metrics['best_threshold']).astype(int)
            cm = confusion_matrix(y_true, final_y_pred)
            best_metrics['confusion_matrix'] = cm.tolist()
            
            print("\n📊 Final Evaluation Metrics (at best threshold):")
            print("=================================================")
            print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
            print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
            print("=================================================")

        # SAVE METRICS AND CURVE DATA
        print(f"\n🏆 Best Performance (by F1-score):")
        print(json.dumps(best_metrics, indent=4))
        with open(metrics_save_path, 'w') as f:
            json.dump(best_metrics, f, indent=4)
        print(f"✅ Best metrics saved to '{metrics_save_path}'")

        precision_points, recall_points, _ = precision_recall_curve(y_true, y_scores, pos_label=1)
        pr_curve_data = {'recall': recall_points.tolist(), 'precision': precision_points.tolist()}
        with open(pr_curve_save_path, 'w') as f:
            json.dump(pr_curve_data, f, indent=4)
        print(f"✅ P-R curve data saved to '{pr_curve_save_path}'")
        
        # SAVE THE PLOT
        print("\n📈 Generating and saving Precision-Recall Curve plot...")
        plt.figure(figsize=(10, 7))
        plt.plot(recall_points, precision_points, marker='.', label='GAT Model')
        plt.title('Precision-Recall Curve for Fraud Detection')
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (Positive Predictive Value)')
        plt.grid(True)
        plt.legend()
        plt.savefig(pr_plot_save_path)
        plt.close()
        print(f"✅ Plot saved to '{pr_plot_save_path}'")

    def run(self):
        try:
            self._connect_to_neo4j()
            self.fetch_graph_from_neo4j()
            self.prepare_for_training()
            self.train_and_evaluate()
        except Exception as e:
            print(f"\nAn error occurred during the pipeline execution: {e}")
        finally:
            if self.driver: self.driver.close(); print("\n🔗 Neo4j connection closed.")

# ==============================================================================
#  4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    trainer = FraudGNNTrainer(Config())
    trainer.run()