import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Loading
print("Loading Cora dataset...")
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

# Convert to scipy sparse matrix
adj = to_scipy_sparse_matrix(data.edge_index)
features = data.x.cpu().numpy()
labels = data.y.cpu().numpy()

# Victim Model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

print("Training GCN...")
for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

train_acc, val_acc, test_acc = test()
print(f'Final Test Accuracy: {test_acc:.4f}')

# Nettack Implementation
class Nettack:
    def __init__(self, model, adj, features, labels, target_node, device='cpu'):
        self.model = model
        self.adj = adj.tolil() # Use LIL for efficient structure modifications
        self.features = features.copy() # Numpy array
        self.labels = labels
        self.target_node = target_node
        self.device = device
        
        # Surrogate model parameters (Linearized GCN)
        self.W1 = model.conv1.lin.weight.detach().cpu().numpy().T
        self.W2 = model.conv2.lin.weight.detach().cpu().numpy().T
        self.W = self.W1.dot(self.W2)
        
        self.num_nodes = adj.shape[0]
        self.num_features = features.shape[1]
        
        # Precompute A_hat^2 (approximate propagation)
        self.adj_norm = self.normalize_adj(self.adj)
        
    def normalize_adj(self, adj):
        # A_hat = D^-0.5 (A + I) D^-0.5
        adj_aug = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_aug.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj_aug).dot(d_mat_inv_sqrt).tocsr()

    def compute_logits(self, adj_norm, features):
        # Logits = A^2 X W
        return adj_norm.dot(adj_norm).dot(features).dot(self.W)
    
    def get_surrogate_loss(self, adj_norm, features, target_node, target_label):
        logits = self.compute_logits(adj_norm, features)
        logits_target = logits[target_node]
        
        # Softmax
        probs = np.exp(logits_target) / np.sum(np.exp(logits_target))
        return probs[target_label]

    def attack(self, n_perturbations):
        print(f"Attacking node {self.target_node} (Class {self.labels[self.target_node]})...")
        
        modified_adj = self.adj.copy()
        
        # Get initial prediction
        adj_norm = self.normalize_adj(modified_adj)
        initial_prob = self.get_surrogate_loss(adj_norm, self.features, self.target_node, self.labels[self.target_node])
        print(f"Initial Correct Class Probability (Surrogate): {initial_prob:.4f}")
        
        for i in range(n_perturbations):
            best_edge = None
            best_prob = 1.0 # We want to minimize this
            
            # Greedy search over edges connected to target node (1-hop)
            candidates = []
            
            # Potential additions (connect target to others)
            potential_neighbors = np.random.choice(self.num_nodes, 50, replace=False)
            for node in potential_neighbors:
                if node != self.target_node and modified_adj[self.target_node, node] == 0:
                    candidates.append((self.target_node, node, 1)) # Add edge
            
            # Potential deletions (remove existing edges)
            neighbors = modified_adj[self.target_node].nonzero()[1]
            for node in neighbors:
                candidates.append((self.target_node, node, 0)) # Remove edge
                
            # Evaluate candidates
            for u, v, action in candidates:
                # Apply perturbation
                if action == 1:
                    modified_adj[u, v] = 1
                    modified_adj[v, u] = 1
                else:
                    modified_adj[u, v] = 0
                    modified_adj[v, u] = 0
                
                # Evaluate
                adj_norm = self.normalize_adj(modified_adj)
                prob = self.get_surrogate_loss(adj_norm, self.features, self.target_node, self.labels[self.target_node])
                
                if prob < best_prob:
                    best_prob = prob
                    best_edge = (u, v, action)
                
                # Revert perturbation
                if action == 1:
                    modified_adj[u, v] = 0
                    modified_adj[v, u] = 0
                else:
                    modified_adj[u, v] = 1
                    modified_adj[v, u] = 1
            
            if best_edge:
                u, v, action = best_edge
                if action == 1:
                    modified_adj[u, v] = 1
                    modified_adj[v, u] = 1
                    print(f"Step {i+1}: Added edge ({u}, {v}). New Prob: {best_prob:.4f}")
                else:
                    modified_adj[u, v] = 0
                    modified_adj[v, u] = 0
                    print(f"Step {i+1}: Removed edge ({u}, {v}). New Prob: {best_prob:.4f}")
            else:
                print("No beneficial perturbation found.")
                break
                
        return modified_adj

# Select Target Node
model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)
probs = torch.exp(out)

target_node = -1
for i in range(data.num_nodes):
    if data.test_mask[i] and pred[i] == data.y[i] and probs[i, pred[i]] > 0.9:
        target_node = i
        break

if target_node == -1:
    print("Could not find a suitable target node.")
    exit()

print(f"Selected Target Node: {target_node}")
print(f"True Label: {data.y[target_node].item()}")
print(f"Initial Prediction: {pred[target_node].item()} (Conf: {probs[target_node, pred[target_node]]:.4f})")

# Run Attack
nettack = Nettack(model, adj, features, labels, target_node, device=device)
modified_adj = nettack.attack(n_perturbations=5)

# Evaluate
modified_edge_index, _ = from_scipy_sparse_matrix(modified_adj)
modified_edge_index = modified_edge_index.to(device)

model.eval()
out_pert = model(data.x, modified_edge_index)
pred_pert = out_pert.argmax(dim=1)
probs_pert = torch.exp(out_pert)

print("\n--- After Attack ---")
print(f"Prediction: {pred_pert[target_node].item()} (Conf: {probs_pert[target_node, pred_pert[target_node]]:.4f})")
if pred_pert[target_node] != data.y[target_node]:
    print("SUCCESS: Attack flipped the label!")
else:
    print("FAILURE: Label not flipped.")
