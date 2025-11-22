# Author: Vinh Dang (dqvinh87@gmail.com)

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
# Load the Cora dataset from Planetoid
# NormalizeFeatures ensures that node features sum to 1 (row-normalization)
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

# Convert to scipy sparse matrix
# This is easier to manipulate for structural attacks (adding/removing edges)
adj = to_scipy_sparse_matrix(data.edge_index)
features = data.x.cpu().numpy()
labels = data.y.cpu().numpy()

# Victim Model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        # First Graph Convolution layer: Input features -> 16 hidden units
        self.conv1 = GCNConv(num_features, 16)
        # Second Graph Convolution layer: 16 hidden units -> Output classes
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        # x: Node feature matrix
        # edge_index: Graph connectivity (adjacency list)
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x) # ReLU activation
        x = F.dropout(x, training=self.training) # Dropout for regularization
        
        # Layer 2
        x = self.conv2(x, edge_index)
        
        # Log Softmax for classification probability
        return F.log_softmax(x, dim=1)

model = GCN(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # Calculate loss only on training nodes
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
        self.adj = adj.tolil() # Use LIL (List of Lists) format for efficient structure modifications
        self.features = features.copy() # Numpy array
        self.labels = labels
        self.target_node = target_node # The node we want to misclassify
        self.device = device
        
        # Surrogate model parameters (Linearized GCN)
        # We extract the weights W1 and W2 from the trained GCN model.
        # We detach them from the computation graph because we don't want to update them during the attack.
        # We transpose (.T) them to match the matrix multiplication shape: A * X * W
        self.W1 = model.conv1.lin.weight.detach().cpu().numpy().T
        self.W2 = model.conv2.lin.weight.detach().cpu().numpy().T
        
        # The linearized GCN collapses the two layers into one linear transformation:
        # Logits = A_hat * A_hat * X * W1 * W2
        # We precompute W = W1 * W2 for efficiency.
        self.W = self.W1.dot(self.W2)
        
        self.num_nodes = adj.shape[0]
        self.num_features = features.shape[1]
        
        # Precompute normalized adjacency matrix A_hat
        self.adj_norm = self.normalize_adj(self.adj)
        
    def normalize_adj(self, adj):
        # GCN uses a specific normalization: A_hat = D^-0.5 * (A + I) * D^-0.5
        # where A is adjacency, I is identity (self-loops), D is degree matrix.
        
        # 1. Add self-loops (A + I)
        adj_aug = adj + sp.eye(adj.shape[0])
        
        # 2. Calculate degrees (row sums)
        rowsum = np.array(adj_aug.sum(1))
        
        # 3. Calculate D^-0.5
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # 4. Compute D^-0.5 * (A + I) * D^-0.5
        return d_mat_inv_sqrt.dot(adj_aug).dot(d_mat_inv_sqrt).tocsr()

    def compute_logits(self, adj_norm, features):
        # Compute the output of the linearized GCN (Surrogate Model)
        # Formula: Z = A_hat^2 * X * W
        # This approximates the 2-layer GCN without the non-linearity (ReLU).
        # It's much faster to compute and easier to get gradients from.
        return adj_norm.dot(adj_norm).dot(features).dot(self.W)
    
    def get_surrogate_loss(self, adj_norm, features, target_node, target_label):
        # Compute the probability of the target label for the target node.
        # We want to MINIMIZE this probability to cause misclassification.
        
        logits = self.compute_logits(adj_norm, features)
        logits_target = logits[target_node]
        
        # Apply Softmax to get probabilities
        probs = np.exp(logits_target) / np.sum(np.exp(logits_target))
        return probs[target_label]

    def attack(self, n_perturbations):
        print(f"Attacking node {self.target_node} (Class {self.labels[self.target_node]})...")
        
        modified_adj = self.adj.copy()
        
        # Get initial prediction confidence using the surrogate model
        adj_norm = self.normalize_adj(modified_adj)
        initial_prob = self.get_surrogate_loss(adj_norm, self.features, self.target_node, self.labels[self.target_node])
        print(f"Initial Correct Class Probability (Surrogate): {initial_prob:.4f}")
        
        # Greedy Attack Loop
        # In each step, we pick ONE edge to flip (add or remove) that maximally decreases the correct class probability.
        for i in range(n_perturbations):
            best_edge = None
            best_prob = 1.0 # Initialize with max probability, we want to find something lower
            
            # Candidate Generation
            # We look for potential edges to add or remove.
            # u is always the target_node. v is a neighbor or potential neighbor.
            candidates = []
            
            # 1. Potential Additions (Connect target_node to a new node v)
            # For efficiency, we randomly sample 50 nodes instead of checking all nodes.
            potential_neighbors = np.random.choice(self.num_nodes, 50, replace=False)
            for node in potential_neighbors:
                # Check if edge doesn't exist yet
                if node != self.target_node and modified_adj[self.target_node, node] == 0:
                    candidates.append((self.target_node, node, 1)) # Action 1: Add edge
            
            # 2. Potential Deletions (Remove existing edge between target_node and v)
            # We check all current neighbors of the target_node.
            neighbors = modified_adj[self.target_node].nonzero()[1]
            for node in neighbors:
                candidates.append((self.target_node, node, 0)) # Action 0: Remove edge
                
            # Evaluate Candidates
            # We try each candidate perturbation, compute the loss, and pick the best one.
            for u, v, action in candidates:
                # Apply perturbation temporarily
                if action == 1:
                    # Add edge (u, v)
                    modified_adj[u, v] = 1
                    modified_adj[v, u] = 1
                else:
                    # Remove edge (u, v)
                    modified_adj[u, v] = 0
                    modified_adj[v, u] = 0
                
                # Re-normalize adjacency matrix with the change
                adj_norm = self.normalize_adj(modified_adj)
                
                # Compute new probability of the correct class
                prob = self.get_surrogate_loss(adj_norm, self.features, self.target_node, self.labels[self.target_node])
                
                # If this perturbation reduces the probability more than the current best, keep it
                if prob < best_prob:
                    best_prob = prob
                    best_edge = (u, v, action)
                
                # Revert perturbation to try the next candidate
                if action == 1:
                    modified_adj[u, v] = 0
                    modified_adj[v, u] = 0
                else:
                    modified_adj[u, v] = 1
                    modified_adj[v, u] = 1
            
            # Apply the Best Perturbation Found
            if best_edge:
                u, v, action = best_edge
                if action == 1:
                    modified_adj[u, v] = 1
                    modified_adj[v, u] = 1
                    print(f"Step {i+1}: Added edge between {u} and {v}. New Correct Class Prob: {best_prob:.4f}")
                else:
                    modified_adj[u, v] = 0
                    modified_adj[v, u] = 0
                    print(f"Step {i+1}: Removed edge between {u} and {v}. New Correct Class Prob: {best_prob:.4f}")
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
    # We want a node that is:
    # 1. In the test set (data.test_mask[i])
    # 2. Correctly classified (pred[i] == data.y[i])
    # 3. High confidence (> 0.9)
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

# Evaluate on perturbed graph
# Convert the modified scipy sparse matrix back to PyTorch Geometric edge_index
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
