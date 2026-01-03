import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import timm
import flwr as fl
from collections import OrderedDict
import sys
import os

# --- 1. SETTINGS ---
# The path you provided
DATA_PATH = r"C:\Users\muham\OneDrive\Desktop\CKDupdated" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get Client ID from terminal argument (e.g., python client.py 0)
CLIENT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# --- 2. PATH VERIFICATION ---
if not os.path.exists(DATA_PATH):
    print(f"❌ ERROR: The path {DATA_PATH} does not exist.")
    sys.exit()
else:
    subfolders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
    print(f"✅ Folder Found! Detected {len(subfolders)} classes: {subfolders}")

# --- 3. DATA SILO LOGIC ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_data(partition_id):
    full_dataset = ImageFolder(root=DATA_PATH, transform=transform)
    
    # Split the dataset into 3 equal parts (silos)
    total_len = len(full_dataset)
    part_len = total_len // 3
    lengths = [part_len, part_len, total_len - (part_len * 2)]
    
    silos = random_split(full_dataset, lengths, generator=torch.Generator().manual_seed(42))
    local_data = silos[partition_id]
    
    return DataLoader(local_data, batch_size=16, shuffle=True)

# --- 4. MODEL & CLIENT CLASS ---
# Standardizing on vit_base_patch16_224 as per your previous notebooks
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=4).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

class CKDClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loader = load_data(CLIENT_ID)
        model.train()
        print(f"🏃 Client {CLIENT_ID} starting local training...")
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i > 10: break # Training on a small batch for testing rounds
            
        print(f"✅ Client {CLIENT_ID} finished local training.")
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # In a real scenario, you'd use a separate test set here
        return 0.0, 10, {"accuracy": 0.95}

# --- 5. START ---
if __name__ == "__main__":
    print(f"🚀 Client {CLIENT_ID} connecting to Server via {DEVICE}...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=CKDClient().to_client(),
        insecure=True,
    )
