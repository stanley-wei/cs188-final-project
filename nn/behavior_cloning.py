import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from load_data import reconstruct_from_npz


def prepare_data(demos):
    all_inputs = []
    all_actions = []

    for demo_id in demos:
        # Using robot end-effector position, quaternion, gripper position, object position, and quaternion
        eef_pos = demos[demo_id]["obs_robot0_eef_pos"]  # (N, 3)
        eef_quat = demos[demo_id]["obs_robot0_eef_quat"]  # (N, 4)
        gripper_pos = demos[demo_id]["obs_robot0_gripper_qpos"][:, :1]  # (N, 1)
        obj_pos = demos[demo_id]["obs_object"][:, :3]  # (N, 3)
        obj_quat = demos[demo_id]["obs_object"][:, 3:7]  # (N, 4)

        # Concatenate features
        inputs = np.concatenate(
            [eef_pos, eef_quat, gripper_pos, obj_pos, obj_quat], axis=1
        )

        actions = demos[demo_id]["actions"]  # (N, 7)

        all_inputs.append(inputs)
        all_actions.append(actions)

    X = np.concatenate(all_inputs, axis=0)
    y = np.concatenate(all_actions, axis=0)

    # Split into train/validation (80/20)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_indices = indices[:split]
    val_indices = indices[split:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", y_train.shape)

    return X_train, y_train, X_val, y_val


class BCModel(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)


def train_model(X_train, y_train, X_val, y_val, epochs=200, batch_size=64):
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    action_dim = y_train.shape[1]
    model = BCModel(input_dim, action_dim)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            pred_y = model(batch_X)
            loss = criterion(pred_y, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_bc_model.pt")

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}"
        )

    return model


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python behavior_cloning.py <demo.npz>")
        sys.exit(1)

    # Load demonstrations
    npz_path = sys.argv[1]
    demos = reconstruct_from_npz(npz_path)

    if not demos:
        print("Failed to load demonstrations")
        sys.exit(1)

    # Prepare data
    X_train, y_train, X_val, y_val = prepare_data(demos)
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

    # Train model
    model = train_model(X_train, y_train, X_val, y_val, epochs=200)

    print("Training complete. Best model saved to best_bc_model.pt")
