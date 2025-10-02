from model import NGP_TCNN
import torch
import torch.nn as nn
import torch.optim as optim
from config import device, opt
from dataio import load_volume_data
import os


def train_stage1_model(num_epochs=10, lr=1e-3, batch_size=65536):
    # Load volume data
    vol, dims = load_volume_data()  # vol: (1,1,D,H,W), dims: [X,Y,Z]
    vol = vol.squeeze(0).squeeze(0)  # (D,H,W)
    D, H, W = vol.shape

    model = NGP_TCNN(opt).to(device)
    model.train()

    # Prepare coordinates grid in [-1,1] to match the INR forward expectation
    z = torch.linspace(-1, 1, D, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # (N,3)
    targets = vol.reshape(-1, 1)  # (N,1)

    N = coords.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch_coords = coords[idx]
            batch_targets = targets[idx]
            optimizer.zero_grad()
            preds = model(batch_coords)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_coords.size(0)
        epoch_loss /= N
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f}")

    print("Training complete.")
    psnr, pred_vol = evaluate_psnr_and_save_raw(model, vol)
    print(f"Final PSNR: {psnr:.2f} dB")

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/stage1_ngp_tcnn.pth")
    return model, pred_vol

def evaluate_psnr_and_save_raw(model, vol, save_path="results/stage1/prediction.raw"):
    model.eval()
    with torch.no_grad():
        D, H, W = vol.shape
        z = torch.linspace(-1, 1, D, device=vol.device)
        y = torch.linspace(-1, 1, H, device=vol.device)
        x = torch.linspace(-1, 1, W, device=vol.device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # (N,3)
        targets = vol.reshape(-1, 1)  # (N,1)
        preds = model(coords)
        mse = nn.MSELoss()(preds, targets).item()
        psnr = -10 * torch.log10(torch.tensor(mse))
        # Save prediction as .raw
        pred_vol = preds.reshape(D, H, W).cpu().numpy()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pred_vol.astype(vol.cpu().numpy().dtype).tofile(save_path)
    return psnr.item(), pred_vol
