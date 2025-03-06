import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from mamba import Mamba, MambaConfig
import argparse
import wandb
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--use-mps', default=True,
                    help='MPS training.')
parser.add_argument('--seed', type=int, default=69, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=0.001,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Minimum dimension of representations')
parser.add_argument('--layer', type=int, default=3,
                    help='Num of layers')
parser.add_argument('--n-test', type=int, default=30,
                    help='Size of test set')
parser.add_argument('--data-path', type=str, default='data/processed/SPY_featured.csv',
                    help='Path to the input CSV file')
parser.add_argument('--use-wandb', default=True, action='store_true',
                    help='Enable Weights & Biases logging')
parser.add_argument('--batch-size', type=int, default=256,
                    help='Batch size for training')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
args.mps = args.use_mps and torch.backends.mps.is_available()
device = torch.device(
    "cuda" if args.cuda else 
    "mps" if args.mps else 
    "cpu"
)

def evaluation_metric(y_test,y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test,y_hat)
    R2 = r2_score(y_test,y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2))

def set_seed(seed,cuda,mps):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    if mps:
        torch.mps.manual_seed(seed)

def dateinf(series, n_test):
    lt = len(series)
    print('Training start:', series.iloc[0])
    print('Training end:', series.iloc[lt-n_test-1])
    print('Testing start:', series.iloc[lt-n_test])
    print('Testing end:', series.iloc[lt-1])

set_seed(args.seed, args.cuda, args.mps)

def prepare_data(data, n_test):
    # Store datetime for later use
    datetime_series = data['datetime']
    
    # Prepare close prices and returns
    close = data.pop('close').values
    ratechg = data['price_change_pct'].values
    
    # Drop columns we don't need or that would cause data leakage
    columns_to_drop = ['price_change', 'price_change_pct', 'close_lag_1', 'close_lag_2', 
                      'close_lag_3', 'close_lag_5', 'close_lag_10', 'datetime']
    data.drop(columns=columns_to_drop, inplace=True)
    
    # Convert datetime related columns to cyclical features if not already present
    if 'hour_sin' not in data.columns:
        data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week']/7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week']/7)
        data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
        data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
    
    # Drop raw datetime components
    datetime_columns = ['hour', 'minute', 'day_of_week', 'day_of_month', 
                       'week_of_year', 'month', 'quarter', 'year']
    data.drop(columns=datetime_columns, inplace=True)
    
    # Convert boolean columns to int
    bool_columns = ['is_morning', 'is_afternoon']
    for col in bool_columns:
        if col in data.columns:
            data[col] = data[col].astype(int)
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Feature selection - keep only the most relevant features
    important_features = [
        'volume', 'vwap', 'num_trades',  # Volume and trade info
        'sma_5', 'sma_10', 'sma_20',  # Moving averages
        'ema_5', 'ema_10', 'ema_20',  # Exponential moving averages
        'bb_upper', 'bb_middle', 'bb_lower',  # Bollinger Bands
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',  # Momentum indicators
        'slowk', 'slowd',  # Stochastic
        'adx', 'cci',  # Trend indicators
        'atr', 'obv',  # Volatility and volume indicators
        'volatility_5', 'volatility_10', 'volatility_20',  # Volatility
        'return_1', 'return_5', 'return_10',  # Returns
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',  # Time features
        'month_sin', 'month_cos'
    ]
    
    numeric_data = numeric_data[important_features]
    
    # Replace infinity values with NaN
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    
    # For each column, fill NaN with the median of that column
    for column in numeric_data.columns:
        median_val = numeric_data[column].median()
        numeric_data[column] = numeric_data[column].fillna(median_val)
        
        # Clip extreme values to 3 standard deviations
        std_val = numeric_data[column].std()
        mean_val = numeric_data[column].mean()
        numeric_data[column] = numeric_data[column].clip(
            lower=mean_val - 3*std_val,
            upper=mean_val + 3*std_val
        )
    
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Split into train and test
    trainX = scaled_data[:-n_test]
    testX = scaled_data[-n_test:]
    trainy = ratechg[:-n_test]
    
    # Print feature information
    print("\nFeature information:")
    print(f"Selected features: {', '.join(important_features)}")
    print(f"Number of features: {scaled_data.shape[1]}")
    print(f"Training samples: {trainX.shape[0]}")
    print(f"Testing samples: {testX.shape[0]}")
    
    return trainX, testX, trainy, close, datetime_series

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Adjust hidden dimension based on input size
        hidden_dim = max(args.hidden, in_dim)  # Increased capacity
        self.config = MambaConfig(
            d_model=hidden_dim, 
            n_layers=args.layer,
            d_state=16  # Added state dimension
        )
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout)
        )
        
        self.mamba = Mamba(self.config)
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_dim // 2, out_dim),
            nn.Tanh()
        )
        
        self.to(device)
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x.squeeze(0))
        x = x.unsqueeze(0)
        
        # Mamba processing
        x = self.mamba(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x.flatten()

def PredictWithData(trainX, trainy, testX):
    if args.use_wandb:
        run = wandb.init(
            project="mamba-stock-prediction",
            name=f"mamba_spy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": "Mamba",
                "architecture": {
                    "input_dim": trainX.shape[1],
                    "hidden_dim": max(args.hidden, trainX.shape[1]),
                    "n_layers": args.layer,
                    "output_dim": 1,
                    "dropout": args.dropout
                },
                "hyperparameters": {
                    "learning_rate": args.lr,
                    "weight_decay": args.wd,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size
                },
                "dataset": {
                    "data_path": args.data_path,
                    "train_size": len(trainX),
                    "test_size": len(testX),
                    "n_features": trainX.shape[1]
                },
                "system": {
                    "device": str(device),
                    "cuda_available": torch.cuda.is_available(),
                    "mps_available": torch.backends.mps.is_available()
                }
            },
            tags=["spy", "mamba", str(device)]
        )
    
    clf = Net(trainX.shape[1], 1)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Calculate steps per epoch
    steps_per_epoch = len(trainX) // args.batch_size + 1
    total_steps = steps_per_epoch * args.epochs
    
    # Use ReduceLROnPlateau instead of OneCycleLR
    scheduler = ReduceLROnPlateau(
        opt,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    if args.use_wandb:
        wandb.watch(clf, log="all", log_freq=10)
    
    # Convert data to PyTorch tensors
    xt = torch.from_numpy(trainX).float().to(device)
    xv = torch.from_numpy(testX).float().to(device)
    yt = torch.from_numpy(trainy).float().to(device)
    
    # Create data loader for training
    train_dataset = torch.utils.data.TensorDataset(xt, yt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for e in range(args.epochs):
        clf.train()
        epoch_losses = []
        
        for batch_x, batch_y in train_loader:
            z = clf(batch_x.unsqueeze(0))
            
            # Check for NaN predictions
            if torch.isnan(z).any():
                print("NaN detected in predictions. Stopping training.")
                break
                
            loss = F.mse_loss(z, batch_y)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print("NaN detected in loss. Stopping training.")
                break
                
            opt.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)
            
            opt.step()
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        scheduler.step(avg_loss)  # Update learning rate based on average loss
        
        if args.use_wandb and e % 5 == 0:
            metrics = {
                "train/loss": avg_loss,
                "train/epoch": e,
                "train/learning_rate": opt.param_groups[0]['lr']
            }
            
            try:
                gradients = torch.cat([p.grad.view(-1) for p in clf.parameters() if p.grad is not None])
                metrics.update({
                    "train/gradient_mean": gradients.mean().item(),
                    "train/gradient_std": gradients.std().item(),
                    "train/gradient_norm": gradients.norm().item()
                })
            except:
                pass
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                metrics["train/best_loss"] = best_loss
            else:
                patience_counter += 1
                
            wandb.log(metrics, step=e)
            print(f'Epoch {e} | Loss: {avg_loss:.6f} | LR: {opt.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {e} epochs")
                break
    
    clf.eval()
    with torch.no_grad():
        # Process test data in batches
        test_predictions = []
        test_loader = torch.utils.data.DataLoader(
            xv, 
            batch_size=args.batch_size, 
            shuffle=False,
            pin_memory=True
        )
        
        for batch_x in test_loader:
            batch_pred = clf(batch_x.unsqueeze(0))
            test_predictions.append(batch_pred.cpu().numpy())
        
        yhat = np.concatenate(test_predictions).flatten()
        
        # Replace any NaN predictions with 0
        yhat = np.nan_to_num(yhat, nan=0.0)
    
    return yhat

# Load and prepare data
data = pd.read_csv(args.data_path)
data['datetime'] = pd.to_datetime(data['datetime'])
trainX, testX, trainy, close, time = prepare_data(data, args.n_test)

# Make predictions
predictions = PredictWithData(trainX, trainy, testX)
time = time[-args.n_test:]
data1 = close[-args.n_test:]

# Calculate predicted prices
finalpredicted_stock_price = []
pred = close[-args.n_test-1]
for i in range(args.n_test):
    pred = close[-args.n_test-1+i] * (1 + predictions[i])
    finalpredicted_stock_price.append(pred)

# Print training/testing dates
dateinf(time, args.n_test)
print('MSE RMSE MAE R2')
evaluation_metric(data1, finalpredicted_stock_price)
if args.use_wandb:
    # Enhanced final metrics logging
    metrics = {
        "test/mse": mean_squared_error(data1, finalpredicted_stock_price),
        "test/rmse": mean_squared_error(data1, finalpredicted_stock_price)**0.5,
        "test/mae": mean_absolute_error(data1, finalpredicted_stock_price),
        "test/r2": r2_score(data1, finalpredicted_stock_price)
    }
    wandb.log(metrics)
    
    # Log predictions plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, data1, label='Stock Price')
    plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')
    plt.title(f'Stock Price Prediction - SPY')
    plt.xlabel('Time', fontsize=12, verticalalignment='top')
    plt.ylabel('Close', fontsize=14, horizontalalignment='center')
    plt.legend()
    wandb.log({"predictions": wandb.Image(plt)})
    
    # Log prediction data as a table
    prediction_table = wandb.Table(data=[[t, actual, pred] 
                                       for t, actual, pred in zip(time, data1, finalpredicted_stock_price)],
                                 columns=["date", "actual", "predicted"])
    wandb.log({"predictions_table": prediction_table})
    
    plt.show()
    wandb.finish()