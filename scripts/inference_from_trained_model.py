from hippie.dataloading import EphysDatasetLabeled
from hippie.model import hippieUnimodalCVAE, hippieUnimodalEmbeddingModelCVAE
from utils import get_embeddings

import torch
import pandas as pd
import argparse
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def perform_knn_classification(embeddings, labels, label_names, dataset_name, embedding_type, output_dir, test_size=0.2):
    """
    Perform KNN classification on embeddings and save predictions.
    
    Args:
        embeddings: numpy array of embeddings
        labels: numpy array of true labels
        label_names: list of label names
        dataset_name: name of the dataset
        embedding_type: type of embedding ('waveform', 'isi', 'joint')
        output_dir: directory to save results
        test_size: proportion of data to use for testing
    
    Returns:
        dict with results including best accuracy, best neighbors, and predictions
    """
    # Skip if we only have one class
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        print(f"  - Skipping {embedding_type} KNN: only one class found")
        return None
    
    # Check if we can do stratified splitting
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples_per_class = np.min(counts)
    
    # If any class has only 1 sample, we can't do stratified splitting
    if min_samples_per_class < 2:
        print(f"  - Warning: Some classes have only 1 sample. Using random split instead of stratified.")
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42, stratify=None
        )
    else:
        # Use stratified splitting
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42, stratify=labels
        )
    
    # Test different numbers of neighbors
    neighbor_options = list(range(5, min(20, len(X_train))))  # Don't exceed training samples
    if len(neighbor_options) == 0:
        neighbor_options = [min(3, len(X_train) - 1)]  # Fallback for very small datasets
    
    bal_accuracies = []
    
    for n_neighbors in neighbor_options:
        if n_neighbors >= len(X_train):
            continue
            
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        bal_accuracies.append(bal_acc)
    
    if len(bal_accuracies) == 0:
        print(f"  - Skipping {embedding_type} KNN: insufficient data")
        return None
    
    # Find best number of neighbors
    best_idx = np.argmax(bal_accuracies)
    best_neighbors = neighbor_options[best_idx]
    best_accuracy = bal_accuracies[best_idx]
    
    # Train final model with best parameters
    knn = KNeighborsClassifier(n_neighbors=best_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels)
    
    # Convert labels back to original names if available
    if label_names is not None:
        y_test_names = [label_names[int(i)] for i in y_test]
        y_pred_names = [label_names[int(i)] for i in y_pred]
    else:
        y_test_names = y_test.astype(str)
        y_pred_names = y_pred.astype(str)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true': y_test_names,
        'pred': y_pred_names,
        'true_encoded': y_test,
        'pred_encoded': y_pred
    })
    
    predictions_path = os.path.join(output_dir, f"{dataset_name}_{embedding_type}_knn_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save performance summary
    summary_df = pd.DataFrame({
        'embedding_type': [embedding_type],
        'best_neighbors': [best_neighbors],
        'balanced_accuracy': [best_accuracy],
        'n_train_samples': [len(X_train)],
        'n_test_samples': [len(X_test)],
        'n_classes': [len(unique_labels)]
    })
    
    summary_path = os.path.join(output_dir, f"{dataset_name}_{embedding_type}_knn_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"  - {embedding_type.title()} KNN: {best_accuracy:.3f} accuracy with {best_neighbors} neighbors")
    print(f"  - Saved predictions to {predictions_path}")
    print(f"  - Saved summary to {summary_path}")
    
    return {
        'embedding_type': embedding_type,
        'best_neighbors': best_neighbors,
        'balanced_accuracy': best_accuracy,
        'confusion_matrix': conf_matrix,
        'predictions': predictions_df,
        'neighbor_options': neighbor_options,
        'all_accuracies': bal_accuracies
    }


class EphysDatasetWithSourceLabels(Dataset):
    """Custom dataset that provides both class and source labels for inference."""
    def __init__(self, waveforms, isi_dists, class_labels, source_label=0, mode="wave", normalize=True):
        self.waveforms = np.array(waveforms)
        self.isi_dists = np.array(isi_dists)
        self.class_labels = np.array(class_labels)
        self.source_label = source_label  # Single source label for all samples
        assert mode in ("wave", "time")
        self.mode = mode
        assert len(self.waveforms) == len(self.isi_dists)
        assert len(self.waveforms) == len(self.class_labels)
        self.normalize = normalize

    def __getitem__(self, idx):
        waveform = torch.as_tensor(self.waveforms[idx, ...]).float()
        isi_dist = torch.as_tensor(self.isi_dists[idx, ...]).float()
        isi_dist = torch.log(isi_dist + 1)

        class_label = torch.as_tensor(self.class_labels[idx]).long()
        source_label = torch.as_tensor(self.source_label).long()
        
        # Combine class and source labels into a 2D tensor
        labels = torch.stack([class_label, source_label])

        if self.normalize:
            min_val = np.min(waveform)
            max_val = np.max(waveform)
            waveform = (waveform - min_val) / (max_val - min_val)
            waveform = waveform * 2 - 1
            isi_dist = (isi_dist - isi_dist.mean()) / isi_dist.std()

        waveform = waveform.view(1, 1, -1)
        waveform = F.interpolate(waveform, size=(50,), mode="linear").view(1, -1)

        isi_dist = isi_dist.view(1, 1, -1)
        isi_dist = F.interpolate(isi_dist, size=(100,), mode="linear").view(1, -1)

        if self.mode == "wave":
            return waveform, labels
        elif self.mode == "time":
            return isi_dist, labels

    def __len__(self):
        return len(self.waveforms)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--z_dim",
    type=int,
    default=10,
    required=False,
    help="Dimensionality of the latent space"
)
parser.add_argument(
    '--dataset',
    type=str,
    default="cellexplorer-celltype",
    help="Dataset to perform inference on"
)
parser.add_argument(
    '--wave-checkpoint',
    type=str,
    required=True,
    help="Path to the waveform model checkpoint"
)
parser.add_argument(
    '--time-checkpoint',
    type=str,
    required=True,
    help="Path to the time model checkpoint"
)
parser.add_argument(
    '--output-dir',
    type=str,
    default="./embeddings_results",
    help="Directory to save embeddings and visualizations"
)
parser.add_argument(
    '--max-labels',
    type=int,
    default=12,
    help="Maximum number of labels to display in UMAP plots (others grouped as 'Other')"
)
parser.add_argument(
    '--min-frequency',
    type=int,
    default=10,
    help="Minimum frequency for a label to be displayed individually"
)
parser.add_argument(
    '--test-size',
    type=float,
    default=0.2,
    help="Proportion of data to use for testing in KNN classification (default: 0.2)"
)

args = parser.parse_args()
accelerator = "gpu" if torch.cuda.is_available() else "cpu"


# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)

# Load dataset
print(f"Loading dataset: {args.dataset}")
wf = pd.read_csv(f"datasets/{args.dataset}/waveforms.csv")
wf = wf.dropna(axis=1)
isi = pd.read_csv(f"datasets/{args.dataset}/isi_dist.csv")
isi = isi.dropna(axis=1)

wf = wf.to_numpy()
isi = isi.to_numpy()

# Load metadata if available
labels = None
label_names = None
if os.path.exists(f"datasets/{args.dataset}/labels.csv"):
    metadata = pd.read_csv(f"datasets/{args.dataset}/labels.csv")
    if 'label' in metadata.columns:
        labels = metadata['label'].astype('category').cat.codes.to_numpy()
        label_names = metadata['label'].unique()
        print(f"Found {len(label_names)} unique labels: {label_names}")

# If no labels in metadata, create dummy labels (for dataset embedding)
if labels is None:
    labels = np.zeros(wf.shape[0])
    label_names = ["unknown"]
    print("No labels found, using dummy labels")

# Create datasets with source labels (using source_label=0 as dummy)
dataset_wave = EphysDatasetWithSourceLabels(wf, isi, labels, source_label=0, mode="wave", normalize=False)
dataset_time = EphysDatasetWithSourceLabels(wf, isi, labels, source_label=0, mode="time", normalize=False)

# Create data loaders
data_loader_wave = torch.utils.data.DataLoader(
    dataset_wave, batch_size=128, shuffle=False
)
data_loader_time = torch.utils.data.DataLoader(
    dataset_time, batch_size=128, shuffle=False
)

# Load model weights first to inspect architecture
try:
    wave_checkpoint = torch.load(args.wave_checkpoint, map_location=torch.device(accelerator))
    time_checkpoint = torch.load(args.time_checkpoint, map_location=torch.device(accelerator))
    print("Checkpoints loaded successfully")
except Exception as e:
    print(f"Error loading checkpoints: {e}")
    exit(1)

# Extract architecture parameters from checkpoint
def extract_model_params_from_checkpoint(checkpoint):
    """Extract model architecture parameters from checkpoint state_dict"""
    state_dict = checkpoint["state_dict"]
    
    # Extract dims from checkpoint
    z_dim = state_dict["model.z_mean.weight"].shape[0]
    class_hidden_dim = state_dict["model.source_embedding.weight"].shape[1]
    num_classes = state_dict["model.class_embedding.weight"].shape[0]
    num_sources = state_dict["model.source_embedding.weight"].shape[0]
    
    # Extract output_size from decoder - use linear_out instead of linear
    if "model.decoder.linear_out.weight" in state_dict:
        output_size = state_dict["model.decoder.linear_out.weight"].shape[0]
    elif "model.decoder.linear.weight" in state_dict:
        output_size = state_dict["model.decoder.linear.weight"].shape[0]
    else:
        output_size = 50
        print("Warning: decoder linear layer not found in checkpoint, using default 50")
    
    return z_dim, class_hidden_dim, output_size, num_classes, num_sources

# Extract parameters for both models
wave_z_dim, wave_class_hidden_dim, wave_output_size, wave_num_classes, wave_num_sources = extract_model_params_from_checkpoint(wave_checkpoint)
time_z_dim, time_class_hidden_dim, time_output_size, time_num_classes, time_num_sources = extract_model_params_from_checkpoint(time_checkpoint)

print(f"Wave model params: z_dim={wave_z_dim}, class_hidden_dim={wave_class_hidden_dim}, output_size={wave_output_size}, num_classes={wave_num_classes}, num_sources={wave_num_sources}")
print(f"Time model params: z_dim={time_z_dim}, class_hidden_dim={time_class_hidden_dim}, output_size={time_output_size}, num_classes={time_num_classes}, num_sources={time_num_sources}")

# Use the actual number of sources and classes from the checkpoints
num_classes = len(np.unique(labels))

# Create models with correct architecture
print("Loading models from checkpoints...")
wave_model = hippieUnimodalCVAE(
    z_dim=wave_z_dim, 
    output_size=wave_output_size, 
    class_hidden_dim=wave_class_hidden_dim, 
    num_sources=wave_num_sources, 
    num_classes=num_classes
)
time_model = hippieUnimodalCVAE(
    z_dim=time_z_dim, 
    output_size=time_output_size, 
    class_hidden_dim=time_class_hidden_dim, 
    num_sources=time_num_sources, 
    num_classes=num_classes
)

wave_model = hippieUnimodalEmbeddingModelCVAE(wave_model)
time_model = hippieUnimodalEmbeddingModelCVAE(time_model)

# Load model weights
try:
    # Handle potential class_embedding mismatch
    if "model.class_embedding.weight" in wave_checkpoint["state_dict"]:
        if wave_checkpoint["state_dict"]["model.class_embedding.weight"].size(0) != num_classes:
            print("Warning: Class embedding size mismatch in wave model. Removing from checkpoint.")
            wave_checkpoint["state_dict"].pop("model.class_embedding.weight")
            
    if "model.class_embedding.weight" in time_checkpoint["state_dict"]:
        if time_checkpoint["state_dict"]["model.class_embedding.weight"].size(0) != num_classes:
            print("Warning: Class embedding size mismatch in time model. Removing from checkpoint.")
            time_checkpoint["state_dict"].pop("model.class_embedding.weight")
    
    wave_model.load_state_dict(wave_checkpoint["state_dict"], strict=False)
    time_model.load_state_dict(time_checkpoint["state_dict"], strict=False)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Set models to evaluation mode
wave_model.eval()
time_model.eval()

# Extract embeddings
print("Extracting embeddings...")
with torch.no_grad():
    waveform_embeddings, isi_embeddings, joint_embeddings = get_embeddings(
        data_loader_wave, data_loader_time, wave_model, time_model
    )

# Save embeddings
print("Saving embeddings...")
embedding_data = {
    'waveform': waveform_embeddings,
    'isi': isi_embeddings,
    'joint': joint_embeddings,
    'labels': labels
}

for name, embeddings in zip(['waveform', 'isi', 'joint'], 
                           [waveform_embeddings, isi_embeddings, joint_embeddings]):
    df = pd.DataFrame(embeddings)
    if labels is not None:
        df['label'] = labels
        if label_names is not None:
            df['label_name'] = pd.Categorical([label_names[i.astype(int)] for i in labels])
    
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{name}_embeddings.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {name} embeddings to {output_path}")

# Perform KNN classification if we have labels
if labels is not None and len(np.unique(labels)) > 1:
    print("\nPerforming KNN classification...")
    
    knn_results = []
    
    # Perform KNN for each embedding type
    for name, embeddings in zip(['waveform', 'isi', 'joint'], 
                               [waveform_embeddings, isi_embeddings, joint_embeddings]):
        print(f"Running KNN classification for {name} embeddings...")
        result = perform_knn_classification(
            embeddings, labels, label_names, args.dataset, name, args.output_dir, args.test_size
        )
        if result is not None:
            knn_results.append(result)
    
    # Create overall summary
    if knn_results:
        print(f"\nKNN Classification Results Summary:")
        print("=" * 50)
        
        summary_data = []
        for result in knn_results:
            summary_data.append({
                'embedding_type': result['embedding_type'],
                'balanced_accuracy': result['balanced_accuracy'],
                'best_neighbors': result['best_neighbors']
            })
            print(f"{result['embedding_type'].title():>10}: {result['balanced_accuracy']:.3f} "
                  f"(k={result['best_neighbors']})")
        
        # Save overall summary
        overall_summary_df = pd.DataFrame(summary_data)
        overall_summary_path = os.path.join(args.output_dir, f"{args.dataset}_knn_overall_summary.csv")
        overall_summary_df.to_csv(overall_summary_path, index=False)
        print(f"\nSaved overall KNN summary to {overall_summary_path}")
        
        # Find best performing embedding type
        best_result = max(knn_results, key=lambda x: x['balanced_accuracy'])
        print(f"\nBest performing embedding: {best_result['embedding_type']} "
              f"({best_result['balanced_accuracy']:.3f} accuracy)")
    
    print("\nKNN classification completed!")
else:
    print("\nSkipping KNN classification: insufficient labels or only one class found")

# Generate UMAP visualizations
print("Generating UMAP visualizations...")

def create_improved_umap_plot(embeddings, labels, label_names, title, output_path, 
                             max_labels=12, min_frequency=10):
    """Create an improved UMAP visualization with better label handling."""
    reducer = umap.UMAP(random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # Handle single label case
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                   alpha=0.7, s=15, color='steelblue')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Count label frequencies
    label_counts = pd.Series(labels).value_counts()
    
    # Determine which labels to show individually
    if len(unique_labels) <= max_labels:
        # Show all labels if we have few enough
        labels_to_show = unique_labels
        plot_labels = labels.copy()
        plot_label_names = [label_names[int(i)] if label_names is not None else f"Label {int(i)}" 
                           for i in labels_to_show]
    else:
        # Show top frequent labels, group others
        frequent_labels = label_counts[label_counts >= min_frequency].head(max_labels - 1).index
        
        # Create new label array
        plot_labels = labels.copy()
        other_mask = ~np.isin(labels, frequent_labels)
        plot_labels[other_mask] = -1  # Use -1 for "Other"
        
        # Create label names
        plot_label_names = []
        labels_to_show = list(frequent_labels) + [-1]
        
        for label in frequent_labels:
            if label_names is not None:
                name = label_names[int(label)]
            else:
                name = f"Label {int(label)}"
            plot_label_names.append(f"{name} (n={label_counts[label]})")
        
        # Add "Other" category
        other_count = np.sum(other_mask)
        plot_label_names.append(f"Other (n={other_count})")
    
    # Choose color palette based on number of categories
    n_categories = len(labels_to_show)
    if n_categories <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_categories]
    elif n_categories <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_categories]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_categories))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each category
    for i, label in enumerate(labels_to_show):
        mask = plot_labels == label
        if np.any(mask):
            ax.scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                      c=[colors[i]], label=plot_label_names[i], 
                      alpha=0.7, s=15, edgecolors='none')
    
    # Customize the plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    
    # Add legend with better positioning
    if n_categories <= 15:
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          frameon=True, fancybox=True, shadow=True)
    else:
        # For many categories, use a more compact legend
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          frameon=True, fancybox=True, shadow=True,
                          ncol=2 if n_categories > 20 else 1)
    
    # Improve legend appearance
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"  - Plotted {n_categories} categories")
    if len(unique_labels) > max_labels:
        print(f"  - Grouped {np.sum(other_mask)} samples into 'Other' category")

def create_umap_plot(embeddings, labels, title, output_path):
    """Legacy function for backward compatibility."""
    create_improved_umap_plot(embeddings, labels, None, title, output_path)

# Generate UMAP plots for each embedding type
for name, embeddings in zip(['waveform', 'isi', 'joint'], 
                           [waveform_embeddings, isi_embeddings, joint_embeddings]):
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{name}_umap.png")
    print(f"Creating {name} UMAP visualization...")
    create_improved_umap_plot(embeddings, labels, label_names, 
                             f"{args.dataset.replace('-', ' ').title()} - {name.title()} Embeddings", 
                             output_path, max_labels=args.max_labels, 
                             min_frequency=args.min_frequency)
    print(f"Saved {name} UMAP visualization to {output_path}")

# Optional: Generate improved paired comparisons between modalities
if labels is not None and len(np.unique(labels)) > 1:
    print("Generating improved comparison plots...")
    
    # Prepare label data for consistent visualization across subplots
    unique_labels = np.unique(labels)
    label_counts = pd.Series(labels).value_counts()
    
    # Determine which labels to show (same logic as individual plots)
    if len(unique_labels) <= args.max_labels:
        labels_to_show = unique_labels
        plot_labels = labels.copy()
        plot_label_names = [label_names[int(i)] if label_names is not None else f"Label {int(i)}" 
                           for i in labels_to_show]
    else:
        frequent_labels = label_counts[label_counts >= args.min_frequency].head(args.max_labels - 1).index
        plot_labels = labels.copy()
        other_mask = ~np.isin(labels, frequent_labels)
        plot_labels[other_mask] = -1
        
        plot_label_names = []
        labels_to_show = list(frequent_labels) + [-1]
        
        for label in frequent_labels:
            if label_names is not None:
                name = label_names[int(label)]
            else:
                name = f"Label {int(label)}"
            plot_label_names.append(f"{name}")
        
        other_count = np.sum(other_mask)
        plot_label_names.append(f"Other")
    
    # Choose consistent colors
    n_categories = len(labels_to_show)
    if n_categories <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_categories]
    elif n_categories <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_categories]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_categories))
    
    # Create the comparison figure
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (name, embeddings) in enumerate(zip(['waveform', 'isi', 'joint'], 
                                               [waveform_embeddings, isi_embeddings, joint_embeddings])):
        reducer = umap.UMAP(random_state=42)
        umap_embeddings = reducer.fit_transform(embeddings)
        
        # Plot each category with consistent colors
        for i, label in enumerate(labels_to_show):
            mask = plot_labels == label
            if np.any(mask):
                axs[idx].scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                               c=[colors[i]], label=plot_label_names[i] if idx == 0 else "", 
                               alpha=0.7, s=12, edgecolors='none')
        
        axs[idx].set_title(f"{name.title()} Embeddings", fontsize=12, fontweight='bold')
        axs[idx].set_xlabel('UMAP 1', fontsize=10)
        axs[idx].set_ylabel('UMAP 2', fontsize=10)
    
    # Add a single legend for all subplots
    if n_categories <= 15:
        fig.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', 
                  frameon=True, fancybox=True, shadow=True)
    else:
        fig.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', 
                  frameon=True, fancybox=True, shadow=True, ncol=2)
    
    plt.suptitle(f"{args.dataset.replace('-', ' ').title()} - Modality Comparison", 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(args.output_dir, f"{args.dataset}_comparison_umap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved improved comparison visualization to {output_path}")
    print(f"  - Comparison plot shows {n_categories} categories across all modalities")

print("Inference completed successfully!")
