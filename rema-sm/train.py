import argparse
import json
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import Sampler

from utils import detect_column_type, sentence_transformer_map, clean_element


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, n_samples_per_class=2):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(self.labels)
        self.label_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels_unique
        }
        self.n_samples_per_class = n_samples_per_class
        self.n_classes_per_batch = batch_size // n_samples_per_class
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self)):
            batch_indices = []
            classes = np.random.choice(
                self.labels_unique, self.n_classes_per_batch, replace=False
            )
            for label in classes:
                indices = np.random.choice(
                    self.label_to_indices[label],
                    self.n_samples_per_class,
                    replace=False,
                )
                batch_indices.extend(indices)
            # np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return len(self.labels) // self.batch_size


class SimCLRLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.sentence_embedder = model

    def forward(self, sentence_features, labels):
        embeddings = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        cosine_sim = torch.matmul(embeddings, embeddings.T)

        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()

        # Mask out the self-contrast cases (diagonal elements) without inplace operation
        diagonal_mask = torch.eye(mask.size(0), device=embeddings.device).bool()
        mask = mask * (~diagonal_mask)

        # Compute the contrastive loss
        exp_sim = torch.exp(cosine_sim / self.temperature)
        sum_exp_sim = exp_sim.sum(1, keepdim=True) - exp_sim.diag().unsqueeze(1)  # avoid self-contrast

        positive_sim = exp_sim * mask
        sum_positive_sim = positive_sim.sum(1, keepdim=True)

        log_prob = torch.log(sum_positive_sim / sum_exp_sim)
        loss = -log_prob.mean()  # Mean over the batch
        return loss


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        model_type="roberta",
        serialization="header_values_prefix",
        augmentation="exact_semantic",
    ):
        self.serialization = serialization
        self.tokenizer = AutoTokenizer.from_pretrained(
            sentence_transformer_map[model_type]
        )
        self.labels = []
        self.items = self._initialize_items(data, augmentation)

    def _initialize_items(self, data, augmentation):
        items = []
        class_id = 0

        for _, categories in data.items():
            for aug_type, columns in categories.items():
                if aug_type in augmentation or aug_type == "original":
                    for column_name, values in columns.items():
                        processed_column_name = (
                            column_name.rsplit("_", 1)[0]
                            if aug_type == "exact"
                            else column_name
                        )
                        # processed_column_name = clean_element(processed_column_name)
                        values = [clean_element(str(value)) for value in values]
                        items.append((processed_column_name, values, class_id))
                        self.labels.append(class_id)
            class_id += 1

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        key, values, class_id = self.items[idx]
        text = self._serialize(key, values)
        return text, class_id

    def _serialize(self, header, values):
        if values:
            col = pd.DataFrame({header: values})[header]
            data_type = detect_column_type(pd.DataFrame({header: values})[header])
        else:
            data_type = "unknown"
        serialization = {
            "header_values_default": f"{self.tokenizer.cls_token}{header}{self.tokenizer.sep_token}{data_type}{self.tokenizer.sep_token}{','.join(map(str, values))}",
            "header_values_prefix": f"{self.tokenizer.cls_token}header:{header}{self.tokenizer.sep_token}datatype:{data_type}{self.tokenizer.sep_token}values:{', '.join(map(str, values))}",
        }
        return serialization[self.serialization]


def evaluate_top_k(model, validation_loader, device, k=1):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in validation_loader:
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            embeddings = model.encode(texts, convert_to_tensor=True, device=device)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    normalized_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

    correct_matches = 0
    total_matches = 0

    for i in range(len(all_labels)):
        similarity = similarity_matrix[i]
        similarity[i] = -1  # Set self-similarity to negative value to exclude it
        top_k_indices = torch.topk(similarity, k).indices
        correct_matches += 1 if all_labels[i] in all_labels[top_k_indices] else 0
        total_matches += 1

    accuracy = correct_matches / total_matches if total_matches > 0 else 0
    return accuracy


def evaluate_recall_at_ground_truth(model, validation_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in validation_loader:
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            embeddings = model.encode(texts, convert_to_tensor=True, device=device)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    normalized_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

    correct_matches = 0
    total_matches = 0

    # Evaluate recall at ground truth
    for i in range(len(all_labels)):
        true_label = all_labels[i]
        k = torch.sum(all_labels == true_label).item() - 1

        similarity = similarity_matrix[i]
        similarity[i] = -1
        top_k_indices = torch.topk(similarity, k).indices

        correct_matches += torch.sum(all_labels[top_k_indices] == true_label).item()
        total_matches += k

    recall_at_ground_truth = correct_matches / total_matches if total_matches > 0 else 0
    return recall_at_ground_truth


def evaluate_metrics(model, validation_loader, device, fixed_k=1):
    model.eval()
    all_embeddings = []
    all_labels = []

    # Collect embeddings and labels
    with torch.no_grad():
        for texts, labels in validation_loader:
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            embeddings = model.encode(texts, convert_to_tensor=True, device=device)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    normalized_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

    correct_matches_fixed_k = 0
    total_matches_fixed_k = 0
    correct_matches_recall = 0
    total_matches_recall = 0

    for i in range(len(all_labels)):
        similarity = similarity_matrix[i]
        similarity[i] = -1  # Set self-similarity to a negative value to exclude it

        # Fixed k evaluation
        top_k_indices_fixed = torch.topk(similarity, fixed_k).indices
        correct_matches_fixed_k += (
            1 if all_labels[i] in all_labels[top_k_indices_fixed] else 0
        )
        total_matches_fixed_k += 1

        # Recall at ground truth evaluation
        true_label = all_labels[i]
        ground_truth_k = torch.sum(all_labels == true_label).item() - 1
        top_k_indices_recall = torch.topk(similarity, ground_truth_k).indices
        correct_matches_recall += torch.sum(
            all_labels[top_k_indices_recall] == true_label
        ).item()
        total_matches_recall += ground_truth_k

    accuracy = (
        correct_matches_fixed_k / total_matches_fixed_k
        if total_matches_fixed_k > 0
        else 0
    )
    recall_at_ground_truth = (
        correct_matches_recall / total_matches_recall if total_matches_recall > 0 else 0
    )

    return accuracy, recall_at_ground_truth


def train_model(
    model,
    num_classes,
    data_loader,
    optimizer,
    model_path,
    loss_type="triplet",
    margin=1,
    epochs=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    best_accuracy = 0
    if loss_type == "triplet":
        loss_fn = losses.BatchHardTripletLoss(
            model=model,
            margin=margin,
            distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
        )
    elif loss_type == "simclr":
        loss_fn = SimCLRLoss(
            model=model,
            temperature=0.5,
        )

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            texts, labels = batch
            labels = torch.tensor(labels, dtype=torch.float, device=device)

            optimizer.zero_grad()

            sentence_features = model.tokenize(texts)
            sentence_features = [
                {k: v.to(device) for k, v in sentence_features.items()}
            ]
            loss = loss_fn(sentence_features, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        # validation_accuracy = evaluate_recall_at_ground_truth(model, data_loader, device)
        # validation_accuracy = evaluate_top_k(model, data_loader, device, k=1)
        accuracy, recall_at_ground_truth = evaluate_metrics(
            model, data_loader, device, fixed_k=1
        )
        # print(f"Epoch {epoch+1}, Loss: {avg_loss}, Val Accuracy: {validation_accuracy}")
        print(
            f"Epoch {epoch+1}, Loss: {avg_loss}, Val Accuracy: {accuracy}, Recall at Ground Truth: {recall_at_ground_truth}"
        )
        validation_accuracy = (accuracy + recall_at_ground_truth) / 2
        # Save best model
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with accuracy: {best_accuracy}")

    print("Training complete with best accuracy:", best_accuracy)


def main():
    parser = argparse.ArgumentParser(
        description="Match columns between source and target tables using pretrained models."
    )
    parser.add_argument(
        "--dataset",
        default="gdc",
        help="Name of the dataset for model customization",
    )
    parser.add_argument(
        "--model_type",
        default="mpnet",
        help="Type of model (roberta, distilbert, mpnet)",
    )
    parser.add_argument(
        "--serialization",
        default="header_values_prefix",
        help="Column serialization method (header_values_default, header_values_prefix)",
    )
    parser.add_argument(
        "--augmentation",
        default="exact_semantic",
        help="Augmentation type (exact, semantic, exact_semantic)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--loss_type",
        default="triplet",
        choices=["triplet", "simclr"],
        help="Type of loss function to use (triplet or simclr)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Margin value for triplet loss",
    )

    args = parser.parse_args()

    file_path = f"{args.dataset}_synthetic_matches.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    dataset = CustomDataset(
        data,
        model_type=args.model_type,
        serialization="header_values_prefix",
        augmentation="exact_semantic",
    )

    labels = dataset.labels
    n_classes = len(np.unique(labels))
    balanced_sampler = BalancedBatchSampler(
        labels, batch_size=args.batch_size, n_samples_per_class=2
    )
    data_loader = DataLoader(
        dataset,
        batch_sampler=balanced_sampler,
        collate_fn=lambda x: ([d[0] for d in x], [d[1] for d in x]),
    )

    model = SentenceTransformer(sentence_transformer_map[args.model_type])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    loss_type = args.loss_type
    model_path = (
        f"models/{args.model_type}-{args.dataset}-{args.serialization}-{args.augmentation}-{str(args.batch_size)}-{str(args.margin)}.pth"
        if loss_type == "triplet"
        else f"models/{args.model_type}-{args.dataset}-{args.serialization}-{args.augmentation}-simclr.pth"
    )
    print(f"Training model using {args.model_type} on {args.dataset} dataset")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    train_model(
        model,
        n_classes,
        data_loader,
        optimizer,
        model_path,
        loss_type=loss_type,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
