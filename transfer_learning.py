import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np


# ========================
# Dataset preparation
# ========================
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels
    
class SE_block(nn.Module):
    def __init__(self, c, r=16):
        super(SE_block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c//r),
            nn.ReLU(inplace = True),
            nn.Linear(c//r, c),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
    
class SE_ResNet(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(SE_ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained = pretrained)
        self.se = SE_block(c = 512)
        self.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.se(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        """ assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads" """

        self.values = nn.Linear(self.head_dim, self.head_dim)
        self.keys = nn.Linear(self.head_dim, self.head_dim)
        self.queries = nn.Linear(self.head_dim, self.head_dim)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask = None):
        bs, c, h, w = x.size()
        x_flat = x.permute(0, 2, 3, 1)
        x_flat = x_flat.reshape(bs, h*w, c)
        query, values, keys = x_flat, x_flat, x_flat
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        out = self.norm(x_flat + out)
        out = out.reshape(bs, h, w, c).permute(0, 3, 1, 2)
        return out

class MHA_ResNet(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(MHA_ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained = pretrained)
        self.mha = MultiHeadAttention(512, 8)
        self.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.mha(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
class MHA_EfficientNet(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(MHA_EfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        self.stage_early = nn.Sequential(*list(self.efficientnet.features)[:7])
        self.mha = MultiHeadAttention(embed_size=192, heads=8)
        self.stage_late = nn.Sequential(*list(self.efficientnet.features)[7:])

        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.stage_early(x)
        x = self.mha(x)
        x = self.stage_late(x)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)

        return self.efficientnet.classifier(x)

# ========================
# build model
# ========================
def build_model(backbone="resnet18", num_classes=3, pretrained=True, att_mech = "none"):
    if backbone == "resnet18":
        if att_mech == "use_se":
            model = SE_ResNet(num_classes=num_classes, pretrained=pretrained)
        elif att_mech == "use_mha":
            model = MHA_ResNet(num_classes=num_classes, pretrained=pretrained)
        else:
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        if att_mech == "use_mha":
            model = MHA_EfficientNet(num_classes=num_classes, pretrained=pretrained)
        else:
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model


# ========================
# model training and val
# ========================
def train_one_backbone(backbone, train_csv, val_csv , test_csv, train_image_dir, val_image_dir, test_image_dir, 
                       epochs=10, batch_size=32, lr=1e-4, img_size=256, save_dir="checkpoints",pretrained_backbone=None,
                       mode = "full_fit", loss_type = "BCE", att_mech = "none"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = ""

    print(f"Using mode {mode}")
    # transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds   = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    model = build_model(backbone, num_classes=3, pretrained=False, att_mech = att_mech).to(device)

    if mode == "onsite_testing":
        epochs = 0 # to skip training
    if mode == "noFine_tuning":
        epochs = 1
        for p in model.parameters():
            p.requires_grad = False
    if mode == "freeze_backbone":
        for p in model.parameters():
            p.requires_grad = False
        if backbone == "resnet18":
            for p in model.fc.parameters():
                p.requires_grad = True
        elif backbone == "efficientnet":
            for p in model.classifier[1].parameters():
                p.requires_grad = True
    if mode == "full_fit":
        for p in model.parameters():
            p.requires_grad = True

    # loss & optimizer
    if loss_type == "focal_loss":
        criterion = nn.BCEWithLogitsLoss(reduction = "none")
    elif loss_type == "cb_loss":
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([0.59, 4, 4.88], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    if mode != "noFine_tuning":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        optimizer = None

    # training
    if mode != "onsite_testing":
        best_val_loss = float("inf")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"best_{backbone}_task3_2.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict, strict = False)
    
    for epoch in range(epochs):
        train_loss = 0
        if mode != "noFine_tuning":
            model.train()
            train_loss = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                if loss_type == "focal_loss":
                    loss = focal_loss(loss, outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)

            train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                if loss_type == "focal_loss":
                    loss = focal_loss(loss, outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{backbone}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model for {backbone} at {ckpt_path}")

    # ========================
    # testing
    # ========================
    if mode == "noFine_tuning" or mode == "onsite_testing":
        print("Testing pretrained model without fine-tuning")
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    if mode == "onsite_testing":
        image_ids = []
        predictions = []

        with torch.no_grad():
            for i, (imgs, _) in enumerate(test_loader):
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                start_idx = i * batch_size
                end_idx = start_idx + len(imgs)
                current_ids = test_ds.data.iloc[start_idx:end_idx].iloc[:, 0].tolist()
                
                image_ids.extend(current_ids)
                predictions.extend(preds)

        submission_df = pd.DataFrame(predictions, columns=["D", "G", "A"])
        submission_df.insert(0, "id", image_ids)

        output_filename = f"submission_{backbone}_task3_2.csv"
        submission_df.to_csv(output_filename, index=False)
        
        print(f"Submission file saved to {output_filename}")

    else:
        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                y_true.extend(labels.numpy())
                y_pred.extend(preds)

        y_true = torch.tensor(y_true).numpy()
        y_pred = torch.tensor(y_pred).numpy()

        disease_names = ["DR", "Glaucoma", "AMD"]

        for i, disease in enumerate(disease_names):  #compute metrics for every disease
            y_t = y_true[:, i]
            y_p = y_pred[:, i]

            acc = accuracy_score(y_t, y_p)
            precision = precision_score(y_t, y_p, average="macro",zero_division=0)
            recall = recall_score(y_t, y_p, average="macro",zero_division=0)
            f1 = f1_score(y_t, y_p, average="macro",zero_division=0)
            kappa = cohen_kappa_score(y_t, y_p)

            print(f"{disease} Results [{backbone}]")
            print(f"Accuracy : {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1-score : {f1:.4f}")
            print(f"Kappa    : {kappa:.4f}")

def focal_loss(bce, outputs, labels):
    alpha = torch.tensor([0.4, 0.7, 0.75], device=outputs.device)
    alpha_t = torch.where(labels == 1, alpha, 1-alpha)
    gamma = 2
    probabilities = torch.sigmoid(outputs)
    pt = torch.where(labels == 1, probabilities, 1-probabilities)
    loss = alpha_t * (1-pt)**gamma * bce
    return loss.mean()

def ensemble_results(weights, mode, test_csv, test_image_dir):
    print("Now doing Ensemble learning!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    probabilities = []
    backbones = ['resnet18', 'efficientnet']
    pre_backbones = ['./checkpoints/best_resnet18_task2_2.pt', './checkpoints/best_efficientnet_task2_1.pt']
    
    for i, backbone_type in enumerate(backbones):
        print(f"Running {backbone_type}")
        model = build_model(backbone_type, num_classes=3, pretrained=False).to(device)
        state_dict = torch.load(pre_backbones[i], map_location="cpu")
        model.load_state_dict(state_dict, strict = False)
        model.eval()

        y_probs = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                y_probs.append(torch.sigmoid(outputs).cpu().numpy())
        probabilities.append(np.concatenate(y_probs, axis=0))

    probs = weights[0] * probabilities[0] + weights[1] * probabilities[1]
    preds = (probs > 0.5).astype(int)
    preds = torch.tensor(preds).numpy()

    print(f"Producing outputs for {mode} testing!")

    if mode == "onsite":
        image_ids = test_ds.data.iloc[:, 0].tolist()

        submission_df = pd.DataFrame(preds, columns=["D", "G", "A"])
        submission_df.insert(0, "id", image_ids)

        output_filename = "submission_ensemble_task4.csv"
        submission_df.to_csv(output_filename, index=False)
        
        print(f"Submission file saved to {output_filename}")

    elif mode == "offsite":
        y_true = []
        for _, labels in test_loader:
            y_true.append(labels.numpy())

        y_true = np.concatenate(y_true, axis = 0)

        disease_names = ["DR", "Glaucoma", "AMD"]

        for i, disease in enumerate(disease_names):
            y_t = y_true[:, i]
            y_p = preds[:, i]

            acc = accuracy_score(y_t, y_p)
            precision = precision_score(y_t, y_p, average="macro",zero_division=0)
            recall = recall_score(y_t, y_p, average="macro",zero_division=0)
            f1 = f1_score(y_t, y_p, average="macro",zero_division=0)
            kappa = cohen_kappa_score(y_t, y_p)

            print(f"{disease} Results [Ensembling]")
            print(f"Accuracy : {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1-score : {f1:.4f}")
            print(f"Kappa    : {kappa:.4f}")


# ========================
# main
# ========================
if __name__ == "__main__":
    ensembling = False # decide whether or not you are doing ensembling (task 4)
    train_csv = "train.csv" # replace with your own train label file path
    val_csv   = "val.csv" # replace with your own validation label file path
    test_csv  = "offsite_test.csv" # replace with your own test label file path#
    train_image_dir ="./images/train"   # replace with your own train image floder path
    val_image_dir = "./images/val"  # replace with your own validation image floder path
    test_image_dir = "./images/offsite_test" # replace with your own test image floder path
    pretrained_backbone = './pretrained_backbone/ckpt_resnet18_ep50.pt'  # replace with your own pretrained backbone path
    if ensembling:
        ensemble_results([0.4, 0.6], 'offsite', test_csv, test_image_dir)
    else:
        backbone = 'resnet18'  # backbone choices: ["resnet18", "efficientnet"]
        train_one_backbone(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
                            epochs=20, batch_size=16, lr=1e-4, img_size=256, pretrained_backbone=pretrained_backbone,
                            mode = "onsite_testing", loss_type = "cb_loss", att_mech="use_mha") # mode choices: [onsite_testing, noFine_tuning, freeze_backbone, full_fit]
                                                                        # loss choices: [focal_loss, cb_loss, BCE]
                                                                        # attention mechanisms: [use_se, use_mha]
