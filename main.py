
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

video_path = "data"

def extract_frames(video_path, size=(224,224)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.resize(frame, size)
        frames.append(frame)

    cap.release()
    return frames

# 
def sample_frames(frames, num_frames=16):
    total = len(frames)

    if total == 0:
        return None

    indices = np.linspace(0, total-1, num_frames).astype(int)
    sample_frames = [frames[i] for i in indices]

    return sample_frames


def process_video(video_path):
    frames = extract_frames(video_path)

    if len(frames) == 0:
        return None

    frames = sample_frames(frames, 16)

    return frames


class videoDataset(Dataset):
    def __init__(self, data_dir, num_frames=16):
        self.data_dir = data_dir
        self.num_frames = num_frames

        self.classes = os.listdir(data_dir)
        self.video_paths = []
        self.labels = []

        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(data_dir, cls)

            for video in os.listdir(cls_path):
                video_path = os.path.join(cls_path, video)

                self.video_paths.append(video_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]

        frames = self.process_video(video_path)

        frames = np.array(frames)
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float()

        return frames, label

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return None

        indices = np.linspace(0, len(frames)-1, self.num_frames).astype(int)
        frames = [frames[i] for i in indices]

        return frames


def create_dataloader(data_dir, batch_size=4):

    dataset = videoDataset(data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    return dataloader

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM_Model, self).__init__()

        self.cnn = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        B, T, C, H, W = x.shape  

        x = x.view(B * T, C, H, W)

        features = self.cnn(x)   
        features = features.view(B, T, -1)  

        lstm_out, _ = self.lstm(features)  

        out = torch.mean(lstm_out, dim=1)  # (B, 256)

        out = self.dropout(out)
        out = self.fc(out)  

        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10

model = CNN_LSTM_Model(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)



num_epochs = 2
dataloader = create_dataloader("data")

for epoch in range(num_epochs):
    model.train()

    total_loss = 0

    for batch_idx, (videos, labels) in enumerate(dataloader):
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


correct = 0
total = 0

all_preds = []
all_labels = []

model.eval()

with torch.no_grad():
    for videos, labels in dataloader:
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)

        # get predicted class
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

