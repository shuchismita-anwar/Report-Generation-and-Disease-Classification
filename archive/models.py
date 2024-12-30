import torch
import torch.nn as nn

class ClinicalDataEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(ClinicalDataEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.fc(x)

class DiseaseClassifier(nn.Module):
    def __init__(self, input_dim=512, clinical_dim=128, num_classes=8):
        super(DiseaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim + clinical_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, clinical_data):
        if x.dim() == 3:
            x = x.squeeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        
        if clinical_data.dim() == 3:
            clinical_data = clinical_data.squeeze(0)
        elif clinical_data.dim() == 1:
            clinical_data = clinical_data.unsqueeze(0)
        
        if x.size(0) != clinical_data.size(0):
            if x.size(0) == 1:
                x = x.expand(clinical_data.size(0), -1)
            elif clinical_data.size(0) == 1:
                clinical_data = clinical_data.expand(x.size(0), -1)
        
        x = torch.cat([x, clinical_data], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
