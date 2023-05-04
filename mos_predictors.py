# PyTorch import
import torch
import torch.nn as nn

# Quality Assessment Music Archives (QAMA)
class QAMA(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, architecture='arch1'):
        super(QAMA, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)
        self.output_layer_arch1 = nn.Linear(self.ssl_features, 1)
        self.output_layer_arch2 = nn.Linear(128, 1)

        self.architecture = architecture
        self.classification = False 
        self.output_layer_class = nn.Linear(self.ssl_features, 5)

    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        transformer_out = res['x']

        # Arch 1: just take features from the Transformer and use a linear projection
        # Arch 2: take features from every transformer layer, attach 2 FC layers, take the average of features and then predict with a linear projection layer
        if self.architecture == 'arch1':
            x = torch.mean(transformer_out, 1)
            if self.classification == False:
                x = self.output_layer_arch1(x)
            else:
                x = self.output_layer_class(x)
        elif self.architecture == 'arch2':
            transformer_layers = torch.stack([layer[2] for layer in res['layer_results']], 0).permute(2, 0, 1, 3)
            x = torch.cat((transformer_layers, transformer_out.unsqueeze(1)), 1)
            x = self.dropout(torch.relu(self.fc1(x)))
            x = torch.relu(self.fc2(x))
            x = torch.mean(torch.mean(x, 1), 1)
            x = self.output_layer_arch2(x)
        return x.squeeze(1)