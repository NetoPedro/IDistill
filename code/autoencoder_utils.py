# Imports
from collections import OrderedDict

# PyTorch Imports
import torch
import torch.nn as nn
from torchinfo import summary



# Class: UNetAutoencoder
class UNetAutoencoder(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=32, img_size=224, embedding_size=256):
        super(UNetAutoencoder, self).__init__()

        # Encoder
        features = init_features
        self.encoder1 = UNetAutoencoder._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetAutoencoder._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetAutoencoder._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetAutoencoder._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Encoder bottleneck
        self.bottleneck_pre_emb = UNetAutoencoder._block(features * 8, features * 16, name="bottleneck_pre_emb")


        
        # Get a fixed size embedding
        in_features_ = torch.rand(1, 3, img_size, img_size)
        in_features_ = self.encoder1(in_features_)
        in_features_ = self.encoder2(self.pool1(in_features_))
        in_features_ = self.encoder3(self.pool2(in_features_))
        in_features_ = self.encoder4(self.pool3(in_features_))
        in_features_ = self.bottleneck_pre_emb(self.pool4(in_features_))
        
        # Pre-save the reshape size for the forward method
        self.reshape_size = [in_features_.shape[0], in_features_.shape[1], in_features_.shape[2], in_features_.shape[3]]
        in_features_ = in_features_.shape[0] * in_features_.shape[1] * in_features_.shape[2] * in_features_.shape[3]

        # Create the embedding layer
        self.embedding_layer = nn.Linear(in_features=in_features_, out_features=embedding_size, bias=False)

        # Reshape it again
        self.recons_embedding = nn.Linear(in_features=embedding_size, out_features=in_features_, bias=False)

        # Create the bottleneck for the decoder
        self.bottleneck_post_emb = UNetAutoencoder._block(features * 16, features * 16, name="bottleneck_post_emb")

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNetAutoencoder._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNetAutoencoder._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNetAutoencoder._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNetAutoencoder._block(features * 2, features, name="dec1")

        # Last convolutional layer
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        return


    def forward(self, x):

        # Go with the encoding process according to U-Net structure
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Apply the first bottleneck
        bottleneck = self.bottleneck_pre_emb(self.pool4(enc4))

        # Flatten the bottleneck
        bottleneck = torch.reshape(bottleneck, (-1, bottleneck.size(1) * bottleneck.size(2) * bottleneck.size(3)))

        # Retrieve the embedding
        embedding = self.embedding_layer(bottleneck)

        # Reconstruct the embedding
        recons_embedding = self.recons_embedding(embedding)

        # Reshape reconstructed embedding
        recons_embedding = torch.reshape(recons_embedding, (-1, self.reshape_size[1], self.reshape_size[2], self.reshape_size[3]))

        # Apply the second bottleneck
        bottleneck = self.bottleneck_post_emb(recons_embedding)

        # Go with the decoding process according to U-Net structure
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1)), embedding

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
