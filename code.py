import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision.utils import make_grid
from torch_snippets import show 
import numpy as np

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Metal Performance Shaders (MPS) is available. Training on Apple Silicon GPU.")
    device = torch.device("mps")
else:
    print("CUDA and MPS are not available. Training on CPU.")
    device = torch.device("cpu")


# Data loading and preprocessing

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = MNIST('~/data', train=True, download=True, transform=transform)
data_loader = DataLoader(train_dataset,batch_size=128, shuffle=True)

"""CGAN with MLP"""
num_class = 10
embedding_dim = 10
latent_dim = 100
img_shape = 28*28

# CGAN MLP Generator and Discriminator
class CGAN_gen(nn.Module):
    def __init__(self):
        super().__init__()

        self.labelembedding = nn.Embedding(num_class, embedding_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, img_shape),
            nn.Tanh()

        )
    def forward(self, noise, labels):
        c = self.labelembedding(labels)
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        return img

class CGAN_disc(nn.Module):
    def __init__(self):
        super().__init__()

        self.labelembedding = nn.Embedding(num_class, embedding_dim)

        self.model = nn.Sequential(
            nn.Linear(img_shape + embedding_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img, labels):
        c = self.labelembedding(labels)
        x = torch.cat([img, c], 1)
        validity = self.model(x)
        return validity
    
generator_GAN = CGAN_gen().to(device)
discriminator_GAN = CGAN_disc().to(device)

# Parameter Count 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

vanilla_gen_params = count_parameters(generator_GAN)
vanilla_disc_params = count_parameters(discriminator_GAN)

print(f"Vanilla Generator Parameters: {vanilla_gen_params:,}")
print(f"Vanilla Discriminator Parameters: {vanilla_disc_params:,}")

# Optimizers and loss function
d_optimizer = optim.Adam(discriminator_GAN.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator_GAN.parameters(), lr=0.0002)
loss = nn.BCELoss()

"""Training steps for CGAN with MLP"""
def discriminator_train_step(real_data, real_labels, fake_data, fake_labels):
    vec_ones = torch.ones(len(real_data), 1).to(device)
    vec_zeros = torch.zeros(len(real_data), 1).to(device)

    discriminator_GAN.zero_grad()

    prediction_real = discriminator_GAN(real_data, real_labels)
    error_real = loss(prediction_real, vec_ones)
    error_real.backward()

    prediction_fake = discriminator_GAN(fake_data, fake_labels)
    error_fake = loss(prediction_fake, vec_zeros)
    error_fake.backward()

    d_optimizer.step()
    return error_real + error_fake

"""Training steps for CGAN with MLP"""
def generator_train_step(fake_data, fake_labels):
    vec_ones = torch.ones(len(fake_data), 1).to(device)

    g_optimizer.zero_grad()

    prediction = discriminator_GAN(fake_data, fake_labels)
    error = loss(prediction, vec_ones)
    error.backward()

    g_optimizer.step()
    return error

"""Sample generation for CGAN with MLP"""
def plot_samples():
  z = torch.randn(64, 100).to(device)
  ordered_labels = torch.tensor([i % 10 for i in range(64)]).to(device)
  sample_images = generator_GAN(z, ordered_labels).data.cpu().view(64, 1, 28, 28)
  grid = make_grid(sample_images, nrow=8, normalize=True)
  show(grid.cpu().detach().permute(1,2,0), sz=5)

"""Noise generation for CGAN with MLP"""
def noise(batch_size):
    n = torch.randn(batch_size, 100)
    return n.to(device)

# training Loop 
num_epochs = 40
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    N = len(data_loader)
    for _, (images, labels) in enumerate(data_loader):
        n_images = len(images)
        
        real_data = images.view(n_images, -1).to(device)
        real_labels = labels.to(device)
        z = noise(n_images).to(device)
        fake_labels = torch.randint(0, num_class, (n_images,)).to(device)

        fake_data = generator_GAN(noise(n_images), fake_labels).to(device)
        # fake_data = fake_data
        d_loss = discriminator_train_step(real_data, real_labels, fake_data, fake_labels)

        fake_labels = torch.randint(0, num_class, (n_images,)).to(device)
        fake_data = generator_GAN(noise(n_images), fake_labels).to(device)
        g_loss = generator_train_step(fake_data, fake_labels)

    if (epoch+1) % 5 == 0:
        plot_samples()
        print(f"Epoch: {epoch+1}")

"""Generation with DCGAN"""
class DCGAN_gen(nn.Module):
    def __init__(self):
        super().__init__()

        self.labelembedding = nn.Embedding(num_class, embedding_dim)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(110, 128, kernel_size=4, stride=1, padding=0),  # [batch, 128, 4, 4]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), 
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # [batch, 64, 8, 8]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), 
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # [batch, 32, 16, 16]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=3),     # [batch, 1, 28, 28]
            nn.Tanh()
        )
    def forward(self, noise, labels):
        c = self.labelembedding(labels)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.expand(-1, -1, noise.size(2), noise.size(3))
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        return img

class DCGAN_disc(nn.Module):
    def __init__(self):
        super().__init__()

        self.labelembedding = nn.Embedding(num_class, embedding_dim)

        self.model = nn.Sequential(
            nn.Conv2d(1 + embedding_dim, 64, kernel_size=4, stride=2, padding=1), # 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), 
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(), 
            nn.Linear(16*6*6, 1),            
            nn.Sigmoid()
        )
    def forward(self, img, labels):
        c = self.labelembedding(labels)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, c], 1)
        validity = self.model(x)
        return validity

generator_DCGAN = DCGAN_gen().to(device)
discriminator_DCGAN = DCGAN_disc().to(device)

# Parameter Count 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

vanilla_gen_params = count_parameters(generator_DCGAN)
vanilla_disc_params = count_parameters(discriminator_DCGAN)

print(f"Vanilla Generator Parameters: {vanilla_gen_params:,}")
print(f"Vanilla Discriminator Parameters: {vanilla_disc_params:,}")

# Optimizers and loss function
d_optimizer = optim.Adam(discriminator_DCGAN.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator_DCGAN.parameters(), lr=0.0002)
loss = nn.BCELoss()

# Noise generation for CGAN with DCGAN
def noise_2d(batch_size):
    n = torch.randn(batch_size, 100, 1, 1)
    return n.to(device)

"""DCGAN discriminator training step"""
def discriminator_train_step(real_data, real_labels, fake_data, fake_labels):
    vec_ones = torch.ones(len(real_data), 1).to(device)
    vec_zeros = torch.zeros(len(real_data), 1).to(device)

    discriminator_DCGAN.zero_grad()

    prediction_real = discriminator_DCGAN(real_data, real_labels)
    error_real = loss(prediction_real, vec_ones)
    error_real.backward()

    prediction_fake = discriminator_DCGAN(fake_data.detach(), fake_labels)
    error_fake = loss(prediction_fake, vec_zeros)
    error_fake.backward()

    d_optimizer.step()
    return error_real + error_fake

"""DCGAN generator training step"""
def generator_train_step(fake_data, fake_labels):
    vec_ones = torch.ones(len(fake_data), 1).to(device)

    g_optimizer.zero_grad()

    prediction = discriminator_DCGAN(fake_data, fake_labels)
    error = loss(prediction, vec_ones)
    error.backward()

    g_optimizer.step()
    return error

"""Plotting function for 2d CGAN"""
def plot_samples_2d():
    z = torch.randn(64, 100, 1, 1).to(device)
    ordered_labels = torch.tensor([i % 10 for i in range(64)]).to(device)
    sample_images = generator_DCGAN(z, ordered_labels).data.cpu()
    grid = make_grid(sample_images, nrow=8, normalize=True)
    show(grid.cpu().detach().permute(1,2,0), sz=5)
    os.makedirs("./Outputs", exist_ok=True)

# Training Loop for CGAN with DCGAN
num_epochs = 40 
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    N = len(data_loader) 
    for _, (images, labels) in enumerate(data_loader):
        n_images = len(images)
        
        real_data = images.view(n_images, 1, 28, 28).to(device)
        real_labels = labels.to(device)
        z = noise_2d(n_images).to(device)
        fake_labels = torch.randint(0, num_class, (n_images,)).to(device)

        fake_data = generator_DCGAN(noise_2d(n_images), fake_labels).to(device)
        
        # fake_data = fake_data
        d_loss = discriminator_train_step(real_data, real_labels, fake_data, fake_labels)

        fake_labels = torch.randint(0, num_class, (n_images,)).to(device)
        fake_data = generator_DCGAN(noise_2d(n_images), fake_labels).to(device)
        g_loss = generator_train_step(fake_data, fake_labels)

    if (epoch+1) % 5 == 0:
        plot_samples_2d()
        print(f"Epoch: {epoch+1}")


"""Part 2: MLP Classifier to classify generated samples"""


class MNIST_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize
classifier = MNIST_Classifier().to(device)

# Loss and optimizer
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(classifier.parameters(), lr=0.001)

# Training Loop for Classifier
num_classifier_epochs = 5 
for epoch in range(num_classifier_epochs):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer_cls.zero_grad()
        outputs = classifier(images)
        loss = criterion_cls(outputs, labels)
        
        loss.backward()
        optimizer_cls.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(data_loader)
    epoch_acc = 100 * correct / total
    
    print(f"Classifier Epoch [{epoch+1}/{num_classifier_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

"""Evaluation of Classifier on Generated Samples"""
def evaluate_cgan_accuracy(generator, classifier, samples_per_class=100):
    generator.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for digit in range(10):
            labels = torch.full((samples_per_class,), digit, dtype=torch.long).to(device)
            z = torch.randn(samples_per_class, 100).to(device)
            fake_images = generator(z, labels)
            fake_images = fake_images.view(-1, 1, 28, 28)
            outputs = classifier(fake_images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            class_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            print(f"Accuracy for digit {digit}: {class_acc:.2f}%")

    overall_accuracy = 100 * correct / total
    print(f"\nOverall CGAN Generation Accuracy: {overall_accuracy:.2f}%")
    return overall_accuracy

def evaluate_dcgan_accuracy(generator, classifier, samples_per_class=100):
    generator.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for digit in range(10):
            labels = torch.full((samples_per_class,), digit, dtype=torch.long).to(device)
            z = torch.randn(samples_per_class, 100, 1, 1).to(device)
            fake_images = generator(z, labels)
            outputs = classifier(fake_images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            class_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            print(f"Accuracy for digit {digit}: {class_acc:.2f}%")

    overall_accuracy = 100 * correct / total
    print(f"\nOverall DCGAN Generation Accuracy: {overall_accuracy:.2f}%\n")
    return overall_accuracy

vanilla_generator = generator_GAN.to(device)
dcgan_generator = generator_DCGAN.to(device)


cgan_accuracy = evaluate_cgan_accuracy(vanilla_generator, classifier)
dcgan_accuracy = evaluate_dcgan_accuracy(dcgan_generator, classifier)