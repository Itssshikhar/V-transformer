import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from vit import ViTForClassification

def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    #Save the config
    configfile = os.path.join(outdir, 'config.json')
    try:
        with open(configfile, 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

    except TypeError as e:
        print(f"Error saving config: {e}")
        raise
    
    #Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    data = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracies': accuracies,
    }
    
    try:
        with open(jsonfile, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4)
    except:
        print(f"Error saving metrics: {e}")
        for key, value in data.items():
            try:
                json.dumps(value)
            except ValueError as ve:
                print(f"Circular reference found in {key}: {ve}")
        raise

    #Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)

def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)
 
def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)

    #Load the config
    configfile = os.path.join(outdir,'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    #Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'metrics.json'):
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']

    #Load the model
    model = ViTForClassification(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies

def visualize_images():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

    classes = ('deer', 'car', 'frog', 'horse', 'ship', 'truck', 'cat', 'bird', 'plane', 'dog')
    #pick any 30 samples randomly
    indices = torch.randperm(len(trainset))[:30]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]
    
    #Visualizing using matplotlib
    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[1])
        ax.set_title(classes[labels[i]])

@torch.no_grad()
def visualize_attention(model, output=None, device="cuda"):
    #Attention maps for first 5 images
    model.eval()
    num_images = 30
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    classes = ('deer', 'car', 'frog', 'horse', 'ship', 'truck', 'cat', 'bird', 'plane', 'dog')
    #Pick any 30 samples randomly
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]
    #Convert the images to tensors
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((32, 32)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = torch.stack([test_transform(image) for image in raw_images])
    #Move images to the device
    images = images.to(device)
    model = model.to(device)
    #Get the attention maps from the last block
    logits, attention_maps = model(images, output_attention=True)
    #Get the predictions
    predictions = torch.argmax(logits, dim=1)
    #Concatenate the attention maps from all blocks
    attention_maps = attention_maps[:, :, 0, 1:]
    #Then average the attention of CLS token over all the heads
    attention_maps = attention_maps.mean(dim=1)
    #Reshape the attention maps to a square
    num_patches = attention_maps.view(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    #Resize the map to the size and the attention maps
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    #Plot the images and the attentions maps
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        #Mask out the attention map of the left image
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        #Show the ground truth and the predictions
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()
