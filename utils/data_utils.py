from torch.utils.data import Subset
import os
import random

def subset_imagefolder(dataset, subset_size):
    # Get the list of class folders
    class_folders = dataset.classes

    # Create a dictionary to store the indices of samples for each class
    subset_indices = {folder: [] for folder in class_folders}

    # Iterate over the dataset and collect indices for each class
    for index, (_, label) in enumerate(dataset.samples):
        class_folder = class_folders[label]
        subset_indices[class_folder].append(index)

    # Select the first 'subset_size' indices for each class
    selected_indices = []
    for folder in class_folders:
        indices = subset_indices[folder]
        if subset_size < len(indices):
            indices_to_add = indices[:subset_size]
        else:
            indices_to_add = indices
        selected_indices.extend(indices_to_add)
        #print(folder, len(indices_to_add))
    # Create a Subset dataset using the selected indices
    subset_dataset = Subset(dataset, selected_indices)

    return subset_dataset



def exclude_dir(dataset, exclude_dir, exclude_size):
    # Get the list of class folders
    class_folders = dataset.classes

    # Create a dictionary to store the indices of samples for each class
    subset_indices = {folder: [] for folder in class_folders}

    # Iterate over the dataset and collect indices for each class
    for index, (_, label) in enumerate(dataset.samples):
        class_folder = class_folders[label]
        subset_indices[class_folder].append(index)

    selected_indices = []
    for folder in class_folders:
        indices = subset_indices[folder]
        #print(folder)
        if folder == exclude_dir:
            indices_to_add = random.sample(indices, exclude_size)
        else:
            indices_to_add = indices
        #print(len(indices_to_add))
        selected_indices.extend(indices_to_add)
    
    # Create a Subset dataset using the selected indices
    subset_dataset = Subset(dataset, selected_indices)

    return subset_dataset
