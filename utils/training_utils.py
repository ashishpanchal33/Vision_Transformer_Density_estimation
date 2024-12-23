import torch
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from tempfile import TemporaryDirectory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as tqdm_bar



def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    dataset_sizes,
    tensorboard_writer,
    model_type,
    class_names,
    num_epochs=25,
    epoch_start = 0
):
    since = time.time()

    models_path = os.path.join("models", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_type)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    best_model_params_path = os.path.join(models_path, f"best_model_{model_type}_params.pt")
    
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    
    
    performance_dict = dict(
                            readings= dict(
                            
                                            train_accuracy = [],
                                            train_loss = [],
                                            val_accuracy = [],
                                            val_loss = []

                                            ),
                            best_epoch = 0
    
    
                            )
    
    for cn in class_names:
        performance_dict['readings'].update({cn:[]})
    
    
    
    cosine_scheduler = ('CosineAnnealingWarmRestarts' in str(scheduler.__class__))
    epoch,iteration = 0,0
    
    
    # Writing model type and hyperparameters to TensorBoard
    tensorboard_writer.add_text("Model Type", model_type)

    for epoch in range(epoch_start,num_epochs+epoch_start):
        print(f"Epoch {epoch}/{num_epochs+epoch_start - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode

                current_lr = optimizer.param_groups[0]["lr"]
                    
                
                
                print(f"current_LR: {current_lr:.8f}")
                tensorboard_writer.add_scalar(f"LR", current_lr, epoch)

            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            class_corrects = torch.zeros(len(class_names), dtype=torch.double, device=device)
            class_totals = torch.zeros(len(class_names), dtype=torch.double, device=device)

            # Use tqdm for progress bar
            data_loader = tqdm_bar(dataloaders[phase], desc=f'Epoch {epoch}, {phase}', leave=False)
            Batch_count = len(data_loader)
            # Iterate over data.
            for iteration,(inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                
                
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    
                    if phase == "train" and cosine_scheduler:
                        
                        if (epoch+1)*(iteration+1) >= scheduler.warmup.total_iters:
                            scheduler.step(epoch=epoch,iteration=iteration,Batch_count=Batch_count) 
                            #cur_lr = optimizer.param_groups[0]["lr"]
                            #print(f"current_LR: {cur_lr:.8f}")
                        
                        
                        
                        
                    
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        if cosine_scheduler and ((epoch+1)*(iteration+1) < scheduler.warmup.total_iters):
                            scheduler.warmup_step()
                            #cur_lr = optimizer.param_groups[0]["lr"]
                            #print(f"current_LR: {cur_lr:.8f}")

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Calculate per-class accuracy
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    class_corrects[t] += p == t
                    class_totals[t] += 1

                # Update tqdm progress bar
                data_loader.set_postfix({'loss': loss.item()})

            if phase == "train" and not(cosine_scheduler):
                scheduler.step()
                
                
                
                
                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            class_accuracy = class_corrects / class_totals
            
            if phase == "train":
                performance_dict['readings']['train_accuracy'].append(epoch_acc)
                performance_dict['readings']['train_loss'].append(epoch_loss)
            elif phase == "val":
                performance_dict['readings']['val_accuracy'].append(epoch_acc)
                performance_dict['readings']['val_loss'].append(epoch_loss)

            
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print("Per-class Accuracy:")
            for i, name in enumerate(class_names):
                print(f"{name}: {class_accuracy[i]:.4f}")
                
                if phase == "val":
                    performance_dict['readings'][name].append(class_accuracy[i])
                
                
                
                

            tensorboard_writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            tensorboard_writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)

            # Write per-class accuracy to TensorBoard
            for i, name in enumerate(class_names):
                tensorboard_writer.add_scalar(f"Per-Class Accuracy/{name}/{phase}", class_accuracy[i], epoch)

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                performance_dict['best_epoch'] = epoch
                # Save best model with model type and hyperparameters in filename
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    
    return model , performance_dict
def visualize_model(model, dataloaders, device, class_names, imshow, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {class_names[preds[j]]}")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def create_writer(
    experiment_name: str, model_name: str, extra: str = None
) -> SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    # Get timestamp of current date (all experiments on certain day live in same folder)
    date = datetime.now().strftime(
        "%Y-%m-%d"
    )  # returns current date in YYYY-MM-DD format
    time = datetime.now().strftime("%H-%M-%S")  # returns current time

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", date, experiment_name, model_name, extra, time)
    else:
        log_dir = os.path.join("runs", date, experiment_name, model_name, time)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
