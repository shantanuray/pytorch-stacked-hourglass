import torch
import torch.backends.cudnn
import torch.nn.parallel
from tqdm import tqdm
import pandas as pd

from stacked_hourglass.loss import joints_mse_loss
from stacked_hourglass.utils.evaluation import accuracy, AverageMeter, final_preds, final_preds_untransformed
from stacked_hourglass.utils.transforms import fliplr, flip_back

# # A list of joints to include in the accuracy reported as part of the progress bar.
_ACC_JOINTS = [1]  #, 2, 3, 4, 5, 6, 11, 12, 15, 16]


def do_training_step(model, optimiser, input, target, target_weight=None):
    assert model.training, 'model must be in training mode.'
    assert len(input) == len(target), 'input and target must contain the same number of examples.'

    with torch.enable_grad():
        # Forward pass and loss calculation.
        output = model(input)
        loss = sum(joints_mse_loss(o, target, target_weight) for o in output)

        # Backward pass and parameter update.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return output[-1], loss.item()


def do_training_epoch(train_loader, model, device, optimiser):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # Put the model in training mode.
    model.train()

    progress = tqdm(enumerate(train_loader), total=len(train_loader), ascii=True, leave=True)
    for i, (input, target, meta) in progress:
        input, target = input.to(device), target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)

        output, loss = do_training_step(model, optimiser, input, target, target_weight)
        # Get list of joints from the data
        # _ACC_JOINTS = range(meta['tpts'].to(device, non_blocking=True).size(0))
        acc = accuracy(output, target, _ACC_JOINTS)

        # measure accuracy and record loss
        losses.update(loss, input.size(0))
        accuracies.update(acc[0], input.size(0))

        # Show accuracy and loss as part of the progress bar.
        progress.set_postfix_str('Loss: {loss:0.4f}, Acc: {acc:0.4f}'.format(
            loss=losses.avg,
            acc=accuracies.avg
        ))

    return losses.avg, accuracies.avg


def do_validation_step(model, input, target, target_weight=None, flip=False):
    assert not model.training, 'model must be in evaluation mode.'
    assert len(input) == len(target), 'input and target must contain the same number of examples.'

    # Forward pass and loss calculation.
    output = model(input)
    loss = sum(joints_mse_loss(o, target, target_weight) for o in output)

    # Get the heatmaps.
    if flip:
        # If `flip` is true, perform horizontally flipped inference as well. This should
        # result in more robust predictions at the expense of additional compute.
        flip_input = fliplr(input.clone().cpu().numpy())
        flip_input = torch.as_tensor(flip_input, dtype=torch.float32)
        flip_output = model(flip_input)
        flip_output = flip_output[-1].cpu()
        flip_output = flip_back(flip_output)
        heatmaps = (output[-1].cpu() + flip_output) / 2
    else:
        heatmaps = output[-1].cpu()

    return heatmaps, loss.item()


def do_validation_epoch(val_loader, model, device, flip=False,
                        debug=False, debug_file=None, epoch=None):
    losses = AverageMeter()
    accuracies = AverageMeter()
    predictions = torch.zeros(len(val_loader.dataset), 1, 2)

    # Put the model in evaluation mode.
    model.eval()

    progress = tqdm(enumerate(val_loader), total=len(val_loader), ascii=True, leave=True)
    for i, (input, target, meta) in progress:
        # Copy data to the training device (eg GPU).
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)

        heatmaps, loss = do_validation_step(model, input, target, target_weight, flip)

        # Get list of joints from the data
        # _ACC_JOINTS = range(meta['tpts'].to(device, non_blocking=True).size(0))
        # Calculate PCKh from the predicted heatmaps.
        acc = accuracy(heatmaps, target.cpu(), _ACC_JOINTS)

        # Calculate locations in original image space from the predicted heatmaps.
        out_res = [meta['out_res'].data.cpu().numpy()[0],
                   meta['out_res'].data.cpu().numpy()[0]]
        if 'out_res' in meta and 'inp_res' in meta and 'rot' in meta:
            coords = final_preds_untransformed(heatmaps, out_res)
            preds = final_preds(coords,
                                meta['center'],
                                meta['scale'],
                                out_res)
        else:
            # Original code
            coords = final_preds_untransformed(heatmaps, out_res)
            preds = final_preds(coords,
                                meta['center'],
                                meta['scale'],
                                [64, 64])
        validation_log = pd.DataFrame()
        if debug:
            pts_df = pd.DataFrame(meta['pts'].data.cpu().numpy().squeeze(axis=1),
                                  columns=['orig_ref_x', 'orig_ref_y', 'orig_prob'])
            tpts_df = pd.DataFrame(meta['tpts'].data.cpu().numpy().squeeze(axis=1),
                                   columns=['xform_ref_x', 'orig_ref_y', 'orig_prob'])
            coords_df = pd.DataFrame(coords.data.cpu().numpy().squeeze(axis=1),
                                     columns=['xform_pred_x', 'xform_pred_y'])
            preds_df = pd.DataFrame(preds.data.cpu().numpy().squeeze(axis=1),
                                    columns=['pred_x', 'pred_y'])
            epoch_df = pd.DataFrame([epoch] * len(pts_df), columns=['epoch'])
            img_df = pd.DataFrame(meta['img_paths'], columns=['img_paths'])
            validation_log = pd.concat([epoch_df,
                                       img_df,
                                       pts_df,
                                       tpts_df,
                                       coords_df,
                                       preds_df], axis=1)
            validation_log.to_csv(debug_file, mode='a', header=False)
        for example_index, pose in zip(meta['index'], preds):
            predictions[example_index] = pose

        # Record accuracy and loss for this batch.
        losses.update(loss, input.size(0))
        accuracies.update(acc[0].item(), input.size(0))

        # Show accuracy and loss as part of the progress bar.
        progress.set_postfix_str('Loss: {loss:0.4f}, Acc: {acc:0.4f}'.format(
            loss=losses.avg,
            acc=accuracies.avg
        ))

    return losses.avg, accuracies.avg, predictions, validation_log
