import os
import time
import torch
import numpy as np
from torch.optim import Adam
import torch.utils.data as Data
from Model import losses
from Model.config import Config as args
from Model.dataset import CETUS
from Model.model import U_Network, SpatialTransformer, TransformerNet, AffineCOMTransform
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs(model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def train():
    model_dir = args.model_dir + str(time.time())
    # create file and declare the device
    make_dirs(model_dir)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # log file
    log_name = str(args.max_epochs) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    out_f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # define the train/valid/test dataset and dataloader
    ds = CETUS(args.data_path, 'train')
    valid_ds = CETUS(args.data_path, 'valid')
    test_ds = CETUS(args.data_path, 'test')
    print("Number of training images: ", len(ds))
    print(f'image shape: {ds[0]["es"].shape}')

    dl = Data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_dl = Data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_dl = Data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    vol_size = ds[0]['es'].shape[1:]

    # create nets
    # create transformerNet
    tnet = TransformerNet().to(device)
    tnet.train()

    # create affine
    affine_transform = AffineCOMTransform(device)

    # create Unet
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    UNet.train()

    # create STN
    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode='nearest').to(device)
    STN.train()

    # calculate the number of parameters
    print('TransformNet:', count_parameters(tnet))
    print('AffineTransform:', count_parameters(affine_transform))
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # Set optimizer and losses
    opt_t = Adam(tnet.parameters(), lr=args.lr)
    opt_u = Adam(UNet.parameters(), lr=args.lr * 10)

    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    dice_fn = losses.compute_label_dice
    grad_loss_fn = losses.gradient_loss

    # record the best dice on valid dataset
    max_dice = 0

    # Training loop.
    for i in range(0, args.max_epochs + 1):
        # evaluation on valid dataset
        # The purpose of performing validation first rather than training is to obtain the initial dice of the validation set
        if i % args.n_save_epoch == 0 or i == args.max_epochs:
            UNet.eval()
            tnet.eval()
            with torch.no_grad():
                dice_list = []
                for valid_iter_, valid_d in enumerate(valid_dl):
                    m, f, ml, fl = valid_d['es'], valid_d['ed'], valid_d['es_gt'], valid_d['ed_gt']

                    # [B, C, W, H]
                    moving_label = ml.to(device).float()
                    fixed_label = fl.to(device).float()
                    moving = m.to(device).float()
                    fixed = f.to(device).float()

                    # Run the data through the model to produce warp and flow field
                    affine_param = tnet(moving, fixed)
                    affine_moving, affine_matrix = affine_transform(moving, affine_param)

                    affine_moving_label = F.grid_sample(moving_label, F.affine_grid(affine_matrix, moving_label.shape,
                                                                                    align_corners=True), mode="nearest",
                                                        align_corners=True)

                    flow_m2f = UNet(affine_moving, fixed)
                    m2f_label = STN_label(affine_moving_label, flow_m2f)

                    # Calculate dice score

                    dice_score = dice_fn(m2f_label, fixed_label)
                    dice_list.append(dice_score.item())
                mean_dice = np.array(dice_list).mean()
                print(f'current dice: {mean_dice}  max dice: {max_dice}')

            if mean_dice >= max_dice:
                max_dice = mean_dice
                # Save model checkpoint
                save_file_name = os.path.join(model_dir, args.saved_unet_name)
                torch.save(UNet.state_dict(), save_file_name)
                save_file_name = os.path.join(model_dir, args.saved_tnet_name)
                torch.save(tnet.state_dict(), save_file_name)
                print('model saved at epoch: %d. max_dice: %f' % (i, mean_dice))

            # evaluation on test dataset
            if i == args.max_epochs:
                best_UNet_model = os.path.join(model_dir, args.saved_unet_name)
                best_UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
                best_UNet.load_state_dict(torch.load(best_UNet_model))

                best_tnet_model = os.path.join(model_dir, args.saved_tnet_name)
                best_tnet = TransformerNet().to(device)
                best_tnet.load_state_dict(torch.load(best_tnet_model))
                with torch.no_grad():
                    dice_list = []
                    dice_before_list = []
                    jac_list = []
                    for test_iter_, test_d in enumerate(test_dl):
                        m, f, ml, fl = test_d['es'], test_d['ed'], test_d['es_gt'], test_d['ed_gt']
                        # [B, C, W, H]
                        moving_label = ml.to(device).float()
                        fixed_label = fl.to(device).float()
                        moving = m.to(device).float()
                        fixed = f.to(device).float()

                        # Run the data through the model to produce warp and flow field
                        affine_param = best_tnet(moving, fixed)
                        affine_moving, affine_matrix = affine_transform(moving, affine_param)

                        affine_moving_label = F.grid_sample(moving_label,
                                                            F.affine_grid(affine_matrix, moving_label.shape,
                                                                          align_corners=True), mode="nearest",
                                                            align_corners=True)

                        flow_m2f = best_UNet(affine_moving, fixed)
                        m2f_label = STN_label(affine_moving_label, flow_m2f)

                        # Calculate dice score
                        dice_score = dice_fn(m2f_label, fixed_label)
                        dice_list.append(dice_score.item())
                        dice_before_list.append(dice_fn(moving_label, fixed_label).item())

                        tar = moving.detach().cpu().numpy()[0, 0, ...]
                        jac_den = np.prod(tar.shape)
                        for flow_item in flow_m2f:
                            jac_det = losses.jacobian_determinant(flow_item.detach().cpu().numpy())
                            jac_list.append(np.sum(jac_det <= 0) / jac_den)

                    mean_dice = np.array(dice_list).mean()
                    before_mean_dice = np.array(dice_before_list).mean()
                print(f'test dice: {mean_dice:.5f}, original dice: {before_mean_dice:.5f}')
                print(f'test jacob mean: {np.array(jac_list).mean()}, jacob std: {np.array(jac_list).std()}')
                break
            UNet.train()
            tnet.train()

        # Generate the moving images and convert them to tensors.
        loss_list, sim_loss_list, grad_loss_list = [], [], []
        for iter_, d in enumerate(dl):
            m, f = d['es'], d['ed']
            # [B, C, W, H, D]
            input_moving = m.to(device).float()
            input_fixed = f.to(device).float()

            # Run the data through the model to produce warp and flow field

            # update the transformer net
            affine_param = tnet(input_moving, input_fixed)
            affine_moving, affine_matrix = affine_transform(input_moving, affine_param)
            sim_loss = sim_loss_fn(affine_moving, input_fixed)
            opt_t.zero_grad()
            sim_loss.backward()
            opt_t.step()

            # update the Unet
            affine_param = tnet(input_moving, input_fixed)
            affine_moving, affine_matrix = affine_transform(input_moving, affine_param)
            flow_m2f = UNet(affine_moving.detach(), input_fixed)
            m2f = STN(affine_moving, flow_m2f)

            # Calculate loss
            sim_loss = sim_loss_fn(m2f, input_fixed)
            grad_loss = grad_loss_fn(flow_m2f)
            loss = sim_loss + args.alpha * grad_loss
            loss_list.append(loss.item())
            sim_loss_list.append(sim_loss.item())
            grad_loss_list.append(grad_loss.item())
            # Backwards and optimize
            opt_u.zero_grad()
            loss.backward()
            opt_u.step()

            print("epoch: %d/%d iter: %d  loss: %f  sim: %f  grad: %f" % (
                i, args.max_epochs, iter_, loss.item(), sim_loss.item(), grad_loss.item()), flush=True)
        print("summary --- epoch: %d, %f, %f, %f" % (i, np.array(loss_list).mean(), np.array(sim_loss_list).mean(),
                                                     np.array(grad_loss_list).mean()))
        print("%d, %f, %f, %f" % (i, np.array(loss_list).mean(), np.array(sim_loss_list).mean(),
                                  np.array(grad_loss_list).mean()), file=out_f)
    out_f.close()


if __name__ == "__main__":
    args = args()
    train()
