import os
import torch
import numpy as np
import torch.utils.data as Data
from Model import losses
from Model.config import Config as args
from Model.dataset import CETUS
from Model.model import U_Network, SpatialTransformer, TransformerNet, AffineCOMTransform
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


def test():
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    test_ds = CETUS(args.data_path, 'test')
    vol_size = test_ds[0]['es'].shape[1:]
    test_dl = Data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # create affine
    affine_transform = AffineCOMTransform(device)

    # create Unet
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # create STN
    STN_label = SpatialTransformer(vol_size, mode='nearest').to(device)

    dice_fn = losses.compute_label_dice

    best_UNet_model = os.path.join(args.checkpoint_path, args.saved_unet_name)
    best_UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    best_UNet.load_state_dict(torch.load(best_UNet_model))
    best_UNet.eval()

    best_tnet_model = os.path.join(args.checkpoint_path, args.saved_tnet_name)
    best_tnet = TransformerNet().to(device)
    best_tnet.load_state_dict(torch.load(best_tnet_model))
    best_tnet.eval()
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


if __name__ == "__main__":
    args = args()
    test()
