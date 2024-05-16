class Config:
    gpu = '0'
    data_path = './CETUS/'
    model = 'vm2'
    result_dir = './Result'
    lr = 1e-5
    max_epochs = 201
    sim_loss = 'mse'
    alpha = 0.04
    batch_size = 1
    n_save_epoch = 5
    model_dir = './Checkpoint/exp_'
    log_dir = './Log'
    saved_unet_name = 'unet_model.pth'
    saved_tnet_name = 'tnet_model.pth'

    # test时参数
    checkpoint_path = "./Checkpoint/exp_1715837905.2272804/"
