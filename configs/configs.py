import ml_collections


def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 32
    config.epochs = 1
    config.mask_ratio = 0.5
    config.epochs_TS = 100
    config.is_train_TS = False
    config.masking_method = "strategy_masking_eff"
    config.output_dir = "../experiments/avenue_learning_masking"  # the checkpoints will be loaded from here
    config.abnormal_score_func = 'L2'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train'
    config.resume = True
    config.resume_TS = False
    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "avenue"
    config.avenue_path = "../dataset/avenue"
    config.avenue_gt_path = "../dataset/avenue/gt_avenue"
    config.percent_abnormal = 0.25
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.start_epoch_TS = 0
    config.print_freq = 10
    config.num_workers = 3
    config.pin_mem = False

    return config


def get_configs_shanghai():
    config = ml_collections.ConfigDict()
    config.batch_size = 100
    config.epochs = 50
    config.mask_ratio = 0.5
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
    config.output_dir = "github_ckpt/shanghai" # the checkpoints will be loaded from here
    config.abnormal_score_func = 'L1'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = "inference"
    config.resume=False

    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "shanghai"
    config.shanghai_path = "/home/alin/datasets/SanhaiTech"
    config.shanghai_gt_path = "/media/alin/hdd/Transformer_Labels/Shanghai_gt"
    config.percent_abnormal = 0.5
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 10
    config.pin_mem = False

    return config
