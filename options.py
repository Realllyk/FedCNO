import argparse
from utils.seed import set_global_seed

def parse_args():
    parser = argparse.ArgumentParser("Local model Experiments")
    
    parser.add_argument(
        '--vul',
        default='reentrancy',
        type=str,
        help='Type of vulnerability'
    )

    parser.add_argument(
        '--epoch',
        default=10,
        type=int,
        help='epochs for global traning'
    )

    parser.add_argument(
        '--local_epoch',
        default=1,
        type=int,
        help='epochs for local training'
    )

    parser.add_argument(
        '--inner_lr',
        default=0.0005,
        type=float,
        help="learning rate for inner model"
    )

    parser.add_argument(
        '--outer_lr',
        default=0.0003,
        type=float,
        help='learning rate for outer model'
    )

    parser.add_argument(
        '--batch',
        default=8,
        type=int,
        help='batch size for data loader'
    )

    parser.add_argument(
        '--input_channels',
        default=428,
        type=int,
        choices=[138, 428],
        help='input channels for LCN model'
    )

    parser.add_argument(
        '--client_num',
        default=4,
        type=int,
        help='num of clients in federated training'
    )

    parser.add_argument(
        '--noise',
        action='store_true'
    )

    parser.add_argument(
        '--noise_type',
        choices=['pure', 'non_noise', 'fn_noise', 'diff_noise', 'sys_noise'],
        default='non_noise',
        help='Is dataset contains noise. If it does, what kind of noise is'
    )

    parser.add_argument(
        '--alpha',
        default=0.1,
        type=float
    )

    parser.add_argument(
        '--beta',
        default=1.0,
        type=float
    )

    parser.add_argument(
        '--noise_rate',
        default=0.05,
        type=float
    )

    parser.add_argument(
        '--device',
        default='cuda:0',
        type=str,
        help="training device"
    )

    parser.add_argument(
        '--cbgru_local_epoch',
        default=8,
        type=int
    )

    parser.add_argument(
        '--cge_local_epoch',
        default=8,
        type=int
    )

    parser.add_argument(
        '--mando_local_epoch',
        default=8,
        type=int,
        help='Local epochs for MANDO'
    )

    parser.add_argument(
        '--cbgru_local_lr',
        default=0.00008,
        type=float,
        help='Local learning rate for CBGRU. Recommended: 0.00005 for reentrancy, 0.0001 for others'
    )

    parser.add_argument(
        '--cge_local_lr',
        default=0.0001,
        type=float
    )

    parser.add_argument(
        '--mando_local_lr',
        default=0.0001,
        type=float
    )

    parser.add_argument(
        '-d',
        '--dropout', 
        type=float, 
        default=0.5, 
        help='dropout rate')
    
    parser.add_argument(
        '--cbgru_net1',
        type=str,
        default='cnn',
        choices=['cnn', 'bilstm', 'bigru']
    )

    parser.add_argument(
        '--cbgru_net2',
        type=str,
        default='bigru',
        choices=['cnn', 'bigru', 'bilstm']
    )

    parser.add_argument(
        '--sample_rate',
        type=float,
        default=0.4,
        help="client sample rate"
    )

    parser.add_argument(
        '--seed',
        type=float,
        default=1.
    )

    parser.add_argument(
        '--relabel_ratio', 
        type=float, 
        default=0.5, 
        help="proportion of relabeled samples among selected noisy samples"
    )

    parser.add_argument(
        '--fine_tuning', 
        action='store_false', 
        help='whether to include fine-tuning stage'
    )
    
    parser.add_argument(
        '--correction', 
        action='store_false', 
        help='whether to correct noisy labels'
    )

    parser.add_argument(
        '--reg_weight', 
        help='weight of regularization term', 
        type=float, 
        required=False
    )

    parser.add_argument(
        '--frac2',
        type=float,
        default=0.1,
        help="fration of selected clients in fine-tuning and usual training stage"
    )

    parser.add_argument(
        '--rounds1', 
        type=int, 
        default=200, 
        help="rounds of training in fine_tuning stage"
    )

    parser.add_argument(
        '--rounds2', 
        type=int, 
        default=200, 
        help="rounds of training in usual training stage"
    )
    
    parser.add_argument(
        '--corr_seed',
        type=int,
        default=13
    )

    parser.add_argument(
        '--iteration1',
        type=int,
        default=50,
        help="enumerate iteration in preprocessing stage"
    )

    parser.add_argument(
        '--confidence_thres', 
        type=float, 
        default=0.5, 
        help="threshold of model's confidence on each sample"
    )

    parser.add_argument(
        '--clean_set_thres', 
        type=float, 
        default=0.1, 
        help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage"
    )

    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help="num of classes for classification"
    )

    parser.add_argument(
        '--global_weight',
        type=float,
        default=0.65,
        help='The weight for global knn labels'
    )

    parser.add_argument(
        '--adjustment_factor',
        type = float,
        default=0.05,
        help="adjust factor for global weight"
    )

    parser.add_argument(
        '--warm_up_epoch',
        type=int,
        default=25,
        help="warm up epoch before train stage"
    )

    parser.add_argument(
        '--warmup_valid_interval',
        type=int,
        default=1,
        help='Validation interval (in warm-up epochs) for Fed_Avg warm-up stage'
    )

    parser.add_argument(
        '--warmup_early_stop_patience',
        type=int,
        default=0,
        help='Early stop patience for warm-up stage; <=0 disables early stopping'
    )

    parser.add_argument(
        '--warmup_early_stop_min_delta',
        type=float,
        default=1e-4,
        help='Minimum validation F1 improvement to reset warm-up early stopping'
    )

    parser.add_argument(
        '--warmup_min_epoch_for_early_stop',
        type=int,
        default=0,
        help='Minimum warm-up epoch before early-stop counter can trigger stopping'
    )

    parser.add_argument(
        '--warmup_valid_f1_smooth_window',
        type=int,
        default=1,
        help='Smoothing window size for warm-up validation F1 used by early stopping (1 disables smoothing)'
    )

    parser.add_argument(
        '--random_noise',
        action= 'store_true',
        help= "deprecated no-op; kept for backward compatibility"
    )

    parser.add_argument(
        '--valid_frac',
        type = float,
        default = 1.0,
        help = "fraction of how much test data uesd to test"
    )

    parser.add_argument(
        '--lab_name',
        type=str,
        default='feature_Fed_LGV',
        # choices=["feature_Fed_LGV", "Fed_LGV", "non_feature_Fed_LGV", "non_Fed_LGV", "Ablation_no_global_Fed_LGV", 'new_feature_Fed_LGV', 'label_Fed_LGV', 'abl_no_glob', 'abl_no_local', 'abl_no_cons'],
        help="name for files to save result"
    )

    parser.add_argument(
        '--result_root',
        type=str,
        default='graduate_final_result',
        help='root directory for global test results'
    )

    parser.add_argument(
        '--num_neigh',
        type = int,
        default = 5,
        help = "number of neighbors for knn algorithm"
    )

    parser.add_argument(
        '--first_epochs', 
        type=int, 
        default=50,
        help="number of rounds before correction"
    )

    parser.add_argument(
        '--last_epochs', 
        type=int, 
        default=50,
        help="number of rounds after correction"
    )

    parser.add_argument(
        '--model_type', 
        type=str, 
        default='CBGRU',
        choices=['CBGRU', 'CGE', 'MANDO'],
        help="predict model used in FedCNO"
    )

    parser.add_argument(
        '--ablation',
        type=str,
        default='full',
        choices=[
            'full',
            'no_local',
            'no_global',
            'no_unc_alpha',
            'no_cons_loss',
            'no_local_no_global',
            'no_global_no_unc_alpha',
        ],
        help='Fed_LGV ablation mode (used by fed_main/Fed_LGV_ablation.py)'
    )

    parser.add_argument(
        '--diff',
        action = 'store_true',
        help='whether noise rate different'
    )

    parser.add_argument(
        '--consistency_score',
        action='store_true',
        help='whether use consistency score'
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='L2 regularization weight decay'
    )

    parser.add_argument(
        '--n_clusters',
        type=int,
        default=15,
        help='Number of clusters for systemic noise generation'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/',
        help='directory of data'
    )

    parser.add_argument(
        '--mando_graph_dir',
        type=str,
        default='./data/mando_graph',
        help='directory of MANDO cached graph data'
    )

    parser.add_argument(
        '--alpha_min',
        type=float,
        default=0.05,
        help='Minimum value for alpha in Fed_LGV client'
    )

    parser.add_argument(
        '--alpha_max',
        type=float,
        default=0.8,
        help='Maximum value for alpha in Fed_LGV client'
    )

    parser.add_argument(
        '--lgv_valid_interval',
        type=int,
        default=1,
        help='Validation interval (in global epochs) for Fed_LGV'
    )

    parser.add_argument(
        '--lgv_early_stop_patience',
        type=int,
        default=15,
        help='Early stop patience for Fed_LGV; <=0 disables early stopping'
    )

    parser.add_argument(
        '--lgv_early_stop_min_delta',
        type=float,
        default=1e-4,
        help='Minimum validation F1 improvement to reset Fed_LGV early stopping'
    )

    parser.add_argument(
        '--lgv_min_epoch_for_early_stop',
        type=int,
        default=5,
        help='Minimum LGV epoch before early-stop counter can trigger stopping'
    )

    parser.add_argument(
        '--lgv_valid_f1_smooth_window',
        type=int,
        default=3,
        help='Smoothing window size for validation F1 used by LGV early stopping (1 disables smoothing)'
    )

    parser.add_argument(
        '--exit_after_warmup_test',
        action='store_true',
        help='Exit program after warm-up test in Fed_LGV'
    )

    parser.add_argument(
        '--run_warmup_global_test',
        action='store_true',
        help='Run and save one global_test on test set right after warm-up stage; default disabled'
    )

    parser.add_argument(
        '--lgv_pseudo_threshold',
        type=float,
        default=0.7,
        help='confidence threshold for Fed_LGV pseudo-label update (trust-score gate)'
    )

    parser.add_argument(
        '--lgv_gate_type',
        type=str,
        default='max_prob',
        choices=['max_prob', 'margin', 'entropy'],
        help='Trust-score gate type for Fed_LGV pseudo-label update'
    )

    parser.add_argument(
        '--lgv_use_triple_gate',
        action='store_true',
        help='Enable triple-gate for Fed_LGV pseudo-label update: changed-label AND improved-prob AND threshold gate'
    )

    parser.add_argument(
        '--lgv_improve_margin',
        type=float,
        default=0.0,
        help='Minimum required gain of pseudo-label max probability over old-label probability when triple-gate is enabled'
    )

    # FedCRD Arguments
    parser.add_argument(
        '--lambda_q',
        type=float,
        default=2.0,
        help='Penalty strength for client-side q_k consistency discrepancy in FedCRD'
    )

    parser.add_argument(
        '--lambda_agg',
        type=float,
        default=2.0,
        help='Penalty strength for server-side aggregation reliability decay in FedCRD'
    )

    parser.add_argument(
        '--alpha_crd',
        type=float,
        default=0.5,
        help='Weight for training process signal in FedCRD reliability correction'
    )

    parser.add_argument(
        '--C_min',
        type=float,
        default=0.5,
        help='[LEGACY/DEPRECATED] Minimum clipping threshold factor in FedCRD; '
             'current implementation does not apply this clipping strategy'
    )

    parser.add_argument(
        '--C_max',
        type=float,
        default=2.0,
        help='[LEGACY/DEPRECATED] Maximum clipping threshold factor in FedCRD; '
             'current implementation does not apply this clipping strategy'
    )

    parser.add_argument(
        '--ema_beta',
        type=float,
        default=0.9,
        help='Momentum coefficient for EMA model in FedCRD'
    )

    parser.add_argument(
        '--crd_q_anchor',
        type=str,
        default='global',
        choices=['global', 'ema'],
        help='anchor model for q_k consistency stats in FedCRD: global or ema'
    )

    parser.add_argument(
        '--crd_rho',
        type=float,
        default=0.7,
        help='rho quantile used to compute adaptive clipping threshold tau^t in FedCRD'
    )

    parser.add_argument(
        '--crd_eps',
        type=float,
        default=1e-8,
        help='numerical stability epsilon used in FedCRD normalization and clipping'
    )

    parser.add_argument(
        '--crd_valid_interval',
        type=int,
        default=1,
        help='Validation interval (in global epochs) for FedCRD'
    )

    parser.add_argument(
        '--crd_early_stop_patience',
        type=int,
        default=15,
        help='Early stop patience for FedCRD; <=0 disables early stopping'
    )

    parser.add_argument(
        '--crd_early_stop_min_delta',
        type=float,
        default=1e-4,
        help='Minimum validation F1 improvement to reset FedCRD early stopping'
    )

    parser.add_argument(
        '--crd_sigma',
        type=str,
        default='softplus',
        help='positive mapping function for FedCRD reliability (currently fixed to softplus)'
    )

    parser.add_argument(
        '--crd_ablation_mode',
        type=str,
        default='full',
        choices=['full', 'no_amb', 'no_cal', 'no_clip', 'only_clip', 'fedavg'],
        help='FedCRD ablation mode used by fed_main/Fed_CRD_ablation.py'
    )

    parser.add_argument(
        '--crd_noamb_variant',
        type=str,
        default='soft',
        choices=['soft', 'const'],
        help='NoAmb variant: soft=min(q_loc,q_glob), const=constant q_k'
    )

    parser.add_argument(
        '--crd_const_q',
        type=float,
        default=1.0,
        help='constant q_k value when --crd_noamb_variant const'
    )

    parser.add_argument(
        '--exp_tag',
        type=str,
        default='crd_ablation',
        help='experiment tag used in run timestamp and post-analysis filtering'
    )

    parser.add_argument(
        '--seed_list',
        type=str,
        default='',
        help='optional comma-separated seeds for external sweep scripts'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='number of workers for parallel training'
    )

    parser.add_argument(
        '--save_crd_diag',
        action='store_true',
        help='save FedCRD diagnostic CSV to result/crd_rq1_diag; default disabled'
    )

    # Binary threshold tuning (used by fed_main/Fed_CRD_Tuned.py).
    # This is useful when argmax gives high precision but lower recall.
    parser.add_argument(
        '--binary_threshold_tune',
        action='store_true',
        help='enable threshold search on validation set and evaluate test set with selected threshold'
    )

    parser.add_argument(
        '--binary_threshold_grid',
        type=str,
        default='0.50',
        help='comma-separated threshold candidates for positive-class probability in binary classification; default fixed at 0.50 for fair comparison'
    )

    # FedELC Arguments
    parser.add_argument('--epoch_of_stage1', type=int, default=20, help='number of epochs for stage 1')
    parser.add_argument('--lambda_pencil', type=float, default=1000, help='lamda for pencil loss')
    parser.add_argument('--alpha_pencil', type=float, default=0.5, help='alpha for pencil loss')
    parser.add_argument('--beta_pencil', type=float, default=0.2, help='beta for pencil loss')
    parser.add_argument('--K_pencil', type=int, default=10, help='number of pencils')

    # FedDSHAR Arguments
    parser.add_argument('--dshar_warmup_epoch', type=int, default=10, help='warmup epochs before dual-strategy training')
    parser.add_argument('--dshar_split_ratio', type=float, default=0.3, help='ratio of noisy subset split by MR confidence')
    parser.add_argument('--dshar_mr_refresh_interval', type=int, default=1, help='refresh interval of MR split')
    parser.add_argument('--dshar_aug_noise_std', type=float, default=0.02, help='gaussian noise std for clean augmentation')
    parser.add_argument('--dshar_aug_mask_ratio', type=float, default=0.1, help='feature mask ratio for clean augmentation')
    parser.add_argument('--dshar_lambda_div', type=float, default=0.1, help='weight for diversity regularization on clean subset')
    parser.add_argument('--dshar_pseudo_threshold', type=float, default=0.8, help='confidence threshold for pseudo-label training')
    parser.add_argument('--dshar_ema_beta', type=float, default=0.99, help='EMA momentum for teacher model')
    parser.add_argument('--dshar_w_clean', type=float, default=1.0, help='weight for clean-branch loss')
    parser.add_argument('--dshar_w_noisy', type=float, default=1.0, help='weight for noisy-branch loss')
    parser.add_argument('--dshar_la_tau', type=float, default=1.0, help='temperature scale for logit adjustment')
    parser.add_argument('--dshar_early_stop_patience', type=int, default=10, help='early stopping patience on validation F1; <=0 disables')
    parser.add_argument('--dshar_early_stop_min_delta', type=float, default=1e-4, help='minimum F1 improvement to reset early stopping counter')

    args = parser.parse_args()
    set_global_seed(args.seed)
    return args
