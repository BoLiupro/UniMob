from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()

    # =======================General args========================
    parser.add_argument('--equal_time_slot', default=True, help='Use equal time slot for trajectory')
    parser.add_argument('--time_slot', type=float, default=0.5, help='time slot. Unit: hour')
    parser.add_argument('--city', type=str, default='Guangzhou',choices=['Guangzhou','Shenzhen','Changsha'], help='city name')
    parser.add_argument('--city_chinese_str', type=str, default='长沙',choices=['广州','深圳','长沙'], help='process city name in start position str of text')
    parser.add_argument('--task', type=str, default='3', help='task identifier for saving model')
    parser.add_argument('--num_month', type=int, default=1, help='how many months starting from Jan of data to use')
    parser.add_argument('--K', type=int, default=48, help='trajectory length')
    parser.add_argument('--F', type=int, default=14, help='condition feature dim')
    parser.add_argument('--vocab_size', type=int, default=890, help='Number of regions=vocab size')
    parser.add_argument('--checkpt_path', type=str, default='checkpt_ETS', help='checkpoint path to save model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--poi_dim', type=int, default=14, help='Input dimension of poi distribution')
    parser.add_argument('--pos_dim', type=int, default=4, help='Input dimension of position distribution')
    parser.add_argument('--time_dim', type=int, default=4, help='Input dimension of time distribution')
    # =======================Encoder args========================
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dim = Number of vec feature dimension input to encoder , shoud be divisible by num_head')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in Transformer Encoder')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers in Transformer Encoder')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='Masking probability for MLM')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for ST_Encoder')
    parser.add_argument('--kernel_set', type=list[int], default=[1, 2, 3], help='kernel set for ST_Encoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in Transformer Encoder')
    parser.add_argument('--Encoder_epoch', type=int, default=80, help='Number of epoch for ST_Encoder training')
    parser.add_argument('--Encoder_batch_size', type=int, default=256, help='batch size for ST_Encoder training')
    # =======================Generator args========================
    parser.add_argument('--generator_epoch', type=int, default=5, help='Number of epoch for TravDiT training')
    parser.add_argument('--generator_batch_size', type=int, default=256, help='batch size')
    # =======================Planer args============================
    parser.add_argument('--unique_poi_types', 
                        type=list[str], 
                        default=['Transportation Facilities','Leisure & Entertainment','Companies & Enterprises','Healthcare',
                            'Commercial & Residential','Tourist Attractions','Automotive','Life Services','Science & Education & Culture',
                            'Shopping & Consumer Goods','Sports & Fitness','Hotels & Accommodations','Financial Institutions','Dining & Cuisine'], 
                        help='poi types for LLM')
    parser.add_argument('--llm_batch_size', type=int, default=16, help='LLM model to use')
    parser.add_argument('--llm_epoch', type=int, default=2, help='Number of epoch for LLM training')
    # =======================Alignment args========================
    parser.add_argument('--alignment_epoch', type=int, default=50, help='Number of epoch for Alignment training')
    parser.add_argument('--alignment_batch_size', type=int, default = 8, help='batch size for Alignment training')


    parser.add_argument('--ignore_unknown_args', action='store_true', help='Ignore unknown command line arguments')

    args, unknown = parser.parse_known_args()

    # if len(unknown)!= 0 and not args.ignore_unknown_args:
    #     print("some unrecognised arguments {}".format(unknown))
    #     raise SystemExit

    return args