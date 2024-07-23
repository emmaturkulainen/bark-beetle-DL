import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)
    parser.add_argument('--datadir', default='data', help='Data directory path.')
    parser.add_argument('--datatype', default='rgb', help='Type of image data. Options rgb, msi or hsi')
    parser.add_argument('--model', default='CNN2D', help='Used deep learning model. Default=CNN2D')
    parser.add_argument('--weights', default='', help='Path to pretrained model weights. Empty string means model weights will be initialized randomly')
    parser.add_argument('--batch-size', default=12, type=int, help='Batch size.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', default=0.00001, help='Initial learning rate')
    parser.add_argument('--weight_decay', default=0.01, help='weight decay used during training')
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu.')
    parser.add_argument('--exp', default='', help='Name of experiment')
    parser.add_argument('--output_dir', default='outputs', help='Output dir for results.')
    parser.add_argument('--optimizer_output_dir', default='optimizer_outputs', help='Output dir for optimizer results.')



    parser.add_argument(
                "opts",
                help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
                """.strip(),
                default=None,
                nargs=argparse.REMAINDER,
            )
    
    args = parser.parse_args()
    return args