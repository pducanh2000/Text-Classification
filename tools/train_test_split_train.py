import argparse


def get_args_parser():
    parser = argparse.ArgumentParser("VN_team_3_Danny", add_help=False)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VN_team_3_Danny', parents=[get_args_parser()])
    args = parser.parse_args()
