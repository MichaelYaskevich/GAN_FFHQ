from scr.interface import handle_train_cmd, handle_eval_cmd, make_parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    if args.name == 'train':
        handle_train_cmd(args.image_size[0], args.epochs[0], args.lr[0], args.dir_to_save[0])
    elif args.name == 'eval':
        handle_eval_cmd(args.image_size[0], args.images_count[0], args.dir_to_save[0], args.path_to_weights[0])
