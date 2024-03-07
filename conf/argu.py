
def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')
    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=768, help='')
    parser.add_argument('--embed_dim', type=int, default=768, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--attention_dim', type=int, default=768, help='')
    parser.add_argument('--max_length', type=int, default=192, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    parser.add_argument('--channel', type=int, default=11, help='')
    parser.add_argument('--reduction', type=int, default=11, help='')
    return parser