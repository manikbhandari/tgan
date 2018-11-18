from paraphraser import *

def get_args(config_file, args):
    name2val                 = json.load(open(config_file))

    name2val['mode']            = 'decode'
    name2val['use_beam_search'] = True
    name2val['restore']         = True
    name2val['use_dropout']     = False
    name2val['dtype']           = tf.float32
    name2val['out_file']        = args.out_file

    named_tuple_constructor  = namedtuple('params', sorted(name2val))
    args                     = named_tuple_constructor(**name2val)
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WORD GCN')
    parser.add_argument('-name',     dest="name",     default='test', help='')
    parser.add_argument('-gpu',      dest="gpu",      default='0',    help='')
    parser.add_argument('-seed',     dest="seed",     default=1234,   type=int, help='')
    parser.add_argument('-out_file', dest="out_file", default='output.txt',     help='')
    args   = parser.parse_args()

    args   = get_args(os.path.join('./checkpoints', args.name, 'params'), args)


    tf.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_gpu(args.gpu)

    model = Paraphraser(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.operation_timeout_in_ms  = 60000

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tf.tables_initializer().run()

        model.fit(sess)

    print('Model Decoded Successfully!!')
