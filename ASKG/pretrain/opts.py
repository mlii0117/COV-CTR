import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--data_dir', type=str, default='', help='')
    parser.add_argument('--cnn_weight', type=str)
    parser.add_argument('--caption_weight', type=str, default='', help='')

    parser.add_argument('--mGPUs', type=bool, default=False, help='whether use multiple GPUs')

    # CNN Model settings
    parser.add_argument('--att_size', default=7, type=int, help='14x14 or 7x7')

    # Caption Model settings
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                        help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--topic_size', type=int, default=512,
                        help='the topic size of each sentence in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                        help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                        help='1024 for densenet, 2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                        help='1024 for densenet, 2048 for resnet, 512 for vgg')
    parser.add_argument('--seq_sent_length', type=int, default=10, help='max length of sequence.')
    parser.add_argument('--seq_word_length', type=int, default=20)
    parser.add_argument('--n_ann', type=int, default=49, help='')
    parser.add_argument('--top_n_keys', type=int, default=10, help='')

    # Optimization: General
    parser.add_argument('--dataset', type=str, default='coco',
                        help='on which dataset are we going to train? coco|iuxray')
    parser.add_argument('--max_epochs', type=int, default=-1, help='number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=16, help='minibatch size')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='minibatch size of val')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--finetune_cnn_after', type=int, default=-1,
                        help='After what epoch do we start fintuning the CNN? (-1 = disable, never finetune; 0 = finetune from start)')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='choose medical terms')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                        help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                        help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay')

    # Optimization: for the CNN Model
    parser.add_argument('--cnn_optim', type=str, default='adam',
                        help='optimization to use for CNN')
    parser.add_argument('--cnn_learning_rate', type=float, default=1e-5)
    parser.add_argument('--cnn_weight_decay', type=float, default=0,
                        help='weight_decay')
    parser.add_argument('--cnn_optim_alpha', type=float, default=0.8,
                        help='cnn alpha for adam')
    parser.add_argument('--cnn_optim_beta', type=float, default=0.999,
                        help='beta used for adam')

    # Scheduled Sampling
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                        help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                        help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=3200,
                        help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                        help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                        help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=0,
                        help='Do we load previous best score when resuming training.')
    parser.add_argument('--cbs', type=bool, default=False,
                        help='whether use constraint beam search.')
    parser.add_argument('--cbs_tag_size', type=int, default=3,
                        help='whether use constraint beam search.')
    parser.add_argument('--cbs_mode', type=str, default='all',
                        help='which cbs mode to use in the decoding stage. cbs_mode: all|unique|novel')
    parser.add_argument('--val_every_epoch', type=int, default=1,
                        help='')

    parser.add_argument('--encoder_path', type=str, default='save',
                        help='directory to store checkpointed encoder models')
    parser.add_argument('--decoder_path', type=str, default='save',
                        help='directory to store checkpointed decoder models')

    # misc
    parser.add_argument('--id', type=str, default='',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                        help='if true then use 80k, else use 110k')

    # classify
    parser.add_argument('--model_dir', type=str, default='./chexnet_models',
                        help='')
    parser.add_argument('--shot', type=str, default='1st',
                        help='')
    parser.add_argument('--log_step', type=int, default=1,
                        help='')
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='')
    parser.add_argument('--num_medterm', type=int, default=1,
                        help='')
    parser.add_argument('--num_disease', type=int, default=1,
                        help='')
    parser.add_argument('--num_termclass', type=int, default=1,
                        help='')

    # self-attention
    parser.add_argument('--pooling', type=str, default='mean',
                        help='sentence feature pooling')
    parser.add_argument('--attention_unit', type=int, default=350,
                        help='number of attention unit')
    parser.add_argument('--attention_hops', type=int, default=1,
                        help='number of attention hops, for multi-hop attention model')

    # transformer
    parser.add_argument('--d_model', type=int, default=512,
                        help='Dimensionality of the embeddings and hidden states.')
    # parser.add_argument('--d_ff', type=int, default=2048,
    #                     help='Dimensionality of PositionwiseFeedForward')
    # parser.add_argument('--n_layers', type=int, default=6,
    #                     help='Number of hidden layers in the Transformer.')
    # parser.add_argument('--n_heads', type=int, default=8,
    #                     help='Number of attention heads for each attention layer in the Transformer.')

    # transformer optim
    parser.add_argument('--factor', type=int, default=1,
                        help='Scale')
    parser.add_argument('--warmup', type=int, default=4000,
                        help='Warmup training steps')
    parser.add_argument('--report_weight', type=str, default='', help='report pretrained gpt path')

    # sample
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--sample', action='store_true')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args