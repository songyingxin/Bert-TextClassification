from main import main
from BertOrigin import args

if __name__ == "__main__":

    data_dir = "/home/songyingxin/datasets/cnews"
    output_dir = ".output"
    cache_dir = ".cache"
    log_dir = ".cnews_log"

    bert_vocab_file = "/home/songyingxin/datasets/pytorch-bert/vocabs/bert-base-chinese-vocab.txt"
    bert_model_dir = "/home/songyingxin/datasets/pytorch-bert/bert-base-chinese"

    from Processors.NewsProcessor import NewsProcessor
    main(args.get_args(data_dir, output_dir,
                       cache_dir, bert_vocab_file, bert_model_dir, log_dir), NewsProcessor)
