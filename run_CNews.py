# coding=utf-8
from main import main
from BertOrigin import args

if __name__ == "__main__":

    data_dir = "/search/hadoop02/suanfa/songyingxin/data/cnews"
    output_dir = ".cnews_output"
    cache_dir = ".cnews_cache"
    log_dir = ".cnews_log"

    bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-chinese-vocab.txt"
    bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-chinese"

    from Processors.NewsProcessor import NewsProcessor
    main(args.get_args(data_dir, output_dir,
                       cache_dir, bert_vocab_file, bert_model_dir, log_dir), NewsProcessor)
