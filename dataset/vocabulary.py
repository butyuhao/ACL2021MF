import json

class T5CopyVocabulary(object):

    def __init__(self, vocab_path, tokenizer, sep=','):
        with open(vocab_path) as out:
            self.d_to_w_group = {} # key:词组的行 value:[(word1, id1), (word2, id2), ...]
            self.i_to_w = {} # id to word
            self.w_to_i = {} # word to id
            self.i_to_cls = {} # word id to cls id. cls id其实就是词组所在的行号
            self.id_to_category = {} # 行号 到 词组中的第一个词
            self.word_to_category_id = {} # 词组中的第一个词 到 行号
            for idx, line in enumerate(out):
                items = line.strip().split(sep)
                self.d_to_w_group[idx] = []
                for w in items:
                    w = w.lower()
                    assert len(w) > 0, "empty line %s" % line.strip()
                    fg_index = len(self.i_to_w)
                    self.d_to_w_group[idx].append((w, fg_index))
                    self.i_to_w[fg_index] = w
                    self.w_to_i[w] = fg_index
                    self.i_to_cls[fg_index] = idx
                self.id_to_category[len(self.id_to_category)] = items[0] # 用一组词的第一个来代表整组词语，key是id，value是一组词中的第一个。
                self.word_to_category_id[items[0]] = len(self.word_to_category_id)
            self.detection_size = len(self.id_to_category)

        self.token_fg_w = {} # word id 到 word的tokens
        for (fg_index, w) in self.i_to_w.items():
            token_word = tokenizer(w, return_tensors="np")['input_ids'][0, :-1].tolist()
            self.token_fg_w[fg_index] = token_word

        self.token_class = {} # 行号 到 词组第一个词的tokens
        for cls_index, w in self.id_to_category.items():
            token_word = tokenizer(w, return_tensors="np")['input_ids'][0, :-1].tolist()
            self.token_class[cls_index] = token_word

    def get_detection_size(self):
        return self.detection_size

    def get_fg_size(self):
        return len(self.i_to_w)

    def get_category(self):
        return self.id_to_category


