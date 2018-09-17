import pickle


class AnswerMapping(object):
    def __init__(self, ans2idx=None, idx2ans=None):
        self.ans2idx = ans2idx or {}
        self.idx2ans = idx2ans or []

    def __len__(self):
        return len(self.idx2ans)

    def tokenize(self, ans, add_token=False):
        tokens = []
        if add_token:
            for a in ans:
                tokens.append(self._add_token(a))
        else:
            for a in ans:
                tokens.append(self.ans2idx[a] if a in self.ans2idx else -1)
        return tokens

    def dump_to_file(self, path):
        with open(path, 'wb') as f:
            pickle.dump([self.ans2idx, self.idx2ans], f)
            print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        with open(path, 'rb') as f:
            ans2idx, idx2ans = pickle.load(f)
            d = cls(ans2idx, idx2ans)
        return d

    def _add_token(self, ans):
        if ans not in self.ans2idx:
            self.idx2ans.append(ans)
            self.ans2idx[ans] = len(self.idx2ans) - 1
        return self.ans2idx[ans]
