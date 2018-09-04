import cv2, random, numpy as np
from keras.models import Model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import cPickle
import os

class Dictionary(object):
    def __init__(self, ans2idx=None, idx2ans=None):
        if ans2idx is None:
            ans2idx = {}
        if idx2ans is None:
            idx2ans = []
        self.ans2idx = ans2idx
        self.idx2ans = idx2ans

    @property
    def ntoken(self):
        return len(self.ans2idx)

    @property
    def padding_idx(self):
        return len(self.ans2idx)

    def tokenize(self, ans, add_ans=False):
        tokens = []
        if add_ans:
            for a in ans:
                tokens.append(self.add_ans(a))
        else:
            for a in ans:
                tokens.append(self.ans2idx(a))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.ans2idx, self.idx2ans], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        ans2idx, idx2ans = cPickle.load(open(path, 'rb'))
        d = cls(ans2idx, idx2ans)
        return d

    def add_ans(self, ans):
        if ans not in self.ans2idx:
            self.idx2ans.append(ans)
            self.ans2idx[ans] = len(self.idx2ans) - 1
        return self.ans2idx[ans]

    def __len__(self):
        return len(self.idx2ans)

class Dataset():
    # phases = ['train', 'test', 'val']
    phases = ['train', 'test']
	base_dir = 'VQADatasetA_20180815'
	vocabulary_size = 10000

	def __init__(self):
		self.dict = Dictionary()
		vid, questions, answers = self.preprocess_text(phases[0])
		self.dict.tokenize(answers, True)
		self.tokenizer = Tokenizer(vocabulary_size)
		self.tokenize.fit_on_texts(questions)

		self.max_video_len = 100
		self.max_question_len = 20
		# the feature map size of each frame
		self.frame_size = 2048

	# split text data into three parts: video_id, questions and answers.
	def preprocess_text(self, phase):
		assert(phase in self.phases)

		vid = []
		questions = []
		answers = []
		fs = open(base_dir+'/'+phase+'.txt')
		for line in fs.readlines():
			parts = line.split(',')
			vid.append(parts[0])
			qs = 1
			for i in range(5):
				questions.append(parts[i])
				answers.extend(parts[i:i+3])
		return vid, questions, answers

	# the generator function for model's input
	def generator(self, batch_size, phase):
		assert(phase in self.phases)

		vid, questions, answers = self.preprocess_text(phase)
		questions = self.tokenizer.text_to_sequences(questions)
		answers = self.dict.tokenize(answers)
		answer_size = max(answers)
		one_hot_answers = [to_categorical(answers[i],answer_size)+\
						   to_categorical(answers[i+1],answer_size)+\
						   to_categorical(answers[i+2],answer_size)\
						    for i in range(0,len(answers),3)]
		
		inds = [i for i in range(len(vid)*5)]
		assert(len(inds) == len(questions))
		assert(len(one_hot_answers) == len(questions))
		while True:
			if phase == 'train':
				random.shuffle(inds)
			count = 0
			while(count<len(inds)):
				X_video = np.zeros((batch_size, self.max_video_len, self.frame_size))
				X_question = np.zeros((batch_size, self.max_question_len), dytpe=np.int32)
				Y = np.zeros((batch_size, answer_size), dytpe=np.int32)
				i = 0
				j = 0
				while i < batch_size:
					try:
						# load feature map of each frame
						cur_video = np.load(self.feature_dir+'/'+vid[(count+j)%len(inds)//5]+'_resnet.npy')
						X_video[i, :cur_video.shape[0]] = cur_video[:self.max_video_len]
						q = questions[inds[(count+j)%len(inds)]]
						X_question[i, :len(q)] = q
						i += 1
						Y[i,:] = one_hot_answers[(count+j)%len(inds)]
					except:
						print('generator error')
				j += 1

			yield [X_video, X_question], Y
			count += j

	# extract video frames' feature
	def compute_frame_feature(self):
		batch_size = 100
		interval = 5
		res_model = ResNet50(weights='imagenet')
		model = Model(input=res_model.input, output=res_model.get_layer('avg_pool').output)

		self.feature_dir = self.base_dir+'/feature'

		for phase in self.phases:
			vid_dir = self.base_dir+'/'+phase
			for v in os.listdir(vid_dir):
				video = cv2.VideoCapture(vid_dir+'/'+str(v))
				vid_descriptors = np.zeros((999*batch_size, 2048))
				frame_count = 0
				stop = false
				for b in range(999):
					batch = np.zeros((batch_size))
					for t in range(batch_size):
						retval, frame = video.read()
						if retval:
							batch[t] = cv2.resize(frame, (224,224)).transpose((2,0,1))
							frame_count+=1
						else:
							stop = True
							break
						for _ in range(interval - 1):
							video.read()
					batch[:, 0] -= 103.939
	                batch[:, 1] -= 116.779
	                batch[:, 2] -= 123.68
	                vid_descriptors[b*batch_size:(b+1)*batch_size] = model.predict_on_batch(batch).reshape(batch.shape[0],-1)
					if stop:
						break
				np.save(self.feature_dir+'/'+v.split('.')[0]+\
						'_resnet.npy', vid_descriptors[:frame_count])

		