from dataset import Dataset
d = Dataset()
# d.compute_frame_feature()

count = 0
for x, y in d.generator(128, 'train'):
    assert (x[1].shape == (128, d.max_question_len))
    assert (x[0].shape == (128, d.max_video_len, d.frame_size))
    assert (y.shape == (128, d.answer_size))
    # print(count)
    count += 1
    # print(y.sum())
    if y.sum() > 128 * 3 or y.sum() < 128:
        print('error!!!')
