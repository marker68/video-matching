import numpy as np


def rank(ref, dict):
    ids = list(range(0, len(dict)))
    ids.sort(key=lambda i: np.linalg.norm(ref-dict[i]))
    return ids


def rank_refine(ref, dict, k):
    ids = rank(ref, dict)
    for i in range(0,k):
        j = ids.pop(0)
        ids.insert(len(ids), j)
    return ids


def accuracy(pred, gt, k, r):
    a = 0
    for i in range(0, len(pred)):
        p = pred[i]
        ranks = rank_refine(p, gt, r)
        if i in ranks[0:k]: a += 1
    return a


def mid(pred, gt, r):
    a = 0.0
    for i in range(0, len(pred)):
        p = pred[i]
        ranks = rank_refine(p, gt, r)
        for j in range(0, len(ranks)):
            if ranks[j] == i: a += 1.0/(j+1)
    return a / len(pred)


def test_with_gt(video_model, X_test, Y_test, r):
    output = video_model.predict(X_test)
    print("Accuracy:")
    print("@1 = " + str(accuracy(output, Y_test, 1, r)))
    print("@5 = " + str(accuracy(output, Y_test, 5, r)))
    print("@10 = " + str(accuracy(output, Y_test, 10, r)))
    print("MID = " + str(mid(output,Y_test, r)))


def test_without_gt(video_model, X_test, Y_test, r, output_file):
    output = video_model.predict(X_test)
    # now matching with Y_test
    n = len(X_test)
    ranks = []
    for i in range(0, n):
        out = output[i]
        ranks.append(rank_refine(out, Y_test, r))
    np.save(open(output_file, 'w'), ranks)

if __name__ == '__main__':
    ref = np.random.randn(10)
    dict = np.random.random_sample(size=(10,10))
    ids = rank(ref,dict)
    print(ids)
    dists = []
    for d in dict:
        dists.append(np.linalg.norm(ref-d))
    print(dists)

    pred = np.random.random_sample(size=(10,10))

    print(accuracy(pred,dict,5))
