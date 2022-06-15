import torch

class KNearestNeighborsClassifier:

    def __init__(self, n_neighbors=[1,5,10,15,20,25,30], batch_size=0):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

    def predict_batch(self, train_x, train_y, test_x_batch):
        # compute predictions for different k neighbors
        scores = torch.matmul(test_x_batch, train_x.t())
        top_indices = torch.topk(scores, self.n_neighbors[-1], dim=1)[1]
        pred_batch = [[] for k in self.n_neighbors]
        for i in range(len(test_x_batch)):
            top_y = train_y[top_indices[i]]
            for j in range(len(self.n_neighbors)):
                freq_y, freq_num = torch.unique(top_y[:self.n_neighbors[j]], return_counts=True)
                pred_batch[j].append(freq_y[torch.argmax(freq_num)].unsqueeze(0))
        pred_all = torch.stack([torch.cat(pred, 0) for pred in pred_batch], 0)
        return pred_all

    def predict(self, train_x, train_y, test_x):
        test_y = []
        if self.batch_size==0:
            batch_size=len(test_x)
        else:
            batch_size=self.batch_size
        index = 0
        steps = len(test_x)//batch_size
        for i in range(1, steps+1):
            test_batch = test_x[index:index+i*batch_size]
            index = index+i*batch_size
            test_y.append(self.predict_batch(train_x, train_y, test_batch))
        if index<len(test_x)-1:
            test_y.append(self.predict_batch(train_x, train_y, test_x[index:]))

        return torch.cat(test_y, 1)
