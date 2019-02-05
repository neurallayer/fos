import numpy as np


def get_normalization(dataloader, max_iter=None, feature_dim=1):
    '''Calculates the mean and standard deviation for a Tensors.

       The dataloader should return either just the batch of image or
       a tuple/list of which the first element is the image.

       The output can be used for normalizing the images
       before feeding them to a neural network.

       Args:
           dataloader: The datalaoder you want to use to get the images
           max_iter: Limit the number of iterations. If None, all the iterations in the
             dataloader will be used. It should be noted that if a batch has more then one sample,
             the actual number of samples equals: samples = max_iter * batch_size
           feature_dim: which dimension has the features to normalize.

        Examples:
            n = get_normalization(image_loader, 1000, 1) #Image tensors with the format  NxCxWxH (PyTorch format): 1
        '''

    s = 0.
    m = 0.
    step = 0
    first = True

    for data in dataloader:
        step += 1

        if max_iter is not None:
            if step > max_iter:
                break

        if not torch.is_tensor(data):
            data = data[0]

        data = data.cpu().numpy()

        if first:
            axis = list(range(len(data.shape)))
            del axis[feature_dim]
            axis = tuple(axis)
            first = False

        s += np.std(data, axis=axis)
        m += np.mean(data, axis=axis)

    return {"mean": m / samples, "std": s / samples}
