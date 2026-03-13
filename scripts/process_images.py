import numpy as np
from skimage import io
from multiprocess import Pool
import os

def process_image_chunk(chunk):
    extract_features = chunk[0][2]
    results = []
    for i in range(len(chunk)):
        image = io.imread(chunk[i][0])
        results.append((chunk[i][1], extract_features(image)))
    return results

def process_images_in_parallel(path, imageList, extract_features, num_processes=8, batch_size=500, save_path='features.npy'):
    first_image = io.imread(path + '/' + imageList[0])
    feature_size = len(extract_features(first_image))
    batch_files = []

    for batch_start in range(0, len(imageList), batch_size):
        batch = imageList[batch_start:batch_start + batch_size]
        batch_features = np.zeros((len(batch), feature_size))

        pairs = [(path + '/' + batch[i], i, extract_features) for i in range(len(batch))]
        chunk_size = len(pairs) // num_processes
        chunks = [pairs[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]
        if len(pairs) % num_processes:
            chunks.append(pairs[num_processes*chunk_size:])

        with Pool(processes=num_processes) as pool:
            for chunk_result in pool.map(process_image_chunk, chunks):
                for idx, feat in chunk_result:
                    batch_features[idx, :] = feat

        batch_file = save_path.replace('.npy', '_batch' + str(batch_start) + '.npy')
        np.save(batch_file, batch_features)
        batch_files.append(batch_file)
        del batch_features

        print('\rProcessed ' + str(min(batch_start + batch_size, len(imageList))) + ' of ' + str(len(imageList)), end='')

    print('\nCombining batches...')
    temp_path = save_path.replace('.npy', '_temp.dat')
    features = np.memmap(temp_path, dtype='float64', mode='w+', shape=(len(imageList), feature_size))

    idx = 0
    for f in batch_files:
        batch = np.load(f)
        features[idx:idx + len(batch)] = batch
        idx += len(batch)
        del batch

    assert features.shape[0] == len(imageList), "Missing images in final features!"
    np.save(save_path, np.array(features))
    del features
    os.remove(temp_path)

    print('Features matrix shape: ', np.load(save_path).shape)
    for f in batch_files:
        os.remove(f)

    return np.load(save_path)