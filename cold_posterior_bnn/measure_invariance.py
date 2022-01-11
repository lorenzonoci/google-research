import numpy as np
from cold_posterior_bnn.datasets import load_cifar10
import tensorflow as tf
from cold_posterior_bnn.core import priorfactory
from cold_posterior_bnn import models
from tqdm import tqdm
#from cold_posterior_bnn.temperature_predict import numpy_augm, load_cifar10_numpy, optimal_temperature
n_seeds = 10
n_augmentations = 200
init_random_seed = 124
# Create a generator.
rng = tf.random.Generator.from_seed(init_random_seed, alg='philox')
tf.random.set_seed(init_random_seed)
np.random.seed(init_random_seed)
max_datapoints = 100


baseline = {
    "name": "baseline",
    "use_gconv": False,
    "n_filters": 16,
    "strides": 2,
    "final_dense": True,
    "double_n_filters": True,
    "padding": 'same',
    "use_blur_pool": False,
    "depth_multiplier": None
}

gcnn = {
    "name": "gcnn_not_inv",
    "use_gconv": True,
    "n_filters": 16,
    "strides": 2,
    "final_dense": True,
    "double_n_filters": True,
    "padding": "same",
    "use_blur_pool": False,
    "depth_multiplier": None
}

models_config = [baseline, gcnn]


def tot_variation(probs1, probs2):
    return np.abs(probs1 - probs2).sum(axis=1).mean(axis=0)


# other fixed parameters
activation = "relu"
# this parameter is useless, the network is not initialized according to the prior. The prior is only used as a
# regularizer
pfac = priorfactory.GaussianPriorFactory(prior_stddev=1.0,
                                             weight=1.0)
num_classes = 10
batch_size=max_datapoints

orig_train_ds = load_cifar10(split="train", data_augmentation=True, subsample_n=max_datapoints, random_rotation=True)
orig_train_ds = orig_train_ds.batch(batch_size)

for model_conf in models_config:
    print("model: {}".format(model_conf["name"]))
    #for seed in tqdm(np.arange(n_seeds)):
        #tf.random.set_seed(seed)
        #np.random.seed(seed)

        #augm_train_ds = load_cifar10(split="train", data_augmentation=True, p4m_augm=True)
    
    model = models.build_resnet_v1(
            input_shape=(32, 32, 3),
            depth=20,
            num_classes=num_classes,
            pfac=pfac,
            use_internal_bias=True,
            use_gconv=model_conf["use_gconv"])
        #if seed == 0: # print model summary at first iteration
        #    model.summary()
    augm_preds = []    
    for i in tqdm(range(n_augmentations)):
        p = model.predict(orig_train_ds)
        augm_preds.append(p)
    augm_preds = np.stack(augm_preds, axis=0)
    # ts = []
    # for i in range(max_datapoints):
    #     T = optimal_temperature(augm_preds[i], n_augmentations, normalize=False)
    #     ts.append(T.numpy())
    tot_vars = []
    for i in range(n_augmentations-1):
        tot_var = tot_variation(probs1=augm_preds[0, :, :], probs2=augm_preds[i+1, :, :])
        tot_vars.append(tot_var)

    #print("optimal temperature stats for model {}. mean: {}, std: {}".format(model_conf["name"], np.mean(ts), np.std(ts)))
    print("invariance for model {}. mean: {}, std: {}".format(model_conf["name"], np.mean(tot_vars), np.std(tot_vars)))
    #print("Effective number of samples: {}".format(max_datapoints*(1 + n_augmentations - np.mean(ts))))
    #print("Optimal temperature (cold): {}".format(np.mean(ts) / n_augmentations))
