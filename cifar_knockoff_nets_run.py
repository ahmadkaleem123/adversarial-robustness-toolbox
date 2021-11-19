from cifar_knockoff_nets import TestKnockoffNets

if __name__ == "__main__":
    dataset = 'cifar10'

    if dataset == 'cifar10':
        train = True
    elif dataset == 'mnist':
        train = True
    else:
        raise Exception(f"Unknown dataset: {dataset}")
    if train:
        load_init = False
    else:
        load_init = True

    knockoff = TestKnockoffNets(
        train=train,
        random=True,
        adaptive=True,
        dataset=dataset,
        load_init=load_init)
    knockoff.runknockoff()
