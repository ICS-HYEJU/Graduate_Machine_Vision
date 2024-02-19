def get_config_dict():
    dataset_info = dict(
        name='wdm',
        train_path='/storage/hjchoi/sample_train/',
        val_path='/storage/hjchoi/sample_train/',
        height=52,
        width=52,
        channel=3,
        batch_size=10,
        num_workers=0,
    )

    path = dict(
        save_base_path='runs'
    )

    model = dict(
        name='vgg'
    )

    solver = dict(
        name='rmsprop',
        gpu_id=0,
        lr0=1e-4,
        momentum=0.937,
        weight_decay=5e-4,
        max_epoch=50,
    )

    scheduler = dict(
        name='cosine',
        lr_max=solver['lr0'],
        lr_min=1e-8
    )

    # Merge all information into a dictionary variable
    config = dict(
        dataset=dataset_info,
        path=path,
        model=model,
        solver=solver,
        scheduler=scheduler
    )

    return config

