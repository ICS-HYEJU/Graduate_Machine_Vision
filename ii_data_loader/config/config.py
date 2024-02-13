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
        name='None'
    )

    solver = dict(
        name='sgd',
        gpu_id=0,
        lr0=1e-4,
        momentum=0.937,
        max_epoch=50,
    )

    # Merge all information into a dictionary variable
    config = dict(
        dataset=dataset_info,
        path=path,
        model=model,
        solver=solver,
    )

    return config

