
from neuralop.data.datasets import CarCFDDataset, DictDataset
from torch.utils.data import DataLoader

def get_data(config):
    # Load CFD body data
    data_module = CarCFDDataset(
        root_dir=config.data.root,
        query_res=[config.data.sdf_query_resolution] * 3,
        n_train=config.data.n_train,
        n_test=config.data.n_test,
        download=config.data.download,
    )

    # train_loader = data_module.train_loader(batch_size=1, shuffle=True)
    # test_loader = data_module.test_loader(batch_size=1, shuffle=False)

    solution_train_db = DictDataset(
        data_module.data[0 : config.data.n_train_solution], data_module.constant
    )
    solution_train_loader = DataLoader(
        solution_train_db,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
    )
    

    residual_train_db = DictDataset(
        data_module.data[
            config.data.n_train_solution : config.data.n_train_solution
            + config.data.n_train_residual
        ],
        data_module.constant,
    )
    residual_train_loader_unprocessed = DataLoader(
        residual_train_db,
        batch_size=1,
        shuffle=True,
    )
    

    residual_calib_db = DictDataset(
        data_module.data[
            config.data.n_train_solution
            + config.data.n_train_residual : config.data.n_train_solution
            + config.data.n_train_residual
            + config.data.n_calib_residual
        ],
        data_module.constant,
    )
    test_db = DictDataset(
        data_module.data[
            config.data.n_train_solution
            + config.data.n_train_residual
            + config.data.n_calib_residual :
        ],
        data_module.constant,
    )
    test_loader = DataLoader(test_db, shuffle=False, batch_size=1)

    return (
        data_module,
        solution_train_db,
        solution_train_loader,
        residual_train_db,
        residual_train_loader_unprocessed,
        residual_calib_db,
        test_db,
        test_loader,
    )