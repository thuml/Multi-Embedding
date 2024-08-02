import torch
import torch.utils.data
import time
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel, MultiPNNModel
from torchfm.model.xdfm import XDeepFM, MultiXDeepFM
from torchfm.model.fwfm import NFwFMModel, MultiNFwFMModel
from torchfm.model.dcnv2 import CrossNetworkV2Model
from torchfm.model.mwd import (
    DNNModel,
    MultiDNNModel,
)
from torchfm.model.awesome import (
    MultiDCNnew2,
    WeightNormAlignedMultiDCNnew2,
    MultiESingleIDCNv2,
    SpaceSimilarityRegularizedMultiDCNnew2,
    SingularValueRegularizedDCNv2,
)
from torchfm.model.rdcnv2 import RestrictedCrossNetworkV2Model
from torchfm.model.wrdcnv2 import WeightedRestrictedCrossNetworkV2Model, WeightedRestrictedMultiDCN
from torchfm.model.finalmlp import FinalMLP, MultiFinalMLP
from utils import CompleteLogger, AverageMeter, ProgressMeter, CriterionWithLoss, EarlyStopper


def get_dataset(name, path):
    if name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    print(field_dims)
    print(sum(field_dims))
    
    ########################################## Appendix  ########################################
    if name == 'space-similarity-regularized-mdcn-4x10-1e-3':
        return SpaceSimilarityRegularizedMultiDCNnew2(field_dims, embed_dims=[10]*4, num_layers=4, mlp_dims=(400, 400), dropout=0.2, reg_weight=1e-3)
    elif name == 'space-similarity-regularized-mdcn-4x10-1e-4':
        return SpaceSimilarityRegularizedMultiDCNnew2(field_dims, embed_dims=[10]*4, num_layers=4, mlp_dims=(400, 400), dropout=0.2, reg_weight=1e-4)
    elif name == 'space-similarity-regularized-mdcn-4x10-1e-5':
        return SpaceSimilarityRegularizedMultiDCNnew2(field_dims, embed_dims=[10]*4, num_layers=4, mlp_dims=(400, 400), dropout=0.2, reg_weight=1e-5)
    elif name == 'singular-value-regularized-dcn-40-1e-3':
        return SingularValueRegularizedDCNv2(field_dims, embed_dim=40, num_layers=4, mlp_dims=(400, 400), dropout=0.2, reg_weight=1e-3)
    elif name == 'singular-value-regularized-dcn-40-1e-4':
        return SingularValueRegularizedDCNv2(field_dims, embed_dim=40, num_layers=4, mlp_dims=(400, 400), dropout=0.2, reg_weight=1e-4)
    elif name == 'singular-value-regularized-dcn-40-1e-5':
        return SingularValueRegularizedDCNv2(field_dims, embed_dim=40, num_layers=4, mlp_dims=(400, 400), dropout=0.2, reg_weight=1e-5)
    elif name == "me-si-dcn-2x10":
        return MultiESingleIDCNv2(field_dims, embed_dims=[10]*2, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == "me-si-dcn-3x10":
        return MultiESingleIDCNv2(field_dims, embed_dims=[10]*3, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == "me-si-dcn-4x10":
        return MultiESingleIDCNv2(field_dims, embed_dims=[10]*4, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == "me-si-dcn-10x10":
        return MultiESingleIDCNv2(field_dims, embed_dims=[10]*10, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'rebuttal-restricted-weighted-mdcn-2x10':
        return WeightedRestrictedMultiDCN(field_dims, embed_dims=[10]*2, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'rebuttal-restricted-weighted-mdcn-3x10':
        return WeightedRestrictedMultiDCN(field_dims, embed_dims=[10]*3, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'rebuttal-restricted-weighted-mdcn-4x10':
        return WeightedRestrictedMultiDCN(field_dims, embed_dims=[10]*4, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'rebuttal-restricted-weighted-mdcn-10x10':
        return WeightedRestrictedMultiDCN(field_dims, embed_dims=[10]*10, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    ########################################## Appendix  ########################################

    ##############################
    #  Main experiments started. #
    ##############################
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=8)
    elif name == 'dcn-10':
        return CrossNetworkV2Model(field_dims, embed_dim=10, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'dcn-20':
        return CrossNetworkV2Model(field_dims, embed_dim=20, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'dcn-30':
        return CrossNetworkV2Model(field_dims, embed_dim=30, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'dcn-40':
        return CrossNetworkV2Model(field_dims, embed_dim=40, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'dcn-100':
        return CrossNetworkV2Model(field_dims, embed_dim=100, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    
    elif name == "mdcn-2x10":
        return MultiDCNnew2(field_dims, embed_dims=[10]*2, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == "mdcn-3x10":
        return MultiDCNnew2(field_dims, embed_dims=[10]*3, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == "mdcn-4x10":
        return MultiDCNnew2(field_dims, embed_dims=[10]*4, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == "mdcn-10x10":
        return MultiDCNnew2(field_dims, embed_dims=[10]*10, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    
    elif name == "weight-norm-aligned-mdcn-2x10":
        return WeightNormAlignedMultiDCNnew2(field_dims, embed_dims=[10]*2, num_layers=4, mlp_dims=(400, 400), dropout=0.2, reg_weight=1.0)
    elif name == "weight-norm-aligned-mdcn-3x10":
        return WeightNormAlignedMultiDCNnew2(field_dims, embed_dims=[10]*3, num_layers=4, mlp_dims=(400, 400), dropout=0.2, reg_weight=1.0)
    elif name == "weight-norm-aligned-mdcn-4x10":
        return WeightNormAlignedMultiDCNnew2(field_dims, embed_dims=[10]*4, num_layers=4, mlp_dims=(400, 400), dropout=0.2, reg_weight=1.0)
    elif name == "weight-norm-aligned-mdcn-10x10":
        return WeightNormAlignedMultiDCNnew2(field_dims, embed_dims=[10]*10, num_layers=4, mlp_dims=(400, 400), dropout=0.2, reg_weight=1.0)
    
    elif name == "dnn-10":
        return DNNModel(field_dims, embed_dim=10, mlp_dims=(400, 400), dropout=0.2)
    elif name == "dnn-20":
        return DNNModel(field_dims, embed_dim=20, mlp_dims=(400, 400), dropout=0.2)
    elif name == "dnn-30":
        return DNNModel(field_dims, embed_dim=30, mlp_dims=(400, 400), dropout=0.2)
    elif name == "dnn-40":
        return DNNModel(field_dims, embed_dim=40, mlp_dims=(400, 400), dropout=0.2)
    elif name == "dnn-100":
        return DNNModel(field_dims, embed_dim=100, mlp_dims=(400, 400), dropout=0.2)
    
    elif name == "mdnnW-2x10":
        return MultiDNNModel(field_dims, embed_dims=[10]*2, mlp_dims=(400, 400), dropout=0.2)
    elif name == "mdnnW-3x10":
        return MultiDNNModel(field_dims, embed_dims=[10]*3, mlp_dims=(400, 400), dropout=0.2)
    elif name == "mdnnW-4x10":
        return MultiDNNModel(field_dims, embed_dims=[10]*4, mlp_dims=(400, 400), dropout=0.2)
    elif name == "mdnnW-10x10":
        return MultiDNNModel(field_dims, embed_dims=[10]*10, mlp_dims=(400, 400), dropout=0.2)
    
    elif name == 'restricted-weighted-dcn-10':
        return WeightedRestrictedCrossNetworkV2Model(field_dims, embed_dim=10, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'restricted-weighted-dcn-20':
        return WeightedRestrictedCrossNetworkV2Model(field_dims, embed_dim=20, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'restricted-weighted-dcn-30':
        return WeightedRestrictedCrossNetworkV2Model(field_dims, embed_dim=30, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'restricted-weighted-dcn-40':
        return WeightedRestrictedCrossNetworkV2Model(field_dims, embed_dim=40, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    elif name == 'restricted-weighted-dcn-100':
        return WeightedRestrictedCrossNetworkV2Model(field_dims, embed_dim=100, num_layers=4, mlp_dims=(400, 400), dropout=0.2)
    
    elif name == 'ipnn-10':
        return ProductNeuralNetworkModel(field_dims, embed_dim=10, mlp_dims=(400, 400), method='inner', dropout=0.2)
    elif name == 'ipnn-20':
        return ProductNeuralNetworkModel(field_dims, embed_dim=10, mlp_dims=(400, 400), method='inner', dropout=0.2)
    elif name == 'ipnn-30':
        return ProductNeuralNetworkModel(field_dims, embed_dim=10, mlp_dims=(400, 400), method='inner', dropout=0.2)
    elif name == 'ipnn-40':
        return ProductNeuralNetworkModel(field_dims, embed_dim=10, mlp_dims=(400, 400), method='inner', dropout=0.2)
    elif name == 'ipnn-100':
        return ProductNeuralNetworkModel(field_dims, embed_dim=10, mlp_dims=(400, 400), method='inner', dropout=0.2)
    
    elif name == 'multi-ipnn-2x10':
        return MultiPNNModel(field_dims, embed_dims=[10]*2, mlp_dims=(400, 400), method='inner', dropout=0.2)
    elif name == 'multi-ipnn-3x10':
        return MultiPNNModel(field_dims, embed_dims=[10]*3, mlp_dims=(400, 400), method='inner', dropout=0.2)
    elif name == 'multi-ipnn-4x10':
        return MultiPNNModel(field_dims, embed_dims=[10]*4, mlp_dims=(400, 400), method='inner', dropout=0.2)
    elif name == 'multi-ipnn-10x10':
        return MultiPNNModel(field_dims, embed_dims=[10]*10, mlp_dims=(400, 400), method='inner', dropout=0.2)
    
    elif name == 'nfwfm-50':
        return NFwFMModel(field_dims, embed_dim=50, mlp_dims=(400, 400), dropouts=(0.2, 0.2))
    elif name == 'nfwfm-100':
        return NFwFMModel(field_dims, embed_dim=100, mlp_dims=(400, 400), dropouts=(0.2, 0.2))
    elif name == 'nfwfm-150':
        return NFwFMModel(field_dims, embed_dim=150, mlp_dims=(400, 400), dropouts=(0.2, 0.2))
    elif name == 'nfwfm-200':
        return NFwFMModel(field_dims, embed_dim=200, mlp_dims=(400, 400), dropouts=(0.2, 0.2))
    elif name == 'nfwfm-500':
        return NFwFMModel(field_dims, embed_dim=500, mlp_dims=(400, 400), dropouts=(0.2, 0.2))
    
    elif name == 'multi-nfwfm-2x50':
        return MultiNFwFMModel(field_dims, embed_dims=[50]*2, mlp_dims=(400, 400), dropouts=(0.2, 0.2))
    elif name == 'multi-nfwfm-3x50':
        return MultiNFwFMModel(field_dims, embed_dims=[50]*3, mlp_dims=(400, 400), dropouts=(0.2, 0.2))
    elif name == 'multi-nfwfm-4x50':
        return MultiNFwFMModel(field_dims, embed_dims=[50]*4, mlp_dims=(400, 400), dropouts=(0.2, 0.2))
    elif name == 'multi-nfwfm-10x50':
        return MultiNFwFMModel(field_dims, embed_dims=[50]*10, mlp_dims=(400, 400), dropouts=(0.2, 0.2))
    
    elif name == 'xdfm-10':
        return XDeepFM(field_dims, embed_dim=10, mlp_dims=(400, 400), dropout=0.2, cross_layer_sizes=(16, 16))
    elif name == 'xdfm-20':
        return XDeepFM(field_dims, embed_dim=20, mlp_dims=(400, 400), dropout=0.2, cross_layer_sizes=(16, 16))
    elif name == 'xdfm-30':
        return XDeepFM(field_dims, embed_dim=30, mlp_dims=(400, 400), dropout=0.2, cross_layer_sizes=(16, 16))
    elif name == 'xdfm-40':
        return XDeepFM(field_dims, embed_dim=40, mlp_dims=(400, 400), dropout=0.2, cross_layer_sizes=(16, 16))
    elif name == 'xdfm-100':
        return XDeepFM(field_dims, embed_dim=100, mlp_dims=(400, 400), dropout=0.2, cross_layer_sizes=(16, 16))
    
    elif name == 'multi-xdfm-2x10':
        return MultiXDeepFM(field_dims, embed_dims=[10]*2, mlp_dims=(400, 400), dropout=0.2, cross_layer_sizes=(16, 16))
    elif name == 'multi-xdfm-3x10':
        return MultiXDeepFM(field_dims, embed_dims=[10]*3, mlp_dims=(400, 400), dropout=0.2, cross_layer_sizes=(16, 16))
    elif name == 'multi-xdfm-4x10':
        return MultiXDeepFM(field_dims, embed_dims=[10]*4, mlp_dims=(400, 400), dropout=0.2, cross_layer_sizes=(16, 16))
    elif name == 'multi-xdfm-10x10':
        return MultiXDeepFM(field_dims, embed_dims=[10]*10, mlp_dims=(400, 400), dropout=0.2, cross_layer_sizes=(16, 16))
    
    elif name == 'finalmlp-10':
        return FinalMLP(field_dims, embed_dim=10, mlp_dims=(400, 400), fs_mlp_dims=(800, ), dropout=0.2)
    elif name == 'finalmlp-20':
        return FinalMLP(field_dims, embed_dim=20, mlp_dims=(400, 400), fs_mlp_dims=(800, ), dropout=0.2)
    elif name == 'finalmlp-30':
        return FinalMLP(field_dims, embed_dim=30, mlp_dims=(400, 400), fs_mlp_dims=(800, ), dropout=0.2)
    elif name == 'finalmlp-40':
        return FinalMLP(field_dims, embed_dim=40, mlp_dims=(400, 400), fs_mlp_dims=(800, ), dropout=0.2)
    elif name == 'finalmlp-100':
        return FinalMLP(field_dims, embed_dim=100, mlp_dims=(400, 400), fs_mlp_dims=(800, ), dropout=0.2)
    
    elif name == 'multi-finalmlp-2x10':
        return MultiFinalMLP(field_dims, embed_dims=[10]*2, mlp_dims=(400, 400), fs_mlp_dims=(800, ), dropout=0.2)
    elif name == 'multi-finalmlp-3x10':
        return MultiFinalMLP(field_dims, embed_dims=[10]*3, mlp_dims=(400, 400), fs_mlp_dims=(800, ), dropout=0.2)
    elif name == 'multi-finalmlp-4x10':
        return MultiFinalMLP(field_dims, embed_dims=[10]*4, mlp_dims=(400, 400), fs_mlp_dims=(800, ), dropout=0.2)
    elif name == 'multi-finalmlp-10x10':
        return MultiFinalMLP(field_dims, embed_dims=[10]*10, mlp_dims=(400, 400), fs_mlp_dims=(800, ), dropout=0.2)
    
    else:
        raise ValueError('unknown model name: ' + name)


def train(model, optimizer, data_loader, criterion, device, epoch, accumulate_gradient=1, log_interval=500):
    model.train()
    batch_time = AverageMeter('Total Time', ':4.2f')
    losses = AverageMeter("Loss", ":5.4f")
    progress = ProgressMeter(len(data_loader), [batch_time, losses], prefix="Epoch: [{}]".format(epoch))
    steps = 0
    optimizer.zero_grad()
    end = time.time()
    for i, (fields, target) in enumerate(data_loader):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        losses.update(loss.item())
        accumulate_loss = loss / accumulate_gradient
        accumulate_loss.backward()
        steps += 1
        if steps % accumulate_gradient == 0:
            optimizer.step()
            optimizer.zero_grad()
        batch_time.update(len(data_loader) * (time.time() - end))
        end = time.time()
        if (i + 1) % log_interval == 0:
            progress.display(i + 1)
    optimizer.zero_grad()


def test(model, data_loader, device, log_interval=500):
    model.eval()
    batch_time = AverageMeter('Total Time', ':4.2f')
    progress = ProgressMeter(len(data_loader), [batch_time], prefix="Test:")
    targets, predicts = list(), list()
    end = time.time()
    with torch.no_grad():
        for i, (fields, target) in enumerate(data_loader):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            batch_time.update(len(data_loader) * (time.time() - end))
            end = time.time()
            if (i + 1) % log_interval == 0:
                progress.display(i + 1)

    return roc_auc_score(targets, predicts)


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         phase,
         seed,
         accumulate_gradient,
         ):
    logger = CompleteLogger(args.log, args.phase)
    print(args)
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length), generator=torch.Generator().manual_seed(seed))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)


    model = get_model(model_name, dataset).to(device)
    print(model.state_dict().keys())

    # count parameters
    for n, p in model.named_parameters():
        print(n, p.numel())
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    if phase == "train":
        if isinstance(model, (RestrictedCrossNetworkV2Model,
                              WeightedRestrictedCrossNetworkV2Model,
                              WeightedRestrictedMultiDCN,
                              WeightNormAlignedMultiDCNnew2,
                              SpaceSimilarityRegularizedMultiDCNnew2,
                              SingularValueRegularizedDCNv2,)):
            criterion = CriterionWithLoss(torch.nn.BCELoss())
        else:
            criterion = torch.nn.BCELoss()
        if hasattr(model, "get_parameters"):
            optimizer = torch.optim.Adam(params=model.get_parameters(learning_rate), weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        save_paths = [logger.get_checkpoint_path("best"), logger.get_checkpoint_path("optimizer")]
        if epoch == 0:
            early_stopper = EarlyStopper(num_trials=3, save_paths=save_paths)
            epoch = 100
        else:
            early_stopper = EarlyStopper(num_trials=epoch, save_paths=save_paths)
        auc = test(model, valid_data_loader, device)
        print(auc)
        for epoch_i in range(epoch):
            train(model, optimizer, train_data_loader, criterion, device, epoch_i, accumulate_gradient)
            torch.save(model.state_dict(), logger.get_checkpoint_path("latest"))
            auc = test(model, valid_data_loader, device)
            print('epoch:', epoch_i, 'validation: auc:', auc)
            if not early_stopper.is_continuable((model, optimizer), auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break
    
    model.load_state_dict(torch.load(logger.get_checkpoint_path("best")))

    auc = test(model, test_data_loader, device)
    print('test auc:', auc)

    logger.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt or avazu/train')
    parser.add_argument('--model_name', default='dcn-10')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--k_dim', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--log', default='logs/test')
    parser.add_argument('--phase', default='train', choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--accumulate_gradient', '--acc_grad', type=int, default=1)
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.phase,
         args.seed,
         args.accumulate_gradient)
