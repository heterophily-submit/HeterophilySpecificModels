import torch
import datasets
import torch.nn as nn

from impl import metrics, PolyConv, models, GDataset, utils
from torch.optim import Adam


def split(resplit: bool = True):
    global baseG, trn_dataset, val_dataset, tst_dataset
    if resplit:
        baseG.mask = datasets.split(baseG, split=args.split, split_id=args.split_id)
    trn_dataset = GDataset.GDataset(*baseG.get_split("train"))
    val_dataset = GDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = GDataset.GDataset(*baseG.get_split("test"))


def buildModel(conv_layer: int = 10,
               aggr: str = "gcn",
               alpha: float = 0.2,
               dpb: float = 0.0,
               dpt: float = 0.0,
               **kwargs):
    if args.multilayer:
        emb = models.Seq([
            models.TensorMod(baseG.x),
            nn.Dropout(p=dpb),
            nn.Sequential(nn.Linear(baseG.x.shape[1], output_channels),
                          nn.ReLU(inplace=True),
                          nn.Linear(output_channels, output_channels)),
            nn.Dropout(dpt, inplace=True)
        ])
    elif args.resmultilayer:
        emb = models.Seq([
            models.TensorMod(baseG.x),
            nn.Dropout(p=dpb),
            nn.Linear(baseG.x.shape[1], output_channels),
            models.ResBlock(
                nn.Sequential(nn.ReLU(inplace=True),
                              nn.Linear(output_channels, output_channels))),
            nn.Dropout(dpt, inplace=True)
        ])
    else:
        emb = models.Seq([
            models.TensorMod(baseG.x),
            nn.Dropout(p=dpb),
            nn.Linear(baseG.x.shape[1], output_channels),
            nn.Dropout(dpt, inplace=True)
        ])

    from functools import partial

    frame_fn = PolyConv.PolyConvFrame
    conv_fn = partial(PolyConv.JacobiConv, **kwargs)
    if args.power:
        conv_fn = PolyConv.PowerConv
    if args.legendre:
        conv_fn = PolyConv.LegendreConv
    if args.cheby:
        conv_fn = PolyConv.ChebyshevConv

    if args.bern:
        conv = PolyConv.Bern_prop(conv_layer)
    else:
        if args.fixalpha:
            from bestHyperparams import fixalpha_alpha
            alpha = fixalpha_alpha[args.dataset]["power" if args.power else (
                "cheby" if args.cheby else "jacobi")]
        conv = frame_fn(conv_fn,
                        depth=conv_layer,
                        aggr=aggr,
                        alpha=alpha,
                        fixed=args.fixalpha)
    comb = models.Combination(output_channels, conv_layer + 1, sole=args.sole)
    gnn = models.Gmodel(emb, conv, comb).to(device)
    return gnn


def work(conv_layer: int = 10,
         aggr: str = "gcn",
         alpha: float = 0.2,
         lr1: float = 1e-3,
         lr2: float = 1e-3,
         lr3: float = 1e-3,
         wd1: float = 0,
         wd2: float = 0,
         wd3: float = 0,
         dpb=0.0,
         dpt=0.0,
         patience: int = 100,
         split_type: str = 'default',
         **kwargs):
    outs = []
    for rep in range(args.repeat):
        utils.set_seed(rep)
        if split_type != 'default':
            split()
        gnn = buildModel(conv_layer, aggr, alpha, dpb, dpt, **kwargs)
        optimizer = Adam([{
            'params': gnn.emb.parameters(),
            'weight_decay': wd1,
            'lr': lr1
        }, {
            'params': gnn.conv.parameters(),
            'weight_decay': wd2,
            'lr': lr2
        }, {
            'params': gnn.comb.parameters(),
            'weight_decay': wd3,
            'lr': lr3
        }])
        val_score = 0
        early_stop = 0
        for i in range(1000):
            utils.train(optimizer, gnn, trn_dataset, loss_fn)
            score, _ = utils.test(gnn, val_dataset, score_fn, loss_fn=loss_fn)
            if score >= val_score:
                early_stop = 0
                val_score = score
            else:
                early_stop += 1
            if early_stop > patience:
                break
        outs.append(val_score)

    test_score, _ = utils.test(gnn, tst_dataset, score_fn, loss_fn=loss_fn)

    return {'val': val_score, 'test': test_score}


if __name__ == '__main__':
    args = utils.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseG = datasets.load_dataset(args.dataset, args.split, args.split_id)
    baseG.to(device)
    trn_dataset, val_dataset, tst_dataset = None, None, None
    output_channels = baseG.y.unique().shape[0]

    loss_fn = nn.CrossEntropyLoss()
    if output_channels <= 2:
        score_fn = metrics.roc_auc
    else:
        score_fn = metrics.multiclass_accuracy
    
    split(resplit=args.split != 'default')

    scores = work(
        conv_layer=10,
        aggr='gcn',
        alpha=args.alpha,
        lr1=args.lr,
        lr2=args.lr,
        lr3=args.lr,
        wd1=args.wd,
        wd2=args.wd,
        wd3=args.wd,
        dpb=args.dpb,
        dpt=args.dpt,
        a=args.a,
        b=args.b
    )

    print(f"Val score {scores['val']:.3f}\nTest score {scores['test']:.3f}")
