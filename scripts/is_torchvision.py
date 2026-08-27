import click, numpy as np, torch
import torch.nn.functional as F
from torchvision.models import inception_v3
import dnnlib, legacy
from training.dataset import ImageFolderDataset
@click.command()
@click.option('--network', required=True)
@click.option('--data', required=True)
@click.option('--num', default=50000, type=int)
@click.option('--splits', default=10, type=int)
@click.option('--batch', default=64, type=int)
@click.option('--seed', default=0, type=int)
def main(network, data, num, splits, batch, seed):
    device = torch.device('cuda')
    torch.manual_seed(seed); np.random.seed(seed)
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
    ds = ImageFolderDataset(path=data, use_labels=True)
    labels = np.array([int(np.argmax(ds.get_label(i))) for i in range(len(ds))])
    probs = np.bincount(labels, minlength=G.c_dim).astype(np.float64)
    probs /= probs.sum()
    net = inception_v3(pretrained=True, transform_input=True).to(device).eval()
    preds = np.zeros((num, 1000), dtype=np.float32)
    n = 0
    with torch.no_grad():
        while n < num:
            b = min(batch, num - n)
            z = torch.randn(b, G.z_dim, device=device)
            cls = np.random.choice(G.c_dim, size=b, p=probs)
            c = torch.zeros(b, G.c_dim, device=device); c[range(b), cls] = 1
            img = G(z, c, truncation_psi=1.0, noise_mode='const')
            img = (img.clamp(-1, 1) + 1) / 2
            img = F.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
            p = F.softmax(net(img), dim=1).cpu().numpy()
            preds[n:n+b] = p; n += b
            print(f'{n}/{num}', end='\r')
    scores = []
    for k in range(splits):
        part = preds[k*(num//splits):(k+1)*(num//splits)]
        py = np.mean(part, axis=0)
        kl = np.mean(np.sum(part * (np.log(part + 1e-16) - np.log(py + 1e-16)), axis=1))
        scores.append(float(np.exp(kl)))
    print(f'\nIS = {np.mean(scores):.4f} +/- {np.std(scores):.4f}')
if __name__ == '__main__':
    main()
