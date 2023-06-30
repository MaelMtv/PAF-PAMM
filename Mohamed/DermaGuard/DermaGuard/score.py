import torch
import piq
from skimage import io

@torch.no_grad()
def main():

    # On convertit notre image en tenseur pour utiliser la fonction brisque qui prend un tenseur en entrée.
    tensor = torch.tensor(io.imread('img.jpg')).permute(2, 0, 1)
    x = tensor[None, ...] / 255.

    if torch.cuda.is_available():
        # On laisse le calcul au GPU
        x = x.cuda()

    brisque_index: torch.Tensor = piq.brisque(x, data_range=1., reduction='none')

    print(f"BRISQUE: {brisque_index.item():0.4f}")

if __name__ == '__main__':
    main()
