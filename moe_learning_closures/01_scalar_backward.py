import torch

from common import get_device


def manual_step(x, w, b, target, lr):
    y = x * w + b
    loss = (y - target) ** 2

    dloss_dy = 2 * (y - target)
    dloss_dw = dloss_dy * x
    dloss_db = dloss_dy
    dloss_dx = dloss_dy * w

    new_w = w - lr * dloss_dw
    new_b = b - lr * dloss_db

    return {
        "y": y,
        "loss": loss,
        "dloss_dx": dloss_dx,
        "dloss_dw": dloss_dw,
        "dloss_db": dloss_db,
        "new_w": new_w,
        "new_b": new_b,
    }


def torch_step(device):
    x = torch.tensor(2.0, device=device, requires_grad=True)
    w = torch.tensor(3.0, device=device, requires_grad=True)
    b = torch.tensor(1.0, device=device, requires_grad=True)
    target = torch.tensor(10.0, device=device)
    lr = 0.1

    y = x * w + b
    loss = (y - target) ** 2
    loss.backward()

    with torch.no_grad():
        new_w = w - lr * w.grad
        new_b = b - lr * b.grad

    return {
        "y": y.detach(),
        "loss": loss.detach(),
        "dloss_dx": x.grad.detach(),
        "dloss_dw": w.grad.detach(),
        "dloss_db": b.grad.detach(),
        "new_w": new_w.detach(),
        "new_b": new_b.detach(),
    }


def main():
    device = get_device()
    x = torch.tensor(2.0, device=device)
    w = torch.tensor(3.0, device=device)
    b = torch.tensor(1.0, device=device)
    target = torch.tensor(10.0, device=device)

    manual = manual_step(x, w, b, target, lr=0.1)
    autograd = torch_step(device)

    print(f"device: {device}")
    for key in manual:
        print(f"{key:10s} manual={manual[key].item():8.4f} autograd={autograd[key].item():8.4f}")
        torch.testing.assert_close(manual[key], autograd[key])


if __name__ == "__main__":
    main()
