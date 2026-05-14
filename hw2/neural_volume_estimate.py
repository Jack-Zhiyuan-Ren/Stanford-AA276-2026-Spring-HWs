import torch

from problem4_helper import NeuralVF


# =========================
# Settings
# =========================

NUM_SAMPLES = 100000  # number of random states to sample
BATCH_SIZE = 4096     # number of samples processed at once
CKPT_PATH = "outputs/vf.ckpt"


def sample_states(num_samples, device="cpu"):
    """
    Uniformly sample states from the 13D state space.

    State:
    x = [px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    """

    lower = torch.tensor([
        -3.0, -3.0, -3.0,        # px, py, pz
        -1.0, -1.0, -1.0, -1.0,  # qw, qx, qy, qz
        -5.0, -5.0, -5.0,        # vx, vy, vz
        -5.0, -5.0, -5.0         # wx, wy, wz
    ], device=device)

    upper = torch.tensor([
         3.0,  3.0,  3.0,
         1.0,  1.0,  1.0,  1.0,
         5.0,  5.0,  5.0,
         5.0,  5.0,  5.0
    ], device=device)

    rand = torch.rand(num_samples, 13, device=device)
    x = lower + (upper - lower) * rand

    return x


def estimate_safe_volume(model, num_samples, batch_size):
    """
    Estimate safe volume ratio using random sampling.

    Safe volume ratio = number of safe samples / total number of samples.
    For the HJ value function, a state is safe if V(x) >= 0.
    """

    num_safe = 0
    num_total = 0

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - start)

            x = sample_states(current_batch_size)

            values = model.values(x)

            # For HJ reachability, safe if V(x) >= 0
            safe_mask = values >= 0.0

            num_safe += safe_mask.sum().item()
            num_total += current_batch_size

    ratio = num_safe / num_total

    # Monte Carlo standard error
    standard_error = (ratio * (1.0 - ratio) / num_total) ** 0.5

    return ratio, standard_error


if __name__ == "__main__":

    torch.manual_seed(0)

    print("Loading neural HJ value function model...")
    model = NeuralVF(ckpt_path=CKPT_PATH)

    ratio, standard_error = estimate_safe_volume(
        model=model,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE
    )

    print()
    print("Model: vf")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Estimated safe volume ratio: {ratio:.6f}")
    print(f"Approximate standard error: {standard_error:.6f}")
    print(
        "Approximate 95% confidence interval: "
        f"[{ratio - 1.96 * standard_error:.6f}, "
        f"{ratio + 1.96 * standard_error:.6f}]"
    )