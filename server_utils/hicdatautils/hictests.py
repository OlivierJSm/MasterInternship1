import numpy as np
import matplotlib.pyplot as plt

def generate_contact_map(size: int = 10, total_reads: int = 2000, decay_rate: float = 0.2, 
                         noise_level: float = 0.1, seed: int = None):
    """
    Generates a synthetic Hi-C-like contact map.

    Parameters
    ----------
    size : int
        Dimension of the square matrix.
        Default = 10.
    total_reads : float
        Desired total sum of all counts.
        Default = 2000.
    decay_rate : float
        Controls how quickly contact frequency decays with genomic distance.
        Default = 0.2.
    noise_level : float
        Adds random variation to simulate experimental noise.
        Default = 0.1.
    seed : int or None
        Random seed for reproducibility.
        Default = None.

    Returns
    -------
    contact_map : np.ndarray
        Symmetric contact matrix (size x size).
    """
    rng = np.random.default_rng(seed)
    
    # Create distance matrix
    i, j = np.indices((size, size))
    distance = np.abs(i - j)
    
    # Exponential decay along distance
    signal = np.exp(-decay_rate * distance)
    
    # Add Gaussian noise
    noise = rng.normal(1, noise_level, size=(size, size))
    contact_map = signal * noise
    
    # Symmetrize
    contact_map = (contact_map + contact_map.T) / 2
    
    # Scale to desired total reads
    contact_map *= total_reads / contact_map.sum()
    
    # Round to integer read counts
    contact_map = np.round(contact_map).astype(int)
    
    return contact_map

def generate_related_contact_map(base_map: np.ndarray, similarity: float, base_decay: float =0.2,
                                 max_extra_noise: float =0.7, seed: int =None):
    """
    Generate a contact map with a controllable level of similarity to a base map.

    Parameters
    ----------
    base_map : np.ndarray
        The original contact matrix.
    similarity : float
        Value between 0 (completely different) and 1 (identical).
    base_decay : float
        Base decay rate to use.
        Default = 0.2
    max_extra_noise : float
        Maximum additional noise at lowest similarity.
        Default = 0.7.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    new_map : np.ndarray
        Modified contact map with controlled similarity.
    """
    # Check if similarity is in bounds
    if 1 < similarity or 0 > similarity:
        raise ValueError(f'similarity not between 0 and 1: {similarity}')

    # Define decay bounds
    min_decay = base_decay/5
    max_decay = base_decay * 3

    # Initialize rng
    rng = np.random.default_rng(seed)
    size = base_map.shape[0]

    # Interpolate decay rate: close to base_decay for high similarity,
    # far away (random within range) for low similarity
    if similarity < 1:
        target_decay = base_decay + (rng.uniform(min_decay, max_decay) - base_decay) * (1 - similarity)
    else:
        target_decay = base_decay

    # Noise level increases as similarity decreases
    noise_level = max_extra_noise * (1 - similarity)

    # Generate a variant map with slightly different decay
    variant_map = generate_contact_map(size=size,
                                       total_reads=base_map.sum(),
                                       decay_rate=target_decay,
                                       noise_level=noise_level,
                                       seed=seed)

    # Blend base and variant
    new_map = similarity * base_map + (1 - similarity) * variant_map

    # Add localized noise (small random patches)
    patch_intensity = 0.2 * (1 - similarity)
    if patch_intensity > 0:
        patches = rng.lognormal(mean=0, sigma=patch_intensity, size=base_map.shape)
        new_map *= patches

    # Clip any remaining negatives
    new_map = np.clip(new_map, 0, None)

    # Symmetrize and rescale to keep total reads the same
    new_map = (new_map + new_map.T) / 2
    total_reads = base_map.sum()
    new_map *= total_reads / new_map.sum()
    new_map = np.round(new_map).astype(int)

    return new_map

def visualize_synthetic_maps(maps: list[np.ndarray], titles: list[str]):
    '''
    Visualizes a list of synthetic contact maps.

    Parameters
    ----------
    maps: list[np.ndarray]
        List of synthetic contact maps.
    titles: list[str]
        List of contact map names.
    
    Returns
    -------
    fig: plt.figure.Figure
        Generated figures.
    axes: plt.axes._axes.Axes
        Generated axes.
    '''
    # Ensure same length of lists
    if len(maps) != len(titles):
        raise ValueError('Number of titles differs from number of maps!')
    
    # Drawing figure
    fig, axes = plt.subplots(1, len(maps), figsize=(12, 4))
    for ax, M, title in zip(axes, maps, titles):
        ax.imshow(M, cmap='Reds')
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return fig, axes