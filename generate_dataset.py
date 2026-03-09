from dataset_generator import generate_noisy_dataset


generate_noisy_dataset(

    clean_dir="clean",
    noise_dir="noise",
    rir_dir="rir",

    output_dir="noisy",

    # Option 1: fixed SNR
    snr_db=10,

    # Option 2: random SNR
    # snr_range=(-5, 15)
)