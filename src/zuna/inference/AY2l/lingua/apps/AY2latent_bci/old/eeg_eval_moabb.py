### RUN 
### >>>   python3 apps/AY2latent_bci/eeg_eval_moabb.py config=apps/AY2latent_bci/configs/config_bci_eval.yaml


import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# model specific imports
from omegaconf import OmegaConf
from apps.AY2latent_bci.transformer import EncoderDecoder
from lingua.checkpoint import load_from_checkpoint
from apps.AY2latent_bci.eeg_eval import TrainArgs
from apps.AY2latent_bci.eeg_data import EEGDataset_v2, EEGProcessor, BCIDatasetArgs, create_dataloader, create_dataloader


device = torch.device("cpu")
# device = torch.device("cuda")

# Load EEG data
# pt_path = "/mnt/shared/datasets/bci/moabb/BNCI2014_001_5600e_64c_2561s.pt"
pt_path =  "/workspace/bci/data/moabb/BNCI2014_001_5600e_64c_2561s.pt"
data = torch.load(pt_path, weights_only=False)
eeg = data["eeg"].numpy().astype(np.float64)  # [N, C, T]
labels = data["labels"].numpy()
class_mapping = data["class_mapping"]
BATCH_SIZE = 1024

#############
### MODEL ###
#############

# model configurations
cli_args = OmegaConf.from_cli()
file_cfg = OmegaConf.load(cli_args.config)
del cli_args.config
default_cfg = OmegaConf.structured(TrainArgs())
cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
cfg = OmegaConf.to_object(cfg)  # converts to nested dataclasses

# load model 
model = EncoderDecoder(cfg.model).to(device).eval()
load_from_checkpoint(cfg.checkpoint.init_ckpt_path, model, model_key="model")

# Prepare EEG tensor
eeg_tensor = torch.tensor(eeg, dtype=torch.float32).to(device)  # [N, C, T]
eeg_tensor = eeg_tensor.permute(0, 2, 1)                    # [N, T, C]
mask = (eeg_tensor.abs().sum(dim=1) != 0).unsqueeze(1).int()  # [N, 1, C]


# Forward through model
data_processor = EEGProcessor(cfg.data).to(device)

chunks = []
latents = []
with torch.no_grad():
    for i in range(0, eeg_tensor.shape[0], BATCH_SIZE):  # slices epochs/samples
        eeg_batch = eeg_tensor[i:i+BATCH_SIZE]  # [B, T, C]
        mask_batch = (eeg_batch.abs().sum(dim=1) != 0).unsqueeze(1).int()
        batch_chunk = {"eeg_signal": eeg_batch, "freq_masks": mask_batch}
        import pdb; pdb.set_trace()  # Debugging breakpoint

        batch_chunk = data_processor.process(**batch_chunk)
        dec_out, _, _ = model(**batch_chunk)             # [B, T, C]
        chunks.append(dec_out.cpu())
        latents.append(batch_chunk["encoder_input"].cpu())

    chunk = torch.cat(chunks, dim=0)   # [N, T, C]
    latent = torch.cat(latents, dim=0)   # [N, T, latent_dim]

    
# with torch.no_grad():
#     for i in range(0, eeg_tensor.shape[1], MAX_SEQ_LEN):
#         eeg_chunk = eeg_tensor[:, i:i+MAX_SEQ_LEN, :]  # [N, chunk_T, C]
#         if eeg_chunk.shape[1] < MAX_SEQ_LEN:
#             # Pad the last chunk if too short
#             pad_len = MAX_SEQ_LEN - eeg_chunk.shape[1]
#             eeg_chunk = F.pad(eeg_chunk, (0, 0, 0, pad_len))  # pad time dim

#         mask_chunk = (eeg_chunk.abs().sum(dim=1) != 0).unsqueeze(1).int()

#         batch_chunk = {"eeg_signal": eeg_chunk, "freq_masks": mask_chunk}
#         batch_chunk = processor.process(**batch_chunk)
#         import pdb; pdb.set_trace()  # Debugging breakpoint
#         dec_out, _, _ = model(**batch_chunk)                         # [N, T, C]
#         chunks.append(dec_out.cpu())                                 # list of [N, T, C]
#         latents.append(batch_chunk["encoder_input"].cpu())           # list of [N, T, latent_dim]

#     dec_out_full = torch.cat(chunks, dim=1)    # [N, total_T, C]
#     latent_full = torch.cat(latents, dim=1)    # [N, total_T, latent_dim]



# Process via EEGProcessor
# processor = EEGProcessor(cfg.data).to(device)
# batch = {"eeg_signal": eeg_tensor, "freq_masks": mask}
# batch = processor.process(**batch)

# with torch.no_grad():
#     import pdb; pdb.set_trace()  # Debugging breakpoint
#     dec_out, _, _ = model(**batch)  # [N, T, C]
#     reconstructed = dec_out.permute(0, 2, 1).cpu().numpy()  # back to [N, C, T]
#     latent = batch["encoder_input"].cpu().numpy()           # usually [N, T, latent_dim]

# Flatten latent space if needed
latent_flat = latent.reshape(latent.shape[0], -1)  # [N, T * latent_dim]

# -------- Classifier evaluation --------
pipeline = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear"))

print("\n[1] Classifier on RAW EEG:")
X_train, X_test, y_train, y_test = train_test_split(eeg, labels, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=[class_mapping[i] for i in sorted(class_mapping)]))

print("\n[2] Classifier on RECONSTRUCTED EEG:")
X_train, X_test, y_train, y_test = train_test_split(reconstructed, labels, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=[class_mapping[i] for i in sorted(class_mapping)]))

print("\n[3] Classifier on LATENT SPACE:")
from sklearn.linear_model import LogisticRegression
latent_clf = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(latent_flat, labels, test_size=0.2, random_state=42)
latent_clf.fit(X_train, y_train)
y_pred = latent_clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=[class_mapping[i] for i in sorted(class_mapping)]))