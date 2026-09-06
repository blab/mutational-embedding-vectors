import os

curr_directory_path = os.path.dirname(os.path.realpath(__file__))

# important global variables
FOLD_ID = 0
N_TARGETS = 1565
CONTEXT_LEN = 1024

# global paths (used for setup of both CovFit and ESM_coronaviridae)
COVFIT_STUFF_PATH = os.path.join(curr_directory_path, "../notebooks/covfit_stuff")
MODEL_DIR = os.path.join(COVFIT_STUFF_PATH, "models")
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
TASK_IDS_FILE = os.path.join(COVFIT_STUFF_PATH, "task_id_dict.pt")

# tokenizer things
TOK_DIR = os.path.join(COVFIT_STUFF_PATH, "Tokenizer")
SPECIAL_TOK_MAP = os.path.join(TOK_DIR, "special_tokens_map.json")
VOCAB_FILE = os.path.join(TOK_DIR, "vocab.txt")

# model paths/config files
MODEL_PATH = {
    "coronaviridae":os.path.join(MODEL_DIR, "model_ESM2_coronaviridae/pytorch_model.bin"),
    "covfit":os.path.join(MODEL_DIR, f"covfit_model_20241007_{FOLD_ID}.ckpt")
}
CONF_PATH = {
    "coronaviridae":os.path.join(MODEL_DIR, "model_ESM2_coronaviridae/config.json"),
    "covfit":os.path.join(COVFIT_STUFF_PATH, "CovFit_Config/config.json")
}

device = "cuda"
