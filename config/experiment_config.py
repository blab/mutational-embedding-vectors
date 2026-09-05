import os

curr_directory_path = os.path.dirname(os.path.realpath(__file__))

# important global variables
FOLD_ID = 0
N_TARGETS = 1565
CONTEXT_LEN = 1024

# global paths (used for setup of both CovFit and ESM_coronaviridae)
COVFIT_STUFF_PATH = os.path.join(curr_directory_path, "../notebooks/covfit_stuff")
MODEL_DIR = os.path.join(COVFIT_STUFF_PATH, "models")
TOK_DIR = os.path.join(COVFIT_STUFF_PATH, "Tokenizer")
TASK_IDS_FILE = os.path.join(COVFIT_STUFF_PATH, "task_id_dict.pt")


# coronaviridae path
MODEL_PATH = {
    "coronaviridae":os.path.join(MODEL_DIR, "model_ESM2_coronaviridae/pytorch_model.bin"),
    "covfit":os.path.join(MODEL_DIR, f"covfit_model_20241007_{FOLD_ID}.ckpt")
}
CONF_PATH = {
    "coronaviridae":os.path.join(MODEL_DIR, "model_ESM2_coronaviridae/config.json"),
}

f"./covfit_stuff/models/covfit_model_20241007_{FOLD_ID}.ckpt"

device = "cuda"
