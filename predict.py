from mos_predictors import QAMA
import click
import os
import fairseq
import torch
import torchaudio
import pandas as pd
from collections import defaultdict
from datetime import datetime
import gdown

@click.command()
@click.option('--data_dir', required=True, type=str)
@click.option('--full_model', required=False, type=bool, default=True)
def predict(data_dir, full_model):
    target_sr = 16000
    
    # Set model path
    DIR_MODELS = 'models/mos_predictors'
    models = ['qama_fold1.pt', 'qama_fold2.pt', 'qama_fold3.pt']
    if full_model:
        models.append('qama_full.pt')
    CHECKPOINT_PATH = 'models/pretrain/checkpoint_best.pt'

    # Download models from gdrive (check if already downloaded)
    start_download = False
    if not os.path.isfile(CHECKPOINT_PATH):
        start_download = True

    if not start_download:
        for m in models:
            if ~os.path.isfile(os.path.join(DIR_MODELS, m)):
                start_download = True
    
    if start_download:
        print('Wait...Downloading models')
        url = 'https://drive.google.com/drive/folders/1V8jwo0uaQS_og6r1By7oooBtXzFvOEIz?usp=share_link'
        gdown.download_folder(url, quiet=True, use_cookies=False)
        print('Download Completed')

    SSL_OUT_DIM = 768
    
    # Load SSL model
    w2v_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([CHECKPOINT_PATH])
    ssl_model = w2v_model[0]
    ssl_model.remove_pretraining_modules()

    # Look for GPU
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Dataframe store results
    dfs = defaultdict(list)

    for model_name in models:
        # Empty lists to store predictions and sample names
        filenames, preds = [], [] 

        # Define MOS predictor
        model = QAMA(ssl_model, SSL_OUT_DIM)
        model = model.to(device)

        # Load weights
        model_path = os.path.join(DIR_MODELS, model_name)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            for filename in os.listdir(data_dir):
                # Create filepath
                filepath = os.path.join(data_dir, filename)

                # Load waveform
                waveform, sr = torchaudio.load(filepath)

                # Check channels
                if waveform.shape[0] > 1:
                    waveform = ((waveform[0,:] + waveform[1,:])/2).unsqueeze(0)
                
                # Check for resampling
                if sr != target_sr:
                    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

                # Prediction
                waveform = waveform.to(device)
                mos_pred = model(waveform)

                # Store data
                filenames.append(filename)
                preds.append(mos_pred)
        
        # Store predictions
        model_name_noext, ext = model_name.split('.')
        preds = torch.cat(preds).cpu().numpy()
        df_out = pd.DataFrame({'filename': filenames, 'preds': preds}).set_index('filename')
        dfs[model_name_noext] = df_out

    # Create one dataframe
    df_predictions = pd.concat(dfs, axis=1)
    df_predictions.columns = [m.split('.')[0] for m in models]
    df_predictions['Mean CV'] = df_predictions.loc[:,df_predictions.columns.str.contains('fold')].mean(axis=1)

    # Save predictions to file
    OUT_DIR = 'prediction_files'
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    out_path = os.path.join(OUT_DIR, dt_string + '_qama.csv')
    df_predictions.to_csv(out_path)
    print(f'Predictions saved: {out_path}')

if __name__ == '__main__':
    predict()