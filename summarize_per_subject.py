#!/usr/bin/env python3
import os
import re
import sys
import csv

# === Configurazione ===
DEFAULT_ROOT_DIR = "results"  
DEFAULT_OUT_FILE = "subject_summary.txt"
MODELS = ["RatioWaveNet"] 
DATASETS = ["bci2a", "bci2b", "HGD"] 
# ======================================================================

SUBJ_LINE_RE = re.compile(
    r"Subject\s*(?P<id>\d+)\s*=>.*?Test Acc:\s*(?P<acc>\d*\.?\d+)",
    re.IGNORECASE | re.DOTALL
)

def parse_folder_name(name):
    parts = name.split('_')
    seed = None
    aug = None
    loso = False
    
    for seg in parts:
        if seg.startswith('seed-'):
            try:
                seed = int(seg.split('seed-')[1])
            except ValueError:
                pass
        elif seg.startswith('aug-'):
            val = seg.split('aug-')[1]
            if val in ('True', 'False'):
                aug = (val == 'True')
        elif seg.lower() == 'loso':
            loso = True
            
    if seed is None or aug is None:
        return None
    return {'seed': seed, 'aug': aug, 'loso': loso}

def collect_run_results(root_dir):
    data = {False: {False: {}, True: {}}, True: {False: {}, True: {}}}
    
    print(f" -> Scansione in corso dentro: {root_dir}...")
    
    for dirpath, _, files in os.walk(root_dir):
        if 'results.txt' not in files:
            continue
            
        folder_name = os.path.basename(dirpath)
        info = parse_folder_name(folder_name)
        
        if not info:
            continue

        seed = info['seed']
        aug = info['aug']
        loso = info['loso']
        
        txt_path = os.path.join(dirpath, 'results.txt')
        
        try:
            with open(txt_path, 'r') as f:
                text = f.read()
            
            subj_map = {}
            for m in SUBJ_LINE_RE.finditer(text):
                sid = int(m.group('id'))
                acc = float(m.group('acc'))
                subj_map[sid] = acc
            
            if subj_map:
                data[aug][loso][seed] = subj_map
        except Exception as e:
            print(f"Errore leggendo {txt_path}: {e}")

    return data

def write_summary(data, root_dir, out_file):
    out_path = os.path.join(root_dir, out_file)
    
    # === CORREZIONE QUI ===
    # Iteriamo su data[a][l].values() per prendere i dizionari, non le chiavi (seed int)
    total_entries = sum(len(subj_map) for a in data for l in data[a] for subj_map in data[a][l].values())
    
    if total_entries == 0:
        print(f" Nessun dato valido trovato in {root_dir}. Controlla che i file results.txt esistano e contengano 'Subject X => ... Test Acc:'.")
        return

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for aug_flag in (False, True):
            for loso_flag in (False, True):
                label = (
                    ('With Aug' if aug_flag else 'Without Aug') +
                    (' - LOSO' if loso_flag else ' - sub-dependent')
                )
                f.write(label + '\n')
                
                seeds = sorted(data[aug_flag][loso_flag].keys())
                if not seeds:
                    f.write('No runs found.\n\n')
                    continue
                
                all_subjs = set()
                for s in seeds:
                    all_subjs.update(data[aug_flag][loso_flag][s].keys())
                subjects = sorted(all_subjs)
                
                header = ['Subject'] + [f"seed-{s}" for s in seeds]
                writer.writerow(header)
                
                for sid in subjects:
                    row = [str(sid)]
                    for s in seeds:
                        subj_map = data[aug_flag][loso_flag].get(s, {})
                        acc = subj_map.get(sid)
                        row.append(f"{acc:.4f}" if acc is not None else '')
                    writer.writerow(row)
                f.write('\n')
    
    print(f" -> Summary creato con successo: {out_path}")

def main(base_search_path):
    for model in MODELS:
        for ds in DATASETS:
            target_dir = os.path.join(base_search_path, model, ds)
            
            if not os.path.exists(target_dir):
                print(f"Attenzione: La cartella non esiste, salto: {target_dir}")
                continue
            
            print(f"\nAnalizzando Dataset: {ds} (Model: {model})")
            data = collect_run_results(target_dir)
            write_summary(data, target_dir, DEFAULT_OUT_FILE)

if __name__ == '__main__':
    working_dir = os.getcwd()
    base_path = sys.argv[1] if len(sys.argv) >= 2 else os.path.join(working_dir, DEFAULT_ROOT_DIR)
    
    print(f"Root folder impostata: {base_path}")
    main(base_path)