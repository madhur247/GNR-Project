import cv2
import numpy as np
import argparse
import os
from PIL import Image
from tqdm import tqdm
import gc
import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser(description="evaluation script")
parser.add_argument("--test_dir", type=str, required=True, help="Path to test directory")
args = parser.parse_args()

LOCAL_MODEL_PATH = "./qwen2_vl_offline_weights" 
PATCH_DIR = os.path.join(args.test_dir,"patches")                   
TEST_CSV_PATH = os.path.join(args.test_dir,"test.csv")                      
OUTPUT_CSV_PATH = "./submission.csv"
CONFIDENCE_THRESHOLD = 0.50     

def rotate_patch(img, times):
    times = times % 4
    if times == 0:
        return img
    codes = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    return cv2.rotate(img, codes[times - 1])

def edge_mask(pw, ph, strip=0.35):
    mask = np.zeros((ph, pw), dtype=np.uint8)
    sw, sh = int(pw * strip), int(ph * strip)
    mask[:, :sw]   = 255
    mask[:, -sw:]  = 255
    mask[:sh, :]   = 255
    mask[-sh:, :]  = 255
    return mask

def match_sift(kp1, des1, kp2, des2, ratio=0.75, tol=2.0):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0, 0.0, 0.0
        
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m_tuple in raw:
        if len(m_tuple) == 2:
            m, n = m_tuple
            if m.distance < ratio * n.distance:
                good.append(m)
                
    if len(good) < 2:
        return 0, 0.0, 0.0
        
    pts1  = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2  = np.float32([kp2[m.trainIdx].pt for m in good])
    diffs = pts1 - pts2
    
    dx = float(np.round(np.median(diffs[:, 0])))
    dy = float(np.round(np.median(diffs[:, 1])))
    
    inliers = int(np.sum((np.abs(diffs[:, 0] - dx) <= tol) &
                         (np.abs(diffs[:, 1] - dy) <= tol)))
                         
    return inliers, dx, dy

def stitch(patch_dir, strip=0.35):
    files  = sorted(f for f in os.listdir(patch_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    images = {f: cv2.imread(os.path.join(patch_dir, f)) for f in files}
    images = {k: v for k, v in images.items() if v is not None}

    anchor = next(f for f in files if os.path.splitext(f)[0] == 'patch_0')
    pw, ph = images[anchor].shape[1], images[anchor].shape[0]
    
    print(f'{len(images)} patches  |  {pw}x{ph}  |  anchor: {anchor}')

    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.005, edgeThreshold=20)
    mask = edge_mask(pw, ph, strip)

    rotated = {}
    feats   = {}
    for name, img in tqdm(images.items(), desc='Extracting Features'):
        for r in range(4):
            rimg = rotate_patch(img, r)
            rotated[(name, r)] = rimg
            gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
            rmask = rotate_patch(mask, r)
            kp, des = sift.detectAndCompute(gray, rmask)
            feats[(name, r)] = (kp, np.float32(des) if des is not None else None)

    placed   = {anchor: (0, 0, 0)}
    unplaced = set(images.keys()) - {anchor}

    pbar = tqdm(total=len(unplaced), desc='Assembling Map')
    
    current_ratio = 0.75
    min_inliers_req = 4
    phase = 1

    while unplaced:
        best_inl  = 0
        best_cand = None
        found_instant = False

        recent_placed = list(placed.items())[::-1]

        for p_name, (px, py, pr) in recent_placed:
            if found_instant: break
            kp_p, des_p = feats[(p_name, pr)]
            
            for u_name in list(unplaced):
                if found_instant: break
                for r_u in range(4):
                    kp_u, des_u = feats[(u_name, r_u)]
                    
                    inl, dx, dy = match_sift(kp_p, des_p, kp_u, des_u, ratio=current_ratio)
                    
                    if abs(dx) >= pw or abs(dy) >= ph:
                        continue
                        
                    if inl > best_inl and inl >= min_inliers_req:
                        best_inl  = inl
                        best_cand = (u_name, int(px + dx), int(py + dy), r_u)
                        
                        if inl >= 10:
                            found_instant = True
                            break

        if best_cand is None:
            if phase == 1:
                print(f'\nStuck on {len(unplaced)} patches (likely Water/Forest). Switching to Phase 2 (Desperation Mode)...')
                current_ratio = 0.95  
                min_inliers_req = 2   
                phase = 2
                continue 
            else:
                print(f'\nNo geometric matches found at all. {len(unplaced)} patches left unplaced.')
                break

        u_name, ux, uy, rot = best_cand
        placed[u_name]  = (ux, uy, rot)
        unplaced.remove(u_name)
        pbar.update(1)

    pbar.close()

    ax0, ay0, _ = placed[anchor]
    coords = [(v[0] - ax0, v[1] - ay0) for v in placed.values()]
    off_x  = max(0, -min(c[0] for c in coords))
    off_y  = max(0, -min(c[1] for c in coords))
    cw = max(c[0] for c in coords) + pw + off_x
    ch = max(c[1] for c in coords) + ph + off_y
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)

    for name, (ax, ay, rot) in placed.items():
        x  = ax - ax0 + off_x
        y  = ay - ay0 + off_y
        cx1, cy1 = max(0, x), max(0, y)
        cx2, cy2 = min(cw, x + pw), min(ch, y + ph)
        ix1, iy1 = cx1 - x, cy1 - y
        canvas[cy1:cy2, cx1:cx2] = rotated[(name, rot)][iy1:iy1+(cy2-cy1), ix1:ix1+(cx2-cx1)]

    print(f'Done — {cw}x{ch} px  |  {len(unplaced)} unplaced')
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


result = stitch(PATCH_DIR, strip=0.35)
result.save('stitched_sift_final.png')



device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Qwen2-VL")

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True
# )

processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH, 
    # quantization_config=quant_config,
    device_map="auto"
)
model.eval()

VALID_TOKENS = ["1", "2", "3", "4", "5"]
token_ids = [processor.tokenizer.encode(token, add_special_tokens=False)[0] for token in VALID_TOKENS]
valid_token_ids_tensor = torch.tensor(token_ids).to(device)

def run_inference(stitched_map):

    max_dim = 1536
    if max(stitched_map.size) > max_dim:
        stitched_map.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        print(f"Resized map to {stitched_map.size} to save VRAM.")

    df = pd.read_csv(TEST_CSV_PATH)
    results = []
    
    print("Starting deterministic VQA extraction...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        q_id = row['id']
        question = row['question']

        prompt_text = (
            f"You are an expert map reader. Analyze the text labels on this map to answer the question.\n"
            f"Question: {question}\n"
            f"1: {row['option_1']}\n"
            f"2: {row['option_2']}\n"
            f"3: {row['option_3']}\n"
            f"4: {row['option_4']}\n"
            f"Answer:"
        )
        
        messages = [
            {"role": "user", "content": [{"type": "image", "image": stitched_map}, {"type": "text", "text": prompt_text}]}
        ]
        
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        next_token_logits = outputs.logits[0, -1, :]

        target_logits = next_token_logits[valid_token_ids_tensor]

        probabilities = torch.nn.functional.softmax(target_logits, dim=-1)

        best_index = torch.argmax(probabilities).item()
        confidence = probabilities[best_index].item()
        prediction = best_index + 1

        if confidence < CONFIDENCE_THRESHOLD:
            prediction = 5
            
        results.append({
            "id": q_id,
            "question_num": q_id,
            "option": prediction
        })
        del inputs, outputs, next_token_logits, target_logits, probabilities
        gc.collect()
        torch.cuda.empty_cache()

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Complete. Output strictly formatted and saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    result = Image.open("stitched_sift_final.png")
    run_inference(result)