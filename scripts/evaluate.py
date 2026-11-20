"""Evaluate base or fine-tuned model on held-out prompts."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rewards import score
from scripts.train import SYSTEM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', default='outputs/date-grpo/heldout.csv')
    p.add_argument('--out', default='outputs/evaluation.json')
    args = p.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    with open(args.data, newline='') as f:
        rows = list(csv.DictReader(f))
    details = []
    for row in rows:
        prompt = SYSTEM + '\nDate: ' + row['prompt'] + '\nISO:'
        batch = tok(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            result = model.generate(**batch, max_new_tokens=24, do_sample=False,
                                    pad_token_id=tok.pad_token_id)
        output = tok.decode(result[0][batch['input_ids'].shape[1]:], skip_special_tokens=True).strip().split('\n')[0].strip()
        details.append({'prompt': row['prompt'], 'expected': row['expected_answer'],
                        'predicted': output, 'reward': score(output, row['expected_answer'])})
    summary = {'model': args.model, 'examples': len(details),
               'average_reward': sum(r['reward'] for r in details) / len(details),
               'exact_match_accuracy': sum(r['predicted'] == r['expected'] for r in details) / len(details),
               'details': details}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k != 'details'}, indent=2))


if __name__ == '__main__':
    main()
