"""
play_20q.py

Interactive 20-questions style guessing game using a tiny PyTorch MLP.
Answers: yes / no / idk  (or y/n/i)
Commands: guess  restart  quit
"""

import sys
import random
import math

# ---- Data: objects and canonical answers (1=yes, -1=no, 0=idk/unknown) ----
objects = ["dog","cat","sparrow","goldfish","apple","car","eagle","tree","shark","airplane"]
answers = [
 [1,1,-1,-1,1,-1,-1,1,1,-1,0,-1,1,0,-1,1,-1,-1,-1,0],  # dog
 [1,1,-1,-1,1,-1,-1,1,1,-1,0,-1,1,0,-1,1,-1,-1,-1,0],  # cat (same canonical as dog here)
 [1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,0,-1,1,-1,-1,1,-1], # sparrow
 [-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1,-1], # goldfish
 [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,0,-1,-1,-1,-1,-1,-1], # apple
 [-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1], # car
 [1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,0,-1,1,-1,-1,1,-1], # eagle
 [-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,0,-1,-1,-1,-1,-1,-1], # tree
 [-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1,1], # shark
 [-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,1], # airplane
]

questions = [
 "Is it alive?",
 "Is it a mammal?",
 "Can it fly?",
 "Does it live underwater?",
 "Is it larger than a breadbox?",
 "Is it commonly kept as a pet?",
 "Is it man-made (not natural)?",
 "Is it found outdoors?",
 "Is it used for transportation?",
 "Is it typically used by humans?",
 "Is it edible?",
 "Is it a household object?",
 "Does it have fur or hair?",
 "Is it powered by electricity?",
 "Is it dangerous to humans?",
 "Does it have feathers?",
 "Is it found in forests?",
 "Is it nocturnal?",
 "Does it live in the ocean?",
 "Can it talk (imitate speech)?"
]

# ---- simple utilities ----
def parse_answer(s):
    s = s.strip().lower()
    if s in ("yes","y","1","true","t"): return 1.0
    if s in ("no","n","0","false","f"): return -1.0
    if s in ("idk","i","dont know","unknown","?","skip"): return 0.0
    return None

# ---- Try to import PyTorch; if unavailable, we'll use fallback NN=False ----
USE_TORCH = True
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    USE_TORCH = False

# ---- If torch available: define/train an MLP ----
model = None
label_to_idx = {o:i for i,o in enumerate(objects)}
idx_to_label = {i:o for o,i in label_to_idx.items()}

if USE_TORCH:
    torch.manual_seed(0)
    X = torch.tensor(answers, dtype=torch.float32)
    y = torch.tensor([label_to_idx[o] for o in objects], dtype=torch.long)

    class MLP(nn.Module):
        def __init__(self, n_in, n_hidden, n_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_out)
            )
        def forward(self,x):
            return self.net(x)

    model = MLP(n_in=20, n_hidden=48, n_out=len(objects))
    opt = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()

    # Train quickly (toy dataset: a few hundred epochs)
    model.train()
    for ep in range(400):
        logits = model(X)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

# ---- Fallback scoring if torch isn't available ----
def fallback_scores(partial_vec):
    """
    Simple scoring: for each object compute a compatibility score:
    +2 for matching non-zero answer, -1 for mismatching non-zero,
    0 for unanswered (0).
    Then convert to softmax-like probabilities.
    """
    scores = []
    for obj_ans in answers:
        s = 0.0
        for u, a in zip(partial_vec, obj_ans):
            if u == 0.0:
                continue
            if u == a:
                s += 2.0
            else:
                s -= 1.0
        scores.append(s)
    # softmax-ish
    maxs = max(scores)
    exps = [math.exp(s - maxs) for s in scores]
    tot = sum(exps)
    probs = [e/tot for e in exps]
    return probs

# ---- interactive loop ----
def interactive():
    print("\nThink of ONE object from this list (don't tell me):")
    print(", ".join(objects))
    print("\nAnswer with: yes / no / idk  (or y / n / i). Commands: guess, restart, quit\n")
    while True:
        partial = [0.0]*20
        asked = [False]*20
        q_order = list(range(20))  # use fixed order; you can randomize or choose by info gain
        q_idx = 0
        while True:
            # If we've asked all questions -> force guess
            if q_idx >= len(q_order):
                print("\nWe've asked all questions — I'll make my final guess now.")
                make_and_show_guess(partial)
                break

            qi = q_order[q_idx]
            asked[qi] = True
            print(f"\nQuestion {q_idx+1}: {questions[qi]}")
            user = input("> ").strip()
            if user.lower() == "quit":
                print("Bye!")
                return
            if user.lower() == "restart":
                print("Restarting a new round — think of a new object.")
                break  # break inner -> restart outer while True
            if user.lower() == "guess":
                make_and_show_guess(partial)
                # after guess, ask if correct
                ok = input("Was I correct? (yes/no) > ").strip().lower()
                if ok in ("yes","y"):
                    print("Nice! Want to play again? (yes/no)")
                    again = input("> ").strip().lower()
                    if again in ("yes","y"):
                        print("Great — think of a new object.")
                        break
                    else:
                        print("Thanks for playing — bye!")
                        return
                else:
                    print("Okay — I'll keep asking.")
                    # continue asking next question
                    q_idx += 1
                    continue

            parsed = parse_answer(user)
            if parsed is None:
                print("Couldn't interpret that — answer yes / no / idk, or try commands: guess / restart / quit.")
                continue
            partial[qi] = parsed

            # make a prediction after each answer
            probs = []
            if USE_TORCH and model is not None:
                with torch.no_grad():
                    inp = torch.tensor([partial], dtype=torch.float32)
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=-1).numpy()[0].tolist()
            else:
                probs = fallback_scores(partial)

            # show top 3 candidates
            ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            top_idx, top_p = ranked[0]
            top3 = [(idx_to_label[i], round(p,3)) for i,p in ranked[:3]]

            print("Top candidates:", top3)

            # auto-guess threshold
            if top_p >= 0.70:
                print(f"\nI'm fairly confident ({top_p:.2f}). My guess is: {idx_to_label[top_idx]}")
                yn = input("Am I right? (yes/no) > ").strip().lower()
                if yn in ("yes","y"):
                    print("Nice! Want to play again? (yes/no)")
                    again = input("> ").strip().lower()
                    if again in ("yes","y"):
                        print("Think of a new object from the list.")
                        break
                    else:
                        print("Thanks for playing — bye!")
                        return
                else:
                    print("Okay, I'll keep going.")
                    # If incorrect, continue to next question
            q_idx += 1

def make_and_show_guess(partial):
    if USE_TORCH and model is not None:
        import numpy as _np
        with torch.no_grad():
            inp = torch.tensor([partial], dtype=torch.float32)
            logits = model(inp)
            probs = torch.softmax(logits, dim=-1).numpy()[0].tolist()
    else:
        probs = fallback_scores(partial)
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    top_idx, top_p = ranked[0]
    print(f"My guess: {idx_to_label[top_idx]}  (confidence {top_p:.3f})")
    print("Full ranking (top 5):", [(idx_to_label[i], round(p,3)) for i,p in ranked[:5]])

# ---- run ----
if __name__ == "__main__":
    try:
        interactive()
    except KeyboardInterrupt:
        print("\nInterrupted — goodbye.")
        sys.exit(0)
