# requires: pip install torch torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ---- toy dataset (use your real CSV import instead) ----
objects = ["dog","cat","sparrow","goldfish","apple","car","eagle","tree","shark","airplane"]
answers = np.array([
 [1,1,-1,-1,1,-1,-1,1,1,-1,0,-1,1,0,-1,1,-1,-1,-1,0],  # dog
 [1,1,-1,-1,1,-1,-1,1,1,-1,0,-1,1,0,-1,1,-1,-1,-1,0],  # cat (similar to dog)
 [1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,0,-1,1,-1,-1,1,-1], # sparrow
 [-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1,-1], # goldfish
 [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,0,-1,-1,-1,-1,-1,-1], # apple
 [-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1], # car
 [1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,0,-1,1,-1,-1,1,-1], # eagle
 [-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,0,-1,-1,-1,-1,-1,-1], # tree
 [-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1,1], # shark
 [-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,1], # airplane
], dtype=np.float32)

# encode labels
le = LabelEncoder()
y = le.fit_transform(objects)        # 0..9
y = torch.tensor(y, dtype=torch.long)
X = torch.tensor(answers, dtype=torch.float32)

# simple MLP
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

model = MLP(n_in=20, n_hidden=64, n_out=len(objects))
opt = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# train (very small dataset: do many epochs or augment)
for epoch in range(400):
    model.train()
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        print(epoch, loss.item())

# inference: partial answers example
model.eval()
# suppose user answered q1 = yes, q3 = yes, q9 = no  -> build partial vector
partial = np.zeros(20, dtype=np.float32)
partial[0] = 1    # q1 yes
partial[2] = 1    # q3 yes
partial[8] = -1   # q9 no
with torch.no_grad():
    logits = model(torch.tensor([partial]))
    probs = torch.softmax(logits, dim=-1).numpy()[0]
top_idx = probs.argmax()
print("Top guess:", le.inverse_transform([top_idx])[0])
print("Top 3:", [(le.inverse_transform([i])[0], float(probs[i])) for i in probs.argsort()[-3:][::-1]])
