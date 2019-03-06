from model import *

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

"""## Loss Computation"""

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        #if self.opt is not None:
        #    self.opt.step()
        #    self.opt.optimizer.zero_grad()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss * norm

"""## Greedy Decoding"""

# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=1, d_model=512, h=8, dropout=.0).to(device)
print(model)
#model_opt = NoamOpt(model.src_embed[0].d_model, 1, 100,
#        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
model_opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

for epoch in range(20):
    #if epoch >= 10:
    #    use_binary = True
    model.train()
    run_epoch(data_gen(V, 32, 20), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 32, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))

"""> This code predicts a translation using greedy decoding for simplicity."""

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) ).to(device)
src_mask = Variable(torch.ones(1, 1, 10) ).to(device)
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


