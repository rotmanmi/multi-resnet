require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cudnn'
local models = require 'models/init'


local DataLoader = require 'dataloader'
opt = {}
opt.dataset = 'cifar10'
opt.manualSeed = 0
opt.nThreads = 4
opt.nThreads = 4
opt.tenCrop = false
opt.batchSize = 1
opt.data = ''
opt.gen = 'gen'

torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

local function factorial(n)
    if (n == 0) then
        return 1
    else
        return n * factorial(n - 1)
    end
end
--[[idx1 can be, 2,3,4
for 2:
idx2 can be 1..22
for 3,4:
idx2 can be 2..22
]]




function getLayerOutputCorrelation(model,power,idx1,idx2)
  firstlayer = model:get(idx1):get(idx2):get(1):get(1):get(1):get(1):get(10).output
  firstlayerafterpower = nn.Power(power):cuda()(firstlayer)
  firstlayerafterpower = nn.View(-1):cuda()(firstlayerafterpower)
  output = model:get(idx1):get(idx2):get(1):get(1):get(1):get(power):get(10).output
  output = nn.MulConstant(factorial(power)):cuda()(output)
  output = nn.View(-1):cuda()(output)


  correlation = torch.dot(output,firstlayerafterpower) / torch.norm(output) / torch.norm(firstlayerafterpower)

  return correlation
end

trainLoader, valLoader = DataLoader.create(opt)

model = torch.load('checkpoints/cifar10bestmodelk3coefs.t7')
model:evaluate()
model = model:cuda()


corrs = torch.Tensor(3,22,3):fill(0)

for n, sample in valLoader:run() do
  temp = sample.input:cuda()
  model:forward(temp)

  for pow=1,3 do
    for j=2,4 do
      for i=2,22 do
          corrs[j-1][i][pow] = corrs[j-1][i][pow] + getLayerOutputCorrelation(model,pow,j,i)

      end
    end
  end


end
print(corrs)

torch.save('corrs.t7',corrs)

--i goes between 1 to 3, corresponds to the power.
--outputmodule =  model:get(2):get(2):get(1):get(1):get(1):get(i):get(10).output


print('Done.')
