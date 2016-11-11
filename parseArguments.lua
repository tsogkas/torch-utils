--[[
Default argument parsing for training a neural network.
--]]
cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a network for semantic segmentation')
cmd:text()
cmd:text('===> Default options')
-- Training parameters ---------------------------------------------------------
cmd:text('===> Training parameters')
cmd:option('-class',      'parotid-left',   'organ class')
cmd:option('-LR',                 0.1,      '(starting) learning rate')
cmd:option('-weightDecay',        5e-4,     'L2 penalty on the weights')
cmd:option('-momentum',           0.9,      'momentum')
cmd:option('-batchSize',          8,        'batch size')
cmd:option('-seed',               123,      'torch manual random number generator seed')
cmd:option('-numEpochs',          25,       'number of epochs to train')
cmd:option('-weightedLoss',       0,        'asymmetric loss function based on label frequency')
-- Data options (#images, preprocessing etc.) ---------------------------------
cmd:text('===> Data parameters')
cmd:option('-numTrain',           40,       '#volumes used for training')
cmd:option('-numVal',             5,        '#volumes used for validation')
cmd:option('-flip',               1,        'add horizontally flipped data')
cmd:option('-rotate',             0,        'add rotated versions of data')
cmd:option('-normalize',          0,        'normalize data (zero mean, unit variance)')
-- Network and weight initialization options -----------------------------------
cmd:text('===> Network and weight initialization options')
cmd:option('-net',                'CADC',   'network architecture')
cmd:option('-initMethod',         'reset',  'weight initialization method')
cmd:option('-initWeight',         0.01,     'weight initialization parameter')
cmd:option('-initBias',           0.01,     'bias initialization parameter')
-- Miscellaneous (device and storing options) ---------------------------------
cmd:text('===> Miscellaneous options')
cmd:option('-gpu',                0,        'device ID (positive if using CUDA)')
cmd:option('-save',               1,        'save model state every epoch')
cmd:option('-load',               1,        'continue from last checkpoint')
cmd:option('-tag',                '',       'additional user-tag')

-- Parse arguments
opts = cmd:parse(arg or {})

-- Use 'float' as the default data type
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opts.seed)


-- CUDA?
if opts.gpu > 0 then
    cuda = true
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opts.gpu)
    cutorch.manualSeed(opts.seed)
end
