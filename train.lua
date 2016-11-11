--[[
Script template for training neural networks.

Stavros Tsogkas <stavros.tsogkas@centralesupelec.fr>
Last update: October 2016
--]]

dofile('startup.lua')
dofile('parseArguments.lua')

--TODO: fix size in ram computation for networks with dilated convolutions
--TODO: check if log print is fixed or if we need to explicitly print details
--TODO: maybe try to first check if the model under question exists and then
--try to load the training data to avoid unnecessary overhead

--------------------------------------------------------------------------------
-- Setup log file and load model
--------------------------------------------------------------------------------
-- We temporarily remove options that are not used to form the saveDir name.
-- We do the same thing for the the class name, so that it is explicitly placed
-- at the beginning of the directory name.
local gpu    = opts.gpu;   opts.gpu   = nil
local save   = opts.save;  opts.save  = nil
local load   = opts.load;  opts.load  = nil
local class  = opts.class; opts.class = nil
opts.saveDir = cmd:string(paths.concat(config.modelsDir, class), opts, {dir=true})
paths.mkdir(opts.saveDir)
opts.gpu = gpu; opts.save = save; opts.load = load; opts.class = class
cmd:log(opts.saveDir .. '/log-' .. os.date('%d-%m-%Y-%X') , opts)

-- Load model
numClasses = numClasses or config.numClasses
model = networks[opts.net](numClasses,opts.initMethod,opts.initWeight,opts.initBias)
if cuda then
    model:cuda()
end

-------------------------------------------------------------------------------
-- Train
-------------------------------------------------------------------------------
data = LOAD_TRAINING_DATA
print('Loaded data used for training:'); print(data)
-- This is useful when the training data are imbalanced, and we want to enforce
-- that there are at least K positive examples in every batch
print('Getting indices of slices containing positive examples')
posSlices = misc.any(data.train.labels:eq(1):view(data.train.labels:size(1),-1),2):view(-1):nonzero()
train(data)

--------------------------------------------------------------------------------
-- Evaluate
--------------------------------------------------------------------------------
data = LOAD_EVALUATION_DATA
evaluate(data)
