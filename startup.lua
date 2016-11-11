--[[
Template script for Loading necessary packages and functions.

This file is part of the CTSeg project.
Stavros Tsogkas <stavros.tsogkas@centralesupelec.fr>
Last update: October 2016
--]]
--------------------------------------------------------------------------------
-- Load modules
--------------------------------------------------------------------------------
require "torch"
require "trepl"
require "optim"
require "image"
require "nn"
config = require "config"
networks = require "networks"
misc = require "misc"


--------------------------------------------------------------------------------
-- Define training and testing functions
--------------------------------------------------------------------------------
function computeErrorStats(prediction,labels)
    -- prediction: Bx1xDxHxW, labels: BxDxHxW
    local s = {}
    prediction = prediction:float()
    labels     = labels:float()
    assert(prediction:nElement() == labels:nElement())
    s.numExamples = labels:nElement()
    s.correct  = prediction:eq(labels):sum()
    s.accuracy = s.correct/s.numExamples
    s.truePos  = labels[prediction:eq(1)]:eq(1):sum()
    s.falsePos = labels[prediction:eq(1)]:eq(2):sum()
    s.trueNeg  = labels[prediction:eq(2)]:eq(2):sum()
    s.falseNeg = labels[prediction:eq(2)]:eq(1):sum()
    s.numPos   = labels:eq(1):sum()
    s.numNeg   = labels:eq(2):sum()
    s.precision   = s.truePos/math.max(1, s.truePos + s.falsePos)
    s.recall      = s.truePos/math.max(1, s.truePos + s.falseNeg)
    s.specificity = s.trueNeg/math.max(1,s.trueNeg + s.falsePos)
    s.fmeasure    = 2 * s.precision * s.recall / math.max(1, s.precision + s.recall)
    return s
end

-- Return mini-batch of scores from previously trained networks ----------------
function getBatch(data, inds)
    local images = data.images:index(1, inds)
    local labels = data.labels:index(1, inds)
    local sz = images:size()
    local containsPositives = util.any(images:view(sz[1],-1),2):sum():gt(0)
    -- Include at least one positive example in the batch
    if not containsPositives and posSlices then
        local inds = torch.randperm(data.images:size(1))
        images[1]:copy(data.images[inds[1]])
    end
    images = images:view(sz[1],1,sz[2],sz[3]) -- add a singleton dimension
    if cuda then
        images = images:cuda()
        labels = labels:cuda()
    else
        images = images:float()
        labels = labels:float()
    end
    return images, labels
end


-- Execute a single epoch of training (or validation) -------------------------
function processEpoch(epoch, mode, data)

    if not data then return end
    local train = mode == 'train' or mode == 'training' or mode == 'Train' or mode == 'TRAIN'
    local numExamples = data.images:size(1);
    local numBatches = math.ceil(numExamples/opts.batchSize)
    local lossTotal = 0
    local inds = torch.randperm(numExamples):long() -- shuffle training examples
    if train then model:training()
    else model:evaluate()
    end
    -- Configure optimizer
    optimState = {
        learningRate = learningRate and learningRate[epoch] or opts.LR,
        learningRateDecay = learningRateDecay and learningRateDecay[epoch] or opts.LRDecay,
        weightDecay = weightDecay and weightDecay[epoch] or opts.weightDecay,
        momentum = momentum and momentum[epoch] or opts.momentum,
        dampening = 0
    }

    print('\n--------------- ' .. mode .. ' epoch ' .. epoch .. '/' .. opts.numEpochs
        ..' [#examples = ' .. numExamples .. ', batchSize = ' .. opts.batchSize
        .. ', #batches: ' .. numBatches .. '] ----------------')
    if train then print('Optimization parameters:'); print(optimState) end

    -- Train using batches
    epochTimer = epochTimer or torch.Timer(); epochTimer:reset()
    for t = 1, numExamples, opts.batchSize do
        local imageBatch, labelBatch = getBatch(data,
            inds[{ {t,math.min(t+opts.batchSize-1, numExamples)} }])

        -- Assertions
        assert(imageBatch:dim() == 4, 'Image batch should be a 5D array')
        assert(labelBatch:dim() == 3, 'Label batch should be a 4D array')

        -- closure to evaluate f(X) and df/dX (L and dL/dw)
        collectgarbage()
        local loss = 0
        local outputBatch = model:forward(imageBatch) -- BxLxDxHxW
        local function feval(x)
            gradParameters:zero()
            local gradOutput
            loss, gradOutput = networks.spatialCrossEntropy(outputBatch,labelBatch,weights)
            model:backward(imageBatch,gradOutput)
            return loss, gradParameters
        end
        if train then -- optimize on current mini-batch
            optim.sgd(feval, parameters, optimState)
        end

        -- Print stats for current batch
        local maxScores,prediction = outputBatch:max(2) -- max score over labels  (B1DHW)
        prediction, labelBatch = prediction:float():view(-1), labelBatch:float():view(-1)
        confusion:batchAdd(prediction,labelBatch) --print(confusion)
        local s, batch = computeErrorStats(prediction,labelBatch), math.ceil(t/opts.batchSize)
        lossTotal = lossTotal + loss
        printEpochMessage(lossTotal,epoch,batch,numBatches,mode,s,opts)
        if train then stats.train = s
        else stats.val = s end
    end -- for t = 1, numExamples, opts.batchSize

    -- print and reset confusion matrix and timings
    print(''); print(confusion); confusion:zero() print('')
    print('==> Time for epoch: ' .. misc.time2string(epochTimer:time().real)
        .. ', time per sample: ' .. misc.time2string(epochTimer:time().real/numExamples) )
end

-- Print epoch message with loss and accuracy information ----------------------
function printEpochMessage(lossTotal,epoch,batch,numBatches,mode,s,opts)
    local batchInfo = string.format('%s %s Epoch %d/%d, Batch %d/%d || LOSS: %-6.4f',
        mode, opts.class, epoch, opts.numEpochs, batch, numBatches, lossTotal/batch)
    local accInfo
    if numClasses == 2 then
        accInfo = string.format('  PRE: %4.1f%%  REC: %4.1f%% SPC: %4.1f%%  FM: %.3f',
            s.precision*100,s.recall*100, s.specificity*100, s.fmeasure)
    elseif numClasses > 2 then
        confusion:updateValids()
        accInfo = ''
        for i=1,numClasses do
            -- print('')
            -- print(confusion.valids)
            -- print(confusion.valids[i])
            accInfo = accInfo..string.format('  C%d: %.2f%%',i, confusion.valids[i]*100)
        end
    end
    print(batchInfo..accInfo)
end

-- Continue from checkpoint ----------------------------------------------------
function loadState(modelPath)
    local state = torch.load(modelPath)
    model = state.model
    optimState = state.optimState
    if cuda then -- copy to gpu before using for computations
        model:cuda()
    end
    parameters, gradParameters = model:getParameters()
end

-- Save progress ---------------------------------------------------------------
function saveState(modelPath)
    -- Save batch normalization stats before clearing state (which deletes them)
    local batchNormStats = {}
    for i=1,model:size() do
        local layer = model:get(i)
        if layer.__typename == 'nn.VolumetricBatchNormalization'
        or layer.__typename == 'nn.SpatialBatchNormalization' then
            batchNormStats[i] = {layer.save_mean, layer.save_std}
        end
    end

    model:clearState() -- Clear intermediate variables

    -- Copy back bnorm mean and std
    for i=1,model:size() do
        local layer = model:get(i)
        if layer.__typename == 'nn.VolumetricBatchNormalization'
        or layer.__typename == 'nn.SpatialBatchNormalization' then
            layer.save_mean = batchNormStats[i].save_mean
            layer.save_std  = batchNormStats[i].save_std
         end
    end

    -- Copy on CPU and save complete state
    local model = model:clone():float()
    local state = { model = model, optimState = optimState,
        stats = stats, opts = opts, confusion = confusion,
        paramRegimes = paramRegimes }
    torch.save(modelPath, state)
end

-- Training code ---------------------------------------------------------------
function train(data)

    -- Display network information
    print('Loaded '..data.train.images:size(1)..' training examples ('
        .. misc.size2string(data.train.images)..')')
    local numInputChannels = nets and #nets*2 or 1
    local dummyInput = torch.Tensor(opts.batchSize,numInputChannels,data.train.images:size(2),data.train.images:size(3))
    if cuda then dummyInput = dummyInput:cuda() end
    print('Network architecture: ' .. opts.net)
    print('Weight initialization: ' .. opts.initMethod .. ' with parameters '
        .. opts.initWeight .. '/' .. opts.initBias .. ' (if applicable)')
    networks.displayNetInfo(model,dummyInput)

    -- Set parameter regimes
    learningRate = torch.logspace(math.log10(opts.LR), math.log10(opts.LR/100), opts.numEpochs)
    confusion = optim.ConfusionMatrix(numClasses)
    stats = {} -- save stats in global variable
    paramRegimes = {}
    paramRegimes.learningRate = learningRate  or opts.LR
    paramRegimes.learningRateDecay = learningRateDecay or opts.LRDecay
    paramRegimes.weightDecay = weightDecay or opts.weightDecay
    paramRegimes.momentum = momentum or opts.momentum

    -- WARNING: This command goes AFTER transfering the network to the GPU!
    -- Retrieve parameters and gradients:
    -- extracts and flattens all the trainable parameters of the model into a vector
    -- Parameters are references: when model:backward is called, these are changed
    parameters,gradParameters = model:getParameters()

    local trainTimer = torch.Timer()
    for epoch = 1, opts.numEpochs do
        local function epoch2modelPath(epoch)
            return paths.concat(opts.saveDir, 'model-epoch-' .. epoch .. '.t7')
        end

        if opts.load > 0 and paths.filep(epoch2modelPath(epoch)) then
            if not paths.filep(epoch2modelPath(epoch+1)) then
                print('\n==> Last checkpoint found: loading state from epoch ' .. epoch)
                loadState(epoch2modelPath(epoch))
            end
        else
            processEpoch(epoch, 'train', data.train)
            processEpoch(epoch, 'val',   data.val)

            if opts.save > 0 then
                print('\n==> Saving model (epoch ' .. epoch .. ')...')
                saveState(epoch2modelPath(epoch))
            end
        end
    end
    print('Done training for ' .. opts.numEpochs .. ' epochs!')
    print('Total time for training: '..misc.time2string(trainTimer:time().real))
end


-- Evaluation code -------------------------------------------------------------
function evaluate(data)
    -- Setup evaluation
    print(''); print('Setting up evaluation...')
    model:evaluate()
    confusion = optim.ConfusionMatrix(numClasses); confusion:zero()
    softmax = nn.SpatialSoftMax()
    stats  = {}
    -- Move to gpu
    if cuda then
        softmax = softmax:cuda()
    end


    -- Test and compute stats for each volume
    local testTimer = torch.Timer()
    local volTimer  = torch.Timer()
    local numTest = #data - opts.numTrain - opts.numVal
    print('Testing on '..numTest..' volumes...')
    for i=opts.numTrain+opts.numVal+1,#data do
        if data[i] then
            print('Testing on volume ' .. i - opts.numTrain - opts.numVal .. '/' .. numTest .. '...')
            local img, seg  = data[i].img.data:float(), data[i].seg.data:float() -- DxHxW
            local scores    = torch.FloatTensor(numClasses,img:size(1),img:size(2),img:size(3))
            -- Remove extreme values and fix groundtruth seg. Our assumption is 
            -- that 0 corresponds to examples that are ignored and numClasses+1
            -- corresponds to background pixels.
            seg[seg:eq(0)] = numClasses
            if cuda then
                img = img:cuda()
                scores = scores:cuda()
            end

            -- Compute scores for each slice individually
            volTimer:reset()
            for j=1,img:size(1) do
                scores[{ {}, j, {}, {} }]:copy(model:forward(img[j]:view(1,1,img:size(2),img:size(3))))
            end
            print('Time for '..img:size(2)..'x'..img:size(3)..'x'..img:size(1)..' volume:'
            ..misc.time2string(volTimer:time().real))
            scores = softmax:forward(scores:view(scores:size(1),scores:size(2),-1)):float()
            assert(torch.all(scores:ge(0)) and torch.all(scores:le(1)))
            local maxScores, predictions = scores:max(1)
            assert(torch.all(predictions:ge(1)) and torch.all(predictions:le(numClasses)))
            assert(predictions:nElement() == seg:nElement())
            -- Compute stats
            confusion:batchAdd(predictions:view(-1),seg:view(-1))
            stats[#stats+1] = computeErrorStats(predictions, seg)
            print(confusion) print('')
            collectgarbage()
        else
            print('Skipping volume '..i..' (empty)')
        end
    end
    print('Total time for testing: '..misc.time2string(testTimer:time().real))

    -- Save results on test data
    print('')
    print('===> Performance on the full test set:')
    print(confusion)
    torch.save(paths.concat(opts.saveDir, 'testSetPerformance.t7'),
        {stats = stats, confusion = confusion})
end
