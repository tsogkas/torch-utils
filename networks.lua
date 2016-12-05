--[[
Package with definitions for popular architectures, and various utility
functions to compute and display information.

Stavros Tsogkas <tsogkas@cs.toronto.edu>
Last update: November 2016
--]]

local P = {}; networks = P;
require "nn"
local misc = require "misc"


-- ============================== NETWORKS ====================================

-- TODO: replace network input arguments to opts and modify initilizeModule functions
local Convolution,MaxPooling,ReLU,CrossMapLRN,Dropout,BatchNormalization
if cudnn then
    Convolution = cudnn.SpatialConvolution
    MaxPooling  = cudnn.SpatialMaxPooling
    CrossMapLRN = cudnn.SpatialCrossMapLRN
    BatchNorm   = cudnn.SpatialBatchNormalization
    ReLU        = cudnn.ReLU
else
    Convolution = nn.SpatialConvolution
    MaxPooling  = nn.SpatialMaxPooling
    CrossMapLRN = nn.SpatialCrossMapLRN
    BatchNorm   = nn.SpatialBatchNormalization
    ReLU        = nn.ReLU    
end
Dropout = nn.Dropout


-- "ImageNet Classification with Deep Convolutional Neural Networks" 
-- Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton (NIPS 2012)
function P.AlexNet(numClasses,opts,initMethod,...)
    -- NOTE: Instead of splitting inputs of conv2 and conv3 in two blocks we use 
    -- a full connection table
    local net = nn.Sequential()
    -- Conv 1
    net:add(Convolution(3,96, 11,11, 4,4))
    net:add(ReLU(true))
    net:add(CrossMapLRN(96))
    net:add(MaxPooling(3,3, 2,2))
    -- Conv 2
    net:add(Convolution(96,256, 5,5, 1,1, 2,2))
    net:add(ReLU(true))
    net:add(CrossMapLRN(256))
    net:add(MaxPooling(3,3, 2,2))
    -- Conv 3
    net:add(Convolution(256,384, 3,3, 1,1, 1,1))
    net:add(ReLU(true))
    -- Conv 4
    net:add(Convolution(384,384, 3,3, 1,1, 1,1))
    net:add(ReLU(true))
    -- Conv 5
    net:add(Convolution(384,256, 3,3, 1,1, 1,1))
    net:add(ReLU(true))
    net:add(MaxPooling(3,3, 2,2))
    -- Conv 6
    net:add(Convolution(256,4096, 6,6, 1,1))
    net:add(ReLU(true))
    net:add(Dropout())
    -- Conv 7
    net:add(Convolution(4096,4096, 1,1, 1,1))
    net:add(ReLU(true))
    net:add(Dropout())
    -- Classification layer
    net:add(Convolution(4096,numClasses, 1,1, 1,1))
    
    if initMethod then
        P.initializeNetWeights(net,initMethod,...)
    end

    -- TODO: add option for batch normalization

    return net
end

-- "Visualizing and Understanding Convolutional Networks" 
-- Matthew D. Zeiler, Rob Fergus (ECCV 2014)
function P.ZF(numClasses,opts,initMethod,...)
    local net = nn.Sequential()
    -- Conv 1
    net:add(Convolution(3,96, 7,7, 2,2))
    net:add(ReLU(true))
    net:add(CrossMapLRN(96))
    net:add(MaxPooling(3,3, 2,2))
    -- Conv 2
    net:add(Convolution(96,256, 5,5, 1,1, 2,2))
    net:add(ReLU(true))
    net:add(CrossMapLRN(256))
    net:add(MaxPooling(3,3, 2,2))
    -- Conv 3
    net:add(Convolution(256,384, 3,3, 1,1, 1,1))
    net:add(ReLU(true))
    -- Conv 4
    net:add(Convolution(384,384, 3,3, 1,1, 1,1))
    net:add(ReLU(true))
    -- Conv 5
    net:add(Convolution(384,256, 3,3, 1,1, 1,1))
    net:add(ReLU(true))
    net:add(MaxPooling(3,3, 2,2))
    -- Conv 6
    net:add(Convolution(256,4096, 6,6, 1,1))
    net:add(ReLU(true))
    net:add(nn.Dropout())
    -- Conv 7
    net:add(Convolution(4096,4096, 1,1, 1,1))
    net:add(ReLU(true))
    net:add(nn.Dropout())
    -- Classification layer
    net:add(Convolution(4096,numClasses, 1,1, 1,1))
    
    if initMethod then
        P.initializeNetWeights(net,initMethod,...)
    end

    return net

end

-- "Gradient-Based Learning Applied to Document Recognition"
-- Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner (Proc IEEE 1998)
function P.LeNet(numClasses,opts,initMethod,...)
    local net = nn.Sequential()
    -- Conv 1
    net:add(Convolution(1,20, 5,5))
    net:add(MaxPooling(2,2, 2,2))
    -- Conv 2
    net:add(Convolution(20,50, 5,5))
    net:add(MaxPooling(2,2, 2,2))
    -- Conv 3
    net:add(Convolution(50,500, 4,4))
    net:add(ReLU(true))
    -- Classification layer
    net:add(Convolution(500,numClasses, 4,4))

    if initMethod then
        P.initializeNetWeights(net,initMethod,...)
    end

    return net
end

function P.GoogLeNet(numClasses,opts,initMethod)
end

-- Architectures described in the paper "Very Deep Convolutional Networks for 
-- Large-Scale Image Recognition" by Karen Simonyan and Andrew Zisserman (ICLR 2015)
function P.VGG(numClasses,numLayers,opts,initMethod,...)

    local function ConvolutionReLU(...)
        local arg = {...}
        local s = nn.Sequential()
        s:add(Convolution(...))
        if opts.batchNormalization then s:add(BatchNorm(arg[2])) end
        s:add(ReLU(true))
        return s
    end

    local net = nn.Sequential()
    -- ConvBlock 1
    net:add(ConvolutionReLU(3, 64, 3,3))
    if numLayers > 11 then net:add(ConvolutionReLU(64,64, 3,3)) end
    net:add(MaxPooling(2,2, 2,2))
    -- ConvBlock 2
    net:add(ConvolutionReLU(64, 128, 3,3))
    if numLayers > 11 then net:add(ConvolutionReLU(128,128, 3,3)) end
    net:add(MaxPooling(2,2, 2,2))
    -- ConvBlock 3
    net:add(ConvolutionReLU(128,256, 3,3))
    net:add(ConvolutionReLU(256,256, 3,3))
    if numLayers > 13 then net:add(ConvolutionReLU(256,256, 3,3)) end
    if numLayers > 19 then net:add(ConvolutionReLU(256,256, 3,3)) end
    net:add(MaxPooling(2,2, 2,2))
    -- ConvBlock 4
    net:add(ConvolutionReLU(256,512, 3,3))
    net:add(ConvolutionReLU(512,512, 3,3))
    if numLayers > 13 then net:add(ConvolutionReLU(512,512, 3,3)) end
    if numLayers > 16 then net:add(ConvolutionReLU(512,512, 3,3)) end
    net:add(MaxPooling(2,2, 2,2))
    -- ConvBlock 5
    net:add(ConvolutionReLU(512,512, 3,3))
    net:add(ConvolutionReLU(512,512, 3,3))
    if numLayers > 13 then net:add(ConvolutionReLU(512,512, 3,3)) end
    if numLayers > 16 then net:add(ConvolutionReLU(512,512, 3,3)) end
    net:add(MaxPooling(2,2, 2,2))
    -- ConvBlock 6
    net:add(ConvolutionReLU(512,4096,7,7))
    net:add(Dropout())
    -- ConvBlock 7
    net:add(ConvolutionReLU(4096,4096,1,1))
    net:add(Dropout())
    -- Classification layer
    net:add(Convolution(4096,1000,1,1))
    
    if initMethod then
        P.initializeNetWeights(net,initMethod,...)
    end

    return net
end

-- "Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs"
-- Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan Yuille (ICLR 2015)
function P.DeepLab(numClasses,opts,initMethod,...)
end

-- "Deep Residual Learning for Image Recognition"
-- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (CVPR 2015)
function P.ResNet(numClasses,numLayers,opts,initMethod)
end

-- "Multi-scale Context Aggregation by Dilated Convolutions" 
-- Fisher Yu and Vladlen Koltun (ICLR 2016)
function P.CADC(numClasses,opts,initMethod,...)
    local net = nn.Sequential()

    local function SpatialConvBNormReLU(...)
        local arg = {...}
        net:add(nn.SpatialDilatedConvolution(...))
        net:add(nn.SpatialBatchNormalization(arg[2]))
        net:add(nn.ReLU(true))
    end

    local C = numClasses
    SpatialConvBNormReLU(1,C, 3,3, 1,1, 1,1, 1,1)
    SpatialConvBNormReLU(C,C, 3,3, 1,1, 1,1, 1,1)
    SpatialConvBNormReLU(C,C, 3,3, 1,1, 2,2, 2,2)
    SpatialConvBNormReLU(C,C, 3,3, 1,1, 4,4, 4,4)
    SpatialConvBNormReLU(C,C, 3,3, 1,1, 8,8, 8,8)
    SpatialConvBNormReLU(C,C, 3,3, 1,1, 16,16, 16,16)
    SpatialConvBNormReLU(C,C, 3,3, 1,1, 1,1, 1,1)
    -- Classification layer
    net:add(nn.SpatialConvolution(C, numClasses, 1,1))

    if initMethod then
        P.initializeNetWeights(net,initMethod,...)
    end

    return net
end


-- ========================== CHECK LAYER TYPES ===============================
-- TODO: make these methods package methods???
local function isTemporalConvolution(m)
    return m.__typename == 'nn.TemporalConvolution'
        or m.__typename == 'cudnn.TemporalConvolution'
end

local function isSpatialConvolution(m)
    return m.__typename == 'nn.SpatialConvolution'
        or m.__typename == 'nn.SpatialConvolutionLocal'
        or m.__typename == 'nn.SpatialConvolutionMM'
        or m.__typename == 'nn.SpatialConvolutionMap'
        or m.__typename == 'nn.SpatialDilatedConvolution'
        or m.__typename == 'nn.SpatialFullConvolution'
        or m.__typename == 'nn.SpatialFullConvolutionMap'
        or m.__typename == 'cudnn.SpatialConvolution'
        or m.__typename == 'cudnn.SpatialFullConvolution'
end

local function isVolumetricConvolution(m)
    return m.__typename == 'nn.VolumetricConvolution'
        or m.__typename == 'nn.VolumetricFullConvolution'
        or m.__typename == 'cudnn.VolumetricConvolution'
end

local function isConvolution(m)
    return isTemporalConvolution(m)
        or isSpatialConvolution(m)
        or isVolumetricConvolution(m)
end

-- ======================= INITIALIZATION METHODS =============================

--TODO: initialize biases to zero (in normal/uniform method) ?
--TODO: create tester with all these methods
--TODO: explicitly turn initalization method names to lowercase
--TODO: make initializeNetWeights method local again? (mmust place it before network definitions)

-- Table with module initialization methods ------------------------------------
local initializeModule = {}
initializeModule['zero'] = function(m)        -- Set all weights to zero (0)
    m.weight:zero()
    if m.bias then
        m.bias:zero()
    end
end
initializeModule['constant'] = function(m,arg) -- Initialize to fixed value
    local wval = arg[1] or 0
    local bval = arg[2]
    m.weight:fill(wval)
    if m.bias and bval then
        m.bias:fill(bval)
    end
end
initializeModule['reset'] = function(m,arg)   -- use default module method (uniform)
    m:reset(arg[1])
end
initializeModule['uniform'] = function(m,arg) -- uniform distribution with given std
    local wstd = arg[1] or 0.01
    local bstd = arg[2]
    wstd = wstd * math.sqrt(3)
    m.weight:uniform(-wstd,wstd)
    if m.bias and bstd then
        bstd = bstd * math.sqrt(3)
        m.bias:uniform(-bstd,bstd)
    end
end
initializeModule['normal'] = function(m,arg)  -- gaussian distribution with given std
    local wstd = arg[1] or 0.01
    local bstd = arg[2]
    m.weight:normal(-wstd,wstd)
    if m.bias and bstd then
        m.bias:normal(-bstd,bstd)
    end
end
initializeModule['xavier'] = function(m,arg) -- Xavier initialization based on fan-in/fan-out
    local mode = arg[1] or 'default'
    local fanIn, fanOut
    if isTemporalConvolution(m) then
        --TODO: complete this
    elseif isSpatialConvolution(m) then
        fanIn = m.nInputPlane * m.kH * m.kW
        fanOut = m.nOutputPlane * m.kH * m.kW
    elseif isVolumetricConvolution(m) then
        fanIn = m.nInputPlane * m.kH * m.kW * m.kT
        fanOut = m.nOutputPlane * m.kH * m.kW * m.kT
    end
    if mode == 'default' then
        local std = math.sqrt(6/(fanIn + fanOut))  -- sqrt(2) * sqrt(3)
        m.weight:uniform(-std,std)
    elseif mode == 'caffe' then
        local std = math.sqrt(1/fanIn)
        m.weight:uniform(-std,std)
    elseif mode == 'heuristic' then
        local std = 1/math.sqrt(fanIn)
        m.weight:uniform(-std,std)
    elseif mode == 'improved' then
        local std = math.sqrt(2/fanIn)
        m.weight:normal(-std,std)
    else error('Invalid Xavier initialization type')
    end
    if m.bias then
        m.bias:zero()
    end
end
-- Add support for single-word arguments
initializeModule['xavier-default']   = function(m)
    return initializeModule['xavier'](m,'default') end
initializeModule['xavier-caffe']     = function (m)
    return initializeModule['xavier'](m,'caffe') end
initializeModule['xavier-heuristic'] = function(m)
    return initializeModule['xavier'](m,'heuristic') end
initializeModule['xavier-improved']  = function(m)
    return initializeModule['xavier'](m,'improved') end

-- Basic wrapper to initialize weights of convolutional layers in a network ----
function P.initializeNetWeights(net, method, ...)
    local arg = {...}
    if not initializeModule[method] then
        error('Initialization method is not supported')
    end

    -- Do the work
    for i=1,#net do
        local m = net.modules[i]
        if isConvolution(m) then
            initializeModule[method](m,arg)
        end
    end
end

-- Initialize conv weights and (optionally) biases at fixed value -------------
function P.initializeConstant(net, wval, bval)
    P.initializeNetWeights(net,'constant',wval,bval)
end

-- Set all weights to zero. If you want different values for weights and biases
-- use P.initializeConstant.
function P.initializeZero(net)
    P.initializeNetWeights(net,'zero')
end

-- Uniform initialization and (optionally) biases -----------------------------
function P.initializeUniform(net,wstd,bstd)
    P.initializeNetWeights(net,'uniform',wstd,bstd)
end

-- Use module default initialization method (affects both weight and biases) ---
function P.initializeReset(net,std)
    P.initializeNetWeights(net,'reset',std)
end

-- Gaussian distribution with given std (optionally for biases too) -----------
function P.initializeNormal(net,wstd,bstd)
    P.initializeNetWeights(net,'normal',wstd,bstd)
end

-- "Efficient backprop" heuristic initialization (Yann Lecun, 1998) -----------
-- Only convolutional weights are affected.
function P.initializeHeuristic(net)
    P.initializeNetWeights(net,'xavier','heuristic')
end

-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
function P.initializeXavier(net)
    P.initializeNetWeights(net,'xavier','default')
end

function P.initializeXavierCaffe(net)
    P.initializeNetWeights(net,'xavier','caffe')
end

-- "Delving Deep into Rectifiers: Surpassing Human-Level Performance on Imagenet
-- Classification", Kaiming He, 2015.
function P.initializeXavierImproved(net)
    P.initializeNetWeights(net,'xavier','improved')
end





-- =============================================================================
-- ===================== NETWORK UTILITY FUNCTIONS =============================
-- =============================================================================

-- Display useful network information such as dimensions of data at each layer,
-- size of receptive fields, memory requirements per layer as well as for the
-- complete network etc. -------------------------------------------------------
function P.displayNetInfo(net,input,train)
    local function displaySequentialNetInfo(net,input,train)
        local train = train or true
        if train == true or train == 1 then
            net:training()
            net:backward(input,net:forward(input))
        else
            net:evaluate()
            net:forward(input)
         end

        local sizeof = misc.sizeof
        local info = {}
        info.numParamTotal = 0
        info.memParamTotal = 0
        info.memDataTotal  = 0
        info.memGradTotal  = 0
        info.memTotal      = 0
        for i=1,net:size() do
            info[i] = {}
            info[i].numParam = 0
            info[i].memParam = 0
            info[i].memData  = 0
            info[i].memGrad  = 0
            info[i].memLayer = 0
            for field,val in pairs(net:get(i)) do
                if torch.isTensor(val) then
                    local memVal = sizeof(val)
                    -- Parameters
                    if field == 'weight' or field == 'bias' then
                        info[i].numParam = info[i].numParam + val:nElement()
                        info[i].memParam = info[i].memParam + memVal
                    end
                    -- Data (stored in intermediate layers)
                    if field == 'output' then
                        info[i].sizeData = val:size()
                        info[i].memData  = info[i].memData + memVal
                    end
                    if field == 'finput' then
                        info[i].memData  = info[i].memData + memVal
                    end
                    -- Gradients
                    if field == 'gradWeight' or field == 'gradBias'
                    or field == 'gradInput'  or field == 'fgradInput'
                    then info[i].memGrad = info[i].memGrad + memVal end
                    -- Total layer memory
                    info[i].memLayer = info[i].memLayer + memVal
                end
            end
            info[i].receptiveFieldSize = P.getReceptiveFieldSizeAtLayer(net,i)
            info.numParamTotal = info.numParamTotal + info[i].numParam
            info.memParamTotal = info.memParamTotal + info[i].memParam
            info.memDataTotal  = info.memDataTotal  + info[i].memData
            info.memGradTotal  = info.memGradTotal  + info[i].memGrad
            info.memTotal      = info.memTotal      + info[i].memLayer
        end

        -- Print network info
        local function dim2string(dim)
            local str = dim[1]
            local numDim
            if torch.isTensor(dim) then numDim = dim:nElement()
            else numDim = #dim end
            for i=2,numDim do str = str .. 'x' .. dim[i] end
            return str
        end
        local printSize = misc.size2string
        print('')
        print('---------------------- Network details: ----------------------')
        print(string.format('%-10s%-10s%-13s%-20s%-10s%-13s',
            ' ','#params','param mem','data dims','RF size','Layer mem'))
        print(string.format('%-10s%-10s%-13s%-20s%-10s%-13s',
            'Input', 'NaN', 'NaN', dim2string(input:size()), 'NaN', printSize(sizeof(input))))
        for i=1,net:size() do
            print(string.format('Layer %-2d: %-10d%-13s%-20s%-10s%-13s',
                i,info[i].numParam, printSize(info[i].memParam), dim2string(info[i].sizeData),
                dim2string(info[i].receptiveFieldSize), printSize(info[i].memLayer) ))
        end
        print('')
        print('--------------------- Memory requirements ---------------------')
        print('Total number of parameters: ' .. info.numParamTotal)
        print('Total memory for parameters: ' .. printSize(info.memParamTotal))
        print('Total memory for data: ' .. printSize(info.memDataTotal) )
        print('Total memory for gradients: ' .. printSize(info.memGradTotal))
        print('Total memory for network: ' .. printSize(info.memTotal) )
        print('')
        return info
    end

    -- TODO: fix size in ram computation for networks with dilated convolutions
    -- TODO: add support for more network types
    if torch.typename(net) == 'nn.Sequential'  then
        return displaySequentialNetInfo(net, input, train)
    else
        error('Network type not supported')
    end
end

-- Compute the dimensions of the output at each layer of the network -----------
function P.getOutputSizeAtLayer(net,layerIndex,input)
    assert(layerIndex > 0 and layerIndex <= net:size(), 'Layer index is out of bounds')
    local inputSize
    if torch.isTensor(input) then
        inputSize = torch.Tensor(input:size():totable())
    elseif torch.type(input) == 'torch.LongStorage' then
        inputSize = torch.Tensor(input:totable())
    elseif type(input) == 'table' then
        inputSize = torch.Tensor(input)
    else  error('Input must be a tensor, a tensor:size() or a table')
    end

    local outputSize    = {}
    local outputSizeMem = 0
    for i = 1,layerIndex do
        l = net:get(i)
        if torch.typename(l) == 'nn.VolumetricConvolution' then
            assert(inputSize:nElement() == 4 or inputSize:nElement() == 5,
                'Input size must be 4D (single input) or 5D (batch)')
            local owidth = math.floor((inputSize[-1] + 2*l.padW - l.kW)/l.dW + 1)
            local oheight = math.floor((inputSize[-2] + 2*l.padH - l.kH)/l.dH + 1)
            local otime = math.floor((inputSize[-3] + 2*l.padT - l.kT)/l.dT + 1)
            if inputSize:nElement() == 5 then
                outputSize = torch.Tensor({inputSize[1],l.nOutputPlane,otime,oheight,owidth})
            else
                outputSize = torch.Tensor({l.nOutputPlane,otime,oheight,owidth})
            end
            outputSizeMem = misc.sizeof(l._type) * outputSize:prod()
            inputSize = outputSize
        elseif torch.typename(l) == 'nn.SpatialConvolution' then
            assert(inputSize:nElement() == 3 or inputSize:nElement() == 3,
                'Input size must be 3D (single input) or 4D (batch)')
            local owidth  = math.floor((inputSize[-1] + 2*l.padW - l.kW)/l.dW + 1)
            local oheight = math.floor((inputSize[-2] + 2*l.padH - l.kH)/l.dH + 1)
            if inputSize:nElement() == 4 then
                outputSize = torch.Tensor({inputSize[1],l.nOutputPlane,oheight,owidth})
            else
                outputSize = torch.Tensor({l.nOutputPlane,oheight,owidth})
            end
            outputSizeMem = misc.sizeof(l._type) * outputSize:prod()
            inputSize = outputSize
        elseif torch.typename(l) == 'nn.SpatialDilatedConvolution' then
            assert(inputSize:nElement() == 3 or inputSize:nElement() == 3,
                'Input size must be 3D (single input) or 4D (batch)')
            local owidth  = math.floor((inputSize[-1] + 2*l.padW - l.dilationW*(l.kW-1))/l.dW + 1)
            local oheight = math.floor((inputSize[-2] + 2*l.padH - l.dilationH*(l.kH-1))/l.dH + 1)
            if inputSize:nElement() == 4 then
                outputSize = torch.Tensor({inputSize[1],l.nOutputPlane,oheight,owidth})
            else
                outputSize = torch.Tensor({l.nOutputPlane,oheight,owidth})
            end
            outputSizeMem = misc.sizeof(l._type) * outputSize:prod()
            inputSize = outputSize
        elseif torch.typename(l) == 'nn.ReLU' or torch.typename(l) == 'nn.Dropout' then
            outputSize = inputSize
            if l.inplace then outputSizeMem = 0 -- if inplace, no extra memory
            else outputSizeMem = misc.sizeof(l._type) * outputSize:prod()
            end
        else error('Layer type not supported yet!')
        end
    end
    return outputSize, outputSizeMem
end

function P.getReceptiveFieldSizeAtLayer(net,layerEnd, layerStart)
    local layerStart = layerStart or 1
    assert(layerStart > 0 and layerStart <= net:size(), 'Reference layer is out of bounds')
    assert(layerEnd > 0 and layerEnd <= net:size(), 'Destination layer is out of bounds')

    --TODO: integrate receptive field and output size computation. outputsize at
    -- layer L is given by: osize(L) = floor((insize - rfsize(L))/strides(L) + 1)
    -- NOT SURE THAT'S THE CASE (SEE Kamnitsas and Glocker paper)
    local rfsize  = torch.ones(1)
    local strides = torch.ones(1)
    for i = layerStart,layerEnd do
        l = net:get(i)
        ltype = torch.typename(l)
        if  ltype == 'nn.VolumetricConvolution' then
            strides = torch.Tensor({l.dT, l.dH, l.dW}):cmul(strides:expand(3))
            rfsize = torch.Tensor({l.kT, l.kH, l.kW}):csub(1):cmul(strides):add(rfsize:expand(3))
        elseif ltype == 'nn.SpatialConvolution' then
            strides = torch.Tensor({l.dH, l.dW}):cmul(strides:expand(2))
            rfsize = torch.Tensor({l.kH, l.kW}):csub(1):cmul(strides):add(rfsize:expand(2))
        elseif ltype == 'nn.SpatialDilatedConvolution' then
            strides = torch.Tensor({l.dH*l.dilationH, l.dW*l.dilationW}):cmul(strides:expand(2))
            rfsize = torch.Tensor({l.kH, l.kW}):csub(1):cmul(strides):add(rfsize:expand(2))
        end
    end
    return rfsize
end

function P.getNetSizeInMem(net)
    local memTotal = 0
    local sizeof = misc.sizeof
    for i=1,net:size() do
        for field,val in pairs(net:get(i)) do
            if torch.isTensor(val) then
                memTotal = memTotal + sizeof(val)
            end
        end
    end
    return memTotal
end



-- ========================== NETWORK LAYERS ==================================
function P.__volumetricCrossEntropy(input,target,weights)
    assert(input:dim()   == 5, 'mini-batch supported only')
    assert(target:dim()  == 4, 'mini-batch supported only')
    assert(input:size(1) == target:size(1), 'Input and target batchsize do not match')
    assert(input:size(3) == target:size(2), 'Input and target depth do not match')
    assert(input:size(4) == target:size(3), 'input and target height do not match')
    assert(input:size(5) == target:size(4), 'input and target width do not match')

    -- Define classification subnetwork
    require 'nn'
    local crossEntropy3D = nn.Sequential()
    crossEntropy3D:add(nn.Log())
    crossEntropy3D:add(nn.SpatialSoftMax())
    crossEntropy3D:add(nn.SpatialClassNLLCriterion())

    local insz = input:size()
    local loss = crossEntropy3D(input:view(insz[1],insz[2],insz[3],-1),
        target:view(insz[1],insz[2],-1))
end

function P.spatialCrossEntropy(input, target, weights)
    --[[
        This function takes as input a a BLHW array (only works in batch mode),
        and a BHW array of targets (labels) and computes the class cross
        entropy loss and respective gradients.
        These will be subsequently used in the net:backward() method.

        INPUTS:
        input: BxLxHxW
        target: BxHxW
        weights: 1xL

        OUTPUTS:
        loss: 1x1 (scalar)
        gradLoss: BxLxHxW (same as input)
    --]]

    assert(input:dim()   == 4, 'mini-batch supported only')
    assert(target:dim()  == 3, 'mini-batch supported only')
    assert(input:size(1) == target:size(1), 'Input and target batchsize do not match')
    assert(input:size(3) == target:size(2), 'Input and target height do not match')
    assert(input:size(4) == target:size(3), 'input and target width do not match')

    require "nn"
    local crossEntropy1D = nn.CrossEntropyCriterion(weights)
    if torch.type(input) == 'torch.CudaTensor' then
        require 'cunn'
        crossEntropy1D = crossEntropy1D:cuda()
    end

    local function transpose(input)
        input = input:transpose(2,4):transpose(2,3):contiguous() -- bdhw -> bwhd -> bhwd
        input = input:view(input:size(1)*input:size(2)*input:size(3), input:size(4))
        return input
    end

    local function transposeBack(input, originalInput)
        input = input:view(originalInput:size(1), originalInput:size(3),
                           originalInput:size(4), originalInput:size(2))
        input = input:transpose(2,4):transpose(3,4):contiguous()  -- bhwd -> bdwh -> bdhw
        return input
    end

    local inputTranspose = transpose(input)
    assert(inputTranspose:dim() == 2)
    target = target:view(-1) -- flatten targets vector
    local valid = target:float():nonzero():view(-1) -- ignore examples with label 0
    local gradLoss = torch.zeros(inputTranspose:size()) -- zero loss by default
    if torch.type(input) == 'torch.CudaTensor' then
        gradLoss = gradLoss:cuda()
    end

    inputTranspose = inputTranspose:index(1,valid)
    target = target[target:gt(0)]
    assert(inputTranspose:size(1) == target:nElement())
    local loss = crossEntropy1D:forward(inputTranspose, target) -- scalar
    gradLoss:indexCopy(1,valid,crossEntropy1D:backward(inputTranspose, target))  -- B*D*H*W*L
    gradLoss = transposeBack(gradLoss, input)

    -- local inputTranspose = transpose(input)
    -- assert(inputTranspose:dim() == 2)
    -- local loss = crossEntropy1D:forward(inputTranspose, target:view(-1)) -- scalar
    -- local gradLoss = crossEntropy1D:backward(inputTranspose, target:view(-1)) -- B*D*H*W*L
    -- gradLoss = transposeBack(gradLoss, input)


    return loss,gradLoss
end

function P.volumetricCrossEntropy(input, target, weights)
    --[[
        This function takes as input a a BLDHW array (only works in batch mode),
        and a BDHW array of targets (labels) and computes the class cross
        entropy loss and respective gradients.
        These will be subsequently used in the net:backward() method.

        INPUTS:
        input: BxLxDxHxW
        target: BxDxHxW
        weights: 1xL

        OUTPUTS:
        loss: 1x1 (scalar)
        gradLoss: BxLxDxHxW (same as input)
    --]]

    assert(input:dim()   == 5, 'mini-batch supported only')
    assert(target:dim()  == 4, 'mini-batch supported only')
    assert(input:size(1) == target:size(1), 'Input and target batchsize do not match')
    assert(input:size(3) == target:size(2), 'Input and target depth do not match')
    assert(input:size(4) == target:size(3), 'input and target height do not match')
    assert(input:size(5) == target:size(4), 'input and target width do not match')

    require "nn"
    local crossEntropy1D = nn.CrossEntropyCriterion(weights)
    if torch.type(input) == 'torch.CudaTensor' then
        require 'cunn'
        crossEntropy1D = crossEntropy1D:cuda()
    end

    local transpose = function(input)
        -- BLDHW -> BWDHL -> BDWHL -> BDHWL
        input = input:transpose(2,5):transpose(2,3):transpose(3,4):contiguous()
        input = input:view(input:size(1)*input:size(2)*input:size(3)*input:size(4), input:size(5))
        return input
    end

    local transposeBack = function(input, originalInput)
        input = input:view(originalInput:size(1), originalInput:size(3),
            originalInput:size(4), originalInput:size(5), originalInput:size(2))
        -- BDHWL -> BLHWD -> BLDWH -> BLDHW
        input = input:transpose(2,5):transpose(3,5):transpose(4,5):contiguous()
        return input
    end

    local inputTranspose = transpose(input)
    assert(inputTranspose:dim() == 2)
    local loss = crossEntropy1D:forward(inputTranspose, target:view(-1)) -- scalar
    local gradLoss = crossEntropy1D:backward(inputTranspose, target:view(-1)) -- B*D*H*W*L
    gradLoss = transposeBack(gradLoss, input)

    return loss,gradLoss
end


return P
