--[[
Various Lua and torch7 utility functions.

Stavros Tsogkas <tsogkas@cs.toronto.edu>
Last update: October 2016
--]]

local P = {}; misc = P;
require "torch"

--Softmax functions ------------------------------------------------------------
function P.softmax(x)
    assert(x:dim() == 1, 'Input must be a vector')
    local expx = torch.exp(x-x:max())
    return expx:cdiv(expx:sum())
end

function P.spatialSoftmax(x)
    if x:dim() == 3 then -- LxHxW
        local labelDim = 1
    elseif x:dim() == 4 then -- BxLxHxW
        local labelDim = 2
    else error('Input must be a 3D (LxHxW) or 4D (BxLxHxW) array')
    end
    local expx = torch.exp(x - x:max(labelDim):expandAs(x))
    return expx:cdiv(expx:sum(labelDim))
end

function P.volumetricSoftmax(x)
    if x:dim() == 4 then -- LxDxHxW
        local labelDim = 1
    elseif x:dim() == 5 then -- BxLxDxHxW
        local labelDim = 2
    else error('Input must be a 4D (LxDxHxW) or 4D (BxLxDxHxW) array')
    end
    local expx = torch.exp(x - x:max(labelDim):expandAs(x))
    return expx:cdiv(expx:sum(labelDim))
end


--------------------------------------------------------------------------------
-------------------------- SIZE, TIMING AND STRINGS ----------------------------
--------------------------------------------------------------------------------
-- Return size of torch tensor in bytes ----------------------------------------
function P.sizeof(x)
    -- TODO: make this implementation-generic (use type sizes for the current machine)
    -- TODO: add support for generic string
    if  torch.isTensor(x) then
        local sizeOfElementInBytes
        if x:type() == 'torch.DoubleTensor' or x:type() == 'torch.LongTensor' then
            sizeOfElementInBytes = 8
        elseif x:type() == 'torch.FloatTensor' or x:type() == 'torch.IntTensor' then
            sizeOfElementInBytes = 4
        elseif x:type() == 'torch.ShortTensor' then
            sizeOfElementInBytes = 2
        elseif x:type() == 'torch.CharTensor' or x:type() == 'torch.ByteTensor' then
            sizeOfElementInBytes = 1
        elseif x:type() == 'torch.CudaTensor' or x:type() == 'torch.CudaIntTensor' then
            sizeOfElementInBytes = 4
        else
            error('Unknown tensor type')
        end
        return (x:nElement() * sizeOfElementInBytes)
    elseif type(x) == 'string' then
        if x == 'torch.DoubleTensor' or x == 'torch.LongTensor' then
            return 8
        elseif x == 'torch.FloatTensor' or x == 'torch.IntTensor' then
            return 4
        elseif x == 'torch.ShortTensor' then
            return 2
        elseif x == 'torch.CharTensor' or x == 'torch.ByteTensor' then
            return 1
        elseif x == 'torch.CudaTensor' or x == 'torch.CudaIntTensor' then
            sizeOfElementInBytes = 4
        else
            error('Unknown tensor type')
        end
    elseif type(x) == 'number' then -- if it's a number we consider  this is the size
        return x
    else
        error('Only tensors and string (tensor types) are supported!')
    end
end

-- Print size of torch tensor (adjusts for size in KB, MB, GB etc) -------------
function P.size2string(x)
    local sizeInBytes = P.sizeof(x)
    local KB = 1024
    local MB = 1024^2
    local GB = 1024^3
    local str
    if sizeInBytes < KB then
        str = string.format('%d B', sizeInBytes)
    elseif sizeInBytes < MB then
        str = string.format('%.2f KB', sizeInBytes/KB)
    elseif sizeInBytes < GB then
        str = string.format('%.2f MB', sizeInBytes/MB)
    else
        str = string.format('%.2f GB', sizeInBytes/GB)
    end
    return str
end

function P.printSizeOf(x) print(P.size2string(x)) end

-- Print timings in sec, minutes, hours (assumes input in sec)) ----------------
function  P.time2string(t)
    local sec  = 1
    local min  = 60
    local hour = 60^2
    local day  = 60^3
    local str
    if t < sec then
        str = string.format('%d msec', t*1000)
    elseif t < min then
        str = string.format('%d sec', t)
    elseif t < hour then
        str = string.format('%.2f min', t/min)
    elseif t < day then
        str = string.format('%.2f hours', t/hour)
    else
        str = string.format('%.2f days', t/day)
    end
    return str
end

function P.printTime(t) print(P.time2str(t)) end

--------------------------------------------------------------------------------
-- ARITHMETIC CHECKING AND VECTOR/MATRIX MANIPULATIONS -------------------------
--------------------------------------------------------------------------------
function P.isnan(x) return x:ne(x) end
function P.isinf(x) return x:eq(math.huge) end
function P.isvec(x) return x:nDimension() == 1 end
function P.ismat(x) return x:isTensor() and x:nDimension() == 2 end
function P.isint(x) return x % 1 == 0 end
function P.isodd(x) return x % 2 == 1 end
function P.vec(x)   return x:view(-1) end

--------------------------------------------------------------------------------
-------------------------- LOGICAL OPERATIONS ----------------------------------
--------------------------------------------------------------------------------
function P.any(a,dim) -- same as torch.any() but can also work dimension-wise
    if dim then
        return a:ne(0):sum(dim):gt(0)
    else
        return torch.any(d)
    end
end
function P.all(a,dim) -- same as torch.all() but can also work dimension-wise
    if dim then
        return a:ne(0):sum(dim):eq(a:size(dim))
    else
        return torch.all(d)
    end
end
function P.land(a,b)return a:gt(0):eq(b:gt(0)) end
function P.lor(a,b) return torch.gt(a+b) end
function P.iou(a,b)
    -- TODO: does not cover case where a and b are masks that have size(2) == 4
    local a,b = function(a,b)
        if not torch.isTensor(a) then
            assert(#a==4, 'Bounding boxes should be in the form: [xmin, xmax, ymin, ymax]')
            a = torch.Tensor(a)
        end
        if not torch.isTensor(b) then
            assert(#b==4, 'Bounding boxes should be in the form: [xmin, xmax, ymin, ymax]')
            b = torch.Tensor(b)
        end
        return a,b
    end

    local max = torch.max
    local min = torch.min
    -- a,b are bounding box coordinates in the form [xmin, xmax, ymin, ymax]
    if a:nDimension() == 1 and b:nDimension() == 1 then
        assert(a:nElement() == 4 and b:nElement() == 4,
        'Bounding boxes should be in the form: [xmin, xmax, ymin, ymax]')
        local intersectionBox = torch.Tensor(4)
        intersectionBox[1] = max(a[1],b[1])
        intersectionBox[2] = max(a[2],b[2])
        intersectionBox[3] = min(a[3],b[3])
        intersectionBox[4] = min(a[4],b[4])
        local iw = intersectionBox[3]-intersectionBox[1]+1;
        local ih = intersectionBox[4]-intersectionBox[2]+1;
        if iw > 0 and ih > 0 then -- compute overlap as area of intersection / area of union
            local unionArea = (a[3]-a[1]+1)*(a[4]-a[2]+1) +
                (b[3]-b[1]+1)*(b[4]-b[2]+1) - iw*ih
            return (iw*ih)/unionArea
        else
            return 0
        end
    -- a,b are matrices with coordinates of multiple bounding boxes
    elseif a:nDimension()==2 and b:nDimension()==2 and a:size(2)==4 and b:size(2)==4 then
        assert(a:size(1) == b:size(1), 'The number of boxes is not the same')
        local numBoxes = a:size(1)
        local intersectionBox = torch.Tensor(numBoxes,4)
        -- we use select() to choose a single result in case of multiple
        -- occurencies of the max value
        intersectionBox[{{},1}] = max(a[{{},1}]:cat(b[{{},1}]), 2):select(2,1)
        intersectionBox[{{},2}] = max(a[{{},2}]:cat(b[{{},2}]), 2):select(2,1)
        intersectionBox[{{},3}] = min(a[{{},3}]:cat(b[{{},3}]), 2):select(2,1)
        intersectionBox[{{},4}] = min(a[{{},4}]:cat(b[{{},4}]), 2):select(2,1)
        local iw = intersectionBox[{{},3}]-intersectionBox[{{},1}]+1;
        local ih = intersectionBox[{{},4}]-intersectionBox[{{},2}]+1;
        local unionArea = (a:select(2,3)-a:select(2,1)+1):cmul(
                           a:select(2,4)-a:select(2,2)+1):add(
                           b:select(2,3)-b:select(2,1)+1):cmul(
                           b:select(2,4)-b:select(2,2)+1):csub(torch.cmul(iw,ih))
        local res = torch.mul(iw,ih):cdiv(unionArea)
        res[iw:le(0)]:zero()
        res[ih:le(0)]:zero()
        return res
    -- a and b are masks of the same size
    else
        assert(a:isSameSizeAs(b), 'Tensors must have the same size')
        local intersection = torch.sum(P.land(a,b))
        local union = torch.sum(P.lor(a,b))
        return union > 0 and intersection/union or 0
    end
end

return P
