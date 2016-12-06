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
function P.sizeof(x,dtype)
    if  torch.isTensor(x) then
        local dtype = x:type()
        local sizeOfElementInBytes
        if dtype == 'torch.DoubleTensor' or dtype == 'torch.LongTensor' then
            sizeOfElementInBytes = 8
        elseif dtype == 'torch.FloatTensor' or dtype == 'torch.IntTensor' then
            sizeOfElementInBytes = 4
        elseif dtype == 'torch.ShortTensor' then
            sizeOfElementInBytes = 2
        elseif dtype == 'torch.CharTensor' or dtype == 'torch.ByteTensor' then
            sizeOfElementInBytes = 1
        elseif dtype == 'torch.CudaTensor' or dtype == 'torch.CudaIntTensor' then
            sizeOfElementInBytes = 4
        else
            error('Unknown tensor type')
        end
        return x:nElement() * sizeOfElementInBytes
    elseif type(x) == 'string' then
        return x:len() -- each char is 1 byte (including a null \0 termination char)
    elseif type(x) == 'number' then -- if it's a number we consider  this is the size
        local dtype = dtype or 'double'
        local sizeOfElementInBytes
        if dtype == 'double' or dtype == 'long' then sizeOfElementInBytes = 8
        elseif dtype == 'int' or dtype == 'float' then sizeOfElementInBytes = 4
        elseif dtype == 'short' then sizeOfElementInBytes = 2
        elseif dtype == 'char' or dtype == 'byte' then sizeOfElementInBytes = 1
        end 
        return x * sizeOfElementInBytes
    else
        error('Only tensors, string and number types are supported!')
    end
end

-- Print size of torch tensor (adjusts for size in KB, MB, GB etc) -------------
function P.size2string(x,dtype)
    local sizeInBytes = P.sizeof(x,dtype)
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

--------------------------------------------------------------------------------
-- ARITHMETIC CHECKING AND VECTOR/MATRIX MANIPULATIONS -------------------------
--------------------------------------------------------------------------------
function P.vec(x) return x:view(-1) end
function P.isvector(x) return x:nDimension() == 1 end
function P.ismatrix(x) return x:nDimension() == 2 end
function P.isnumber(x) return type(x) == 'number' end
function P.isstring(x) return type(x) == 'string' end
function P.isinteger(x) return P.isnumber(x) and (x % 1 == 0)
                        or torch.isTensor(x) and torch.remainder(x,1):eq(0) end
function P.isodd(x) return P.isnumber(x) and (x % 2 == 1)
                    or torch.isTensor(x) and torch.remainder(x,2):eq(0) end
function P.isnan(x) return (P.isnumber(x) and x~=x) or (torch.isTensor(x) and x:ne(x)) end
function P.isinf(x) return (P.isnumber(x) and math.abs(x) == inf) 
                    or (torch.isTensor(x) and torch.abs(x):eq(math.huge)) end

--------------------------------------------------------------------------------
-------------------------- LOGICAL OPERATIONS ----------------------------------
--------------------------------------------------------------------------------
function P.any(a,dim) -- same as torch.any() but can also work dimension-wise
    return dim and a:ne(0):sum(dim):gt(0) or torch.any(d)  
end
function P.all(a,dim) -- same as torch.all() but can also work dimension-wise
    return dim and a:ne(0):sum(dim):eq(a:size(dim)) or torch.all(d) 
end
function P.land(a,b)return a:gt(0):eq(b:gt(0)) end
function P.lor(a,b) return torch.gt(a+b) end
function P.iou(a,b)
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
    -- a and b are masks of the same size
    if a:type() == 'torch.ByteTensor' and b:type() == 'torch.ByteTensor' then
        assert(a:isSameSizeAs(b), 'Tensors must have the same size')
        local intersection = torch.sum(P.land(a,b))
        local union = torch.sum(P.lor(a,b))
        return union > 0 and intersection/union or 0 
    -- a,b are bounding box coordinates in the form [xmin, xmax, ymin, ymax]
    elseif a:nDimension() == 1 and b:nDimension() == 1 then
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
    else error('Inputs have invalid format')
    end
end

return P
