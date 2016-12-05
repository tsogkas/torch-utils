require 'paths'
require 'image'
local BSD,parent = torch.class('BSDS500','Dataset')

function BSD:__init(path)
    parent:__init(self)
end

function BSD:__setPaths(rootDir)
    self.paths.imageDir      = paths.concat(rootDir,'images')
    self.paths.labelDir      = paths.concat(rootDir,'groundtruth')
    self.paths.imageTrainDir = paths.concat(self.paths.imageDir,'train')
    self.paths.imageValDir   = paths.concat(self.paths.imageDir,'val')
    self.paths.imageTestDir  = paths.concat(self.paths.imageDir,'test')
    self.paths.labelTrainDir = paths.concat(self.paths.labelDir,'train')
    self.paths.labelValDir   = paths.concat(self.paths.labelDir,'val')
    self.paths.labelTestDir  = paths.concat(self.paths.labelDir,'test')
end


function BSD:paths()
    if self.paths.rootDir  then print('Root directory:   '..self.paths.rootDir) end
    if self.paths.imageDir then print('Images directory: '..self.paths.imageDir) end
    if self.paths.labelDir then print('Labels directory: '..self.paths.labelDir) end
end

function BSD:size()
    return self.size or #self.imageFiles
end

local function readFilenames(dir)
    local list = {}
    for file in paths.files(dir) do 
        local ext = file:lower():sub(-4)
        if ext == '.jpg'  then
            table.insert(list,file)
        end
    end
    return list
end

function BSD:subset(set)
    -- Select type of subset
    local suffix
    if set:lower() == 'train' then suffix = 'TrainDir'
    elseif set:lower() == 'val' then suffix = 'ValDir'
    elseif set:lower() == 'test' then suffix = 'TestDir'
    else error('Invalid subset') end
    -- Create new BSD object to store the subset
    local s = self.new()
    -- Store file names for images and labels
    s.imageFiles = table.sort(readFilenames(self['image'..suffix]))
    s.labelFiles = table.sort(readFilenames(self['label'..suffix]))
    if #s == 0 then error('No files found!') end
    assert(#s.imageFiles == #s.labelFiles,'Image and label file count mismatch')
    return s
end

function BSD:readImage(img)
    if torch.type(img) == 'number' then img = tostring(img)..'.jpg' end 
    if paths.filep(paths.concat(self.imageTrainDir,img)) then
        img = paths.concat(self.imageTrainDir,img)
    elseif paths.filep(paths.concat(self.imageValDir,img)) then
        img = paths.concat(self.imageValDir,img)
    elseif paths.filep(paths.concat(self.imageTestDir,img)) then
        img = paths.concat(self.imageTestDir,img)
    else error('Image file not found')
    end 
    return image.load(img)
end

function BSD:shuffle()
end

function BSD:computeStats()
end