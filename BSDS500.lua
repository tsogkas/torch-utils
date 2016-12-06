require 'paths'
require 'image'
local matio = require 'matio'
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

function BSD:size() return self.size or #self.imageFiles end

-- Read all .jpg or .mat files in a directory 
local function readFilenames(dir)
    local list = {}
    for file in paths.files(dir) do 
        local ext = file:lower():sub(-4)
        if ext == '.jpg' or ext == '.mat'  then -- only .jpg or .mat files used in BSDS500
            list[#list+1] = file
        end
    end
    return list
end

-- Create instance that contains one of the train,val,test, or trainval subsets
function BSD:subset(set)
    -- Create new BSD object to store the subset
    local s = self.new()
    -- Select type of subset and store file names for images and labels
    if set:lower() == 'train' then 
        s.imageFiles = readFilenames(self.paths.imageTrainDir)
        s.labelFiles = readFilenames(self.paths.labelTrainDir)
    elseif set:lower() == 'val' then
        s.imageFiles = readFilenames(self.paths.imageValDir)
        s.labelFiles = readFilenames(self.paths.labelValDir)
    elseif set:lower() == 'test' then
        s.imageFiles = readFilenames(self.paths.imageTestDir)
        s.labelFiles = readFilenames(self.paths.labelTestDir)
    elseif set:lower() == 'trainval' then
        -- Concatenate filenames for train and val sets
        s.imageFiles = readFilenames(self.paths.imageTrainDir)
        local t = readFilenames(self.paths.imageValDir)
        for i=1,#t do s1.imageFiles[#s1.imageFiles+1] = t[i] end 
        -- Same for labels 
        s.labelFiles = readFilenames(self.paths.labelTrainDir)
        t = readFilenames(self.paths.labelValDir)
        for i=1,#t do s.labelFiles[#s.labelFiles+1] = t[i] end
        table.sort(s.imageFiles) -- sort results alphabetically
        table.sort(s.labelFiles)
    else error('Invalid subset') 
    end
    assert(#s > 0, 'No files found!')
    assert(#s.imageFiles == #s.labelFiles,'Image and label file count mismatch')
    return s
end

-- Shortcut methods for convenience in creating subsets 
function BSD:trainData()     return BSD:subset('train') end
function BSD:valData()       return BSD:subset('val')   end
function BSD:testData()      return BSD:subset('test')  end
function BSD:trainValData()  return BSD:subset('trainval')  end

-- Read single image
function BSD:readImage(img)
    -- If the image has already been loaded and stored, return it
    if self.images and self.images[img] then return self.images[img] end
    -- Otherwise, turn the name into a valid name string
    if torch.type(img) == 'number' then img = tostring(img)..'.jpg' end
    -- Look for it everywhere and return the set where you found it 
    if paths.filep(paths.concat(self.imageTrainDir,img)) then
        return image.load(paths.concat(self.imageTrainDir,img)), 'train'
    elseif paths.filep(paths.concat(self.imageValDir,img)) then
        return image.load(paths.concat(self.imageValDir,img)), 'val'
    elseif paths.filep(paths.concat(self.imageTestDir,img)) then
        return image.load(paths.concat(self.imageTestDir,img)), 'test'
    else error('Image file not found')
    end 
end

-- Read single label map
function BSD:readSegmentation(seg)
    -- If the image has already been loaded and stored, return it
    if self.labels and self.labels[seg] then return self.labels[seg] end
    -- Otherwise, turn the name into a valid name string
    if torch.type(seg) == 'number' then seg = tostring(seg)..'.mat' end
    -- Look for it everywhere and return the set where you found it
    local matfile,set
    if paths.filep(paths.concat(self.labelTrainDir,seg)) then
        matfile,set = matio.load(paths.concat(self.labelTrainDir,seg)), 'train'
    elseif paths.filep(paths.concat(self.labelValDir,seg)) then
        matfile,set = matio.load(paths.concat(self.labelValDir,seg)), 'val'
    elseif paths.filep(paths.concat(self.labelTestDir,seg)) then
        matfile,set = matio.load(paths.concat(self.imageTestDir,seg)), 'test'
    else error('Image file not found')
    end 
    return matfile.groundTruth, set
end

-- Store a table with all images in the set
function BSD:readAllData()
    -- Store images
    for _,file in ipairs(self.imageFiles) do
        local id = file:sub(1,-4) -- we assume that the extension is always .jpg
        self.images[id] = self:readImage(file)
    end
    -- Store labels
    for _,file in ipairs(self.labelFiles) do
        local id = file:sub(1,-4) -- we assume that the extension is always .jpg
        self.labels[id] = self:readSegmentation(file)
    end
end 
