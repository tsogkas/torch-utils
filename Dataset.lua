require "paths"
require "image"
local Dataset = torch.class('Dataset')

function Dataset:__init(path)
    self.paths = {}
    if paths.dirp(path) then
        self.paths.rootDir = path
    else
        error('Given path does not correspond to an existing directory')
    end
    self:__setPaths(path)
end

function Dataset:__setPaths(rootDir)
    -- This function should be set individually for each of the inhereted classes
    --[[
    self.paths.images = {}
    self.paths.labels = {}
    self.paths.images.train = {}
    self.paths.images.val   = {}
    self.paths.images.test  = {}
    self.paths.labels.train = {}
    self.paths.labels.val   = {}
    self.paths.labels.test  = {}
    --]]
end
