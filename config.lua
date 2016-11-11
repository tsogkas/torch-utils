--[[
Global configuration options for YOUR_PROJECT.
--]]

local P = {}; config = P;
require "paths"

-- Directory paths
config.rootDir      = ""
config.logsDir      = paths.concat(config.rootDir, "logs")
config.modelsDir    = paths.concat(config.rootDir, "models")
config.dataDir      = paths.concat(config.rootDir, "data")
config.externalDir  = paths.concat(config.rootDir, "external")

-- Configuration parameters
-- ADD YOUR PARAMETERS HERE

return P
