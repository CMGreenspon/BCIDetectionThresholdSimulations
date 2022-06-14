using LsqFit, Distributions
function TransformedStaircaseSimulation(ValidStims::Vector{Int},  pDetected::Vector, MaxTrials::Int = 50, 
    NumReversion::Int = 5, NumPerms::Int = 1000, NumAFC::Int = 2)