using LsqFit, Distributions
""" 
ThresholdEstimates, DetectionRates = ConstantSimulation(Stims::Vector, pDetected::Vector, NumReps::Int;
NumPermutations::Int = 1000, NumAFC::Int = 2)

Takes the stimulus levels, the corresponding probability of detection for each stimulus, and the number of 
trials/repeats and simulates the method of constant stimulation to resolve the detection threshold.
"""
function ConstantSimulation(Stims::Vector,
                            pDetected::Vector,
                            NumReps::Int;
                            NumPermutations::Int = 1000,
                            NumAFC::Int = 2,
                            BoundSig::Bool = true)
    if NumAFC < 1
        error("NumAFC < 1")
    elseif NumAFC == 1
        chance = 0;
    else
        chance = 1/NumAFC
    end
    # Create once for comparison
    pD_Repeated = repeat(pDetected, inner=(1,NumReps))
    
    # Initialize outputs
    t_est = fill(NaN, NumPermutations)
    pd_all = fill(NaN, length(Stims), NumPermutations)
    
    for p = 1:NumPermutations
        # Get the proportion of trials where the draw is below the p(detected) at each intensity
        pd = mean(
            (rand(length(Stims), NumReps) .< pD_Repeated) .| # First draw greater than pd
            (rand(length(Stims), NumReps) .< chance) # Second draw is greater than chance
            , dims = 2)
        pd = (pd .- chance) ./ (1-chance) # Scale for chance
        pd[pd.<0] .= 0 # Remove values below 0
        pd_all[:,p] = pd
        # Get fair estimates of the detection threshold and jnd
        dt_idx = findmin(abs.(pd.-0.5))[2]
        jnd_idx = [findmin(abs.(pd.-0.25))[2], findmin(abs.(pd.-0.75))[2]]
        jnd_est = (Stims[jnd_idx[2]] - Stims[jnd_idx[1]]) / 2
        k_est = sigma2k(jnd2sigma(jnd_est))
        # Fit a sigmoid to the values
        try
            if BoundSig
                sig_fit = curve_fit(sigmoid, [0;Stims;100], vec([0;pd;1]), [k_est, Stims[dt_idx]])
            else
                sig_fit = curve_fit(sigmoid, Stims, vec(pd), [k_est, Stims[dt_idx]])
            end
            if sig_fit.param[2] > 0 && sig_fit.param[2] < 100
                t_est[p] = sig_fit.param[2]
            end
        catch
            continue
        end
    end

    return t_est, pd_all
end

function GetConstantTargets(ValidStims::Vector{Int}, pDetected::Vector{Float64};  NumStimLevels::Int = 8,
    Mode::String = "Constrained", PsychRange::Vector{Float64} = [0.2, 0.8])
    stim_idx = zeros(NumStimLevels)
    # Get indices
    if Mode == "Unconstrained" # Full range
        stim_idx = unique(Int.(round.(range(1, length(ValidStims), NumStimLevels))))
    elseif Mode == "Constrained" # Subset based on prior knowledge
        p_index_range = [findmin(abs.(pDetected.-PsychRange[1]))[2],
                         findmin(abs.(pDetected.-PsychRange[2]))[2]]
        stim_idx = unique(Int.(round.(range(p_index_range[1], p_index_range[2], NumStimLevels))))
    else
        error("Invalid mode, must be 'Constrained'/'Unconstrained'.")
    end
    # Take subset
    TargetStims = ValidStims[stim_idx]
    TargetpDetected = pDetected[stim_idx]
    
    return TargetStims, TargetpDetected
end