using LsqFit, Distributions
function ConstantSimulation(Stims::Vector, pDetected::Vector, NumTrials::Int; NumPerms::Int = 1000, NumAFC::Int = 2)

    t_est = Vector(undef, NumPerms); fill!(t_est, NaN)
    pd = zeros(length(Stims))
    sigmoid(x::Vector, coeffs::Vector) = 1 ./ (1 .+ exp.(-coeffs[1].*(x.-coeffs[2])))
    
    for p = 1:NumPerms
        # Get the proportion of trials where the draw is below the p(detected) at each intensity
        pd = mean(
            (rand(length(Stims), NumTrials) .< repeat(pDetected, inner=(1,NumTrials))) .| # First draw greater than pd
            (rand(length(Stims), NumTrials) .< 1/NumAFC) # Second draw is greater than chance
            , dims = 2)
        pd = (pd .- 1/NumAFC) ./ (1/NumAFC) # Scale for chance
        pd[pd.<0] .= 0 # Remove values below 0

        # Get fair estimates of the detection threshold and jnd
        dt_idx = findmin(abs.(pd.-0.5))[2]
        jnd_idx = [findmin(abs.(pd.-0.25))[2], findmin(abs.(pd.-0.75))[2]]
        jnd_est = (Stims[jnd_idx[2]] - Stims[jnd_idx[1]]) / 2
        k_est = sigma2k(jnd2sigma(jnd_est))
        # Fit a sigmoid to the values
        try
            sig_fit = curve_fit(sigmoid, [0;Stims;100], vec([0;pd;1]), [k_est, Stims[dt_idx]])
            if sig_fit.param[2] > 0 && sig_fit.param[2] < 100
                t_est[p] = sig_fit.param[2]
            end
        catch
            continue
        end
    end
    if NumPerms == 1
        return pd
    else
        return t_est
    end
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