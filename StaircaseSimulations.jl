using LsqFit, Distributions
function TransformedStaircaseSimulation(ValidStims::Vector{Int},  pDetected::Vector{Float64}; MaxTrials::Int = 1000, 
    NumPerms::Int = 1000, NumAFC::Int = 2, Criterion::Vector{Int} = [3,1], InitAmp::Int = 0,
    InitStepSize::Int = 20, DecreaseStepCoeff::Float64 = 0.75, IncreaseStepCoeff::Float64 = 0.75,
    MinStepSize::Int = 2, MaxStepSize::Int = 20, MaxReversions::Int = 7, SkipFirstNReversions::Int = 0)

    # Error checking
    if SkipFirstNReversions >= MaxReversions
        error("SkipFirstNReversions >= MaxReversions")
    end

    # Small computations
    afc_chance = 1/NumAFC
    num_valid_stims = length(ValidStims) 
    max_stim_level = maximum(ValidStims)
    min_stim_level = minimum(ValidStims)

    # Initialize variables for perms
    detection_history = fill(NaN, MaxTrials,NumPerms)
    amplitude_history = fill(NaN, MaxTrials,NumPerms)
    reversion_history = fill(NaN, MaxTrials,NumPerms)
    estimated_thresholds = fill(NaN, NumPerms)
    num_trials = fill(NaN, NumPerms)

    for p = 1:NumPerms
        # Initialize variables for trials
        step_size = InitStepSize
        current_direction = 0
        new_direction = 0
        consecutive_answers = [0,0]
        criterion_reached = false
        threshold_found = false
        t = 0
        num_reversions = 0

        if InitAmp == 0 # Choose a random stim
            StimAmp = ValidStims[rand(1:num_valid_stims)]
        else
            StimAmp = InitAmp
        end
        
        while t < MaxTrials && !threshold_found
            # Iterate the trial counter
            t += 1
            # Get current stim level
            stim_idx = findall(ValidStims .== StimAmp)
            if length(stim_idx) == 0 # If invalid stim then find nearest valid
                (_, stim_idx) = findmin(abs.(ValidStims .- StimAmp))
                StimAmp = ValidStims[stim_idx[1]]
            end
            amplitude_history[t,p] = StimAmp
            pd_at_stim = pDetected[stim_idx[1]] # Will error if the StimAmp is not a member of ValidStims
            
            # Determine if detected
            if rand() < pd_at_stim || (NumAFC > 1 && rand() < afc_chance)
                # Deterimine if the 'correct' interval was selected. Equal to p(detected) + 1/NumAFC
                detection_history[t,p] = 1
                consecutive_answers[1] += 1
            else
                detection_history[t,p] = 0
                consecutive_answers[2] += 1
            end

            # Evaluate criterion and change StimAmp
            if consecutive_answers[1] >= Criterion[1]
                # Criterion in a row correct means decrease amplitude
                new_direction = -1
                StimAmp = StimAmp - step_size
                consecutive_answers = [0,0]
                criterion_reached = true

            elseif consecutive_answers[2] >= Criterion[2]
                # Criterion in a row incorrect means increase amplitude
                new_direction = 1
                StimAmp = StimAmp + step_size
                consecutive_answers = [0,0]
                criterion_reached = true
            end

            # Ensure StimAmp is in range and is valid (also count as reversion/early stopping criteria)
            if StimAmp > max_stim_level
                StimAmp = max_stim_level
                reversion_history[t,p] = true
            elseif StimAmp < min_stim_level
                StimAmp = min_stim_level
                reversion_history[t,p] = true
            end

            # Update step_size (but only if we've changed stim level at least once)
            if criterion_reached && (new_direction != current_direction)
                if new_direction == 1
                    step_size = step_size * IncreaseStepCoeff
                elseif new_direction == -1
                    step_size = step_size * DecreaseStepCoeff
                end
                
                # Ensure step size is not out of bounds
                if step_size < MinStepSize
                    step_size = MinStepSize
                elseif step_size > MaxStepSize
                    step_size = MaxStepSize
                end

                # Reset boolean
                criterion_reached = false
                # Update direction
                current_direction = new_direction
                # Store reversion index
                reversion_history[t,p] = true
            end

            if reversion_history[t,p] == true
                num_reversions += 1
            end

            # Stopping criteria
            if num_reversions >= MaxReversions
                rev_idx = findall(isnan.(reversion_history[:,p]) .== 0)[1+SkipFirstNReversions:MaxReversions]
                rev_amps = amplitude_history[rev_idx,p]
                estimated_thresholds[p] = mean(rev_amps)
                num_trials[p] = maximum(rev_idx)
                threshold_found = true # End while loop
            end
        end
    end

    return amplitude_history, detection_history, reversion_history, estimated_thresholds, num_trials
end

function GetTransformedStaircaseTarget(NumAFC::Int, Criterion::Vector{Int})
    Target = Float64[]
    if NumAFC > 1 # Need to adjust for chance
        chance = 1/NumAFC
        if Criterion[1] >= Criterion[2] && Criterion[2] == 1 
            NR = 0.5^(1/(Criterion[1]/Criterion[2])) # N-th root
            Target = (NR - chance) / (1-chance)
        else
            error("Criterion[1] must be greater than Criterion[2]")
        end
    else
        Target = Criterion[1]/sum(Criterion)
    end
    
    return Target
end

function PosthocEstimateStaircaseThreshold(TrialAmplitudes::Matrix{Int}, ReversionIndices::Matrix{Float64},
     MaxReversions::Int; SkipFirstNReversions::Int = 0)

    # Error checking
    if SkipFirstNReversions < 0
         error("SkipFirstNReversions < 0")
    end
    if MaxReversions <= SkipFirstNReversions
        error("MaxReversions <= SkipFirstNReversions")
    end
    if size(ReversionIndices) != size(TrialAmplitudes)
        error("size(ReversionIndices) != size(TrialAmplitudes)")
    end

    NumPerms = size(TrialAmplitudes,2)
    # Initalize output
    EstimatedThreshold = zeros(NumPerms)
    for p = 1:NumPerms
        # Find the first MaxReversions indices of the ReversionIndices
        rev_idx = findall(isnan.(ReversionIndices[:,p]) .== 0)[1+SkipFirstNReversions:MaxReversions]
        rev_amps = TrialAmplitudes[rev_idx,p]
        EstimatedThreshold[p] = mean(rev_amps)
    end

    return EstimatedThreshold
end

function SortedStaircaseStatistics(ThresholdEstimate::Matrix{Float64}, StaircaseStopPoint::Matrix{Float64},
     ThresholdGroundTruth::Float64)

    # First vectorize inputs
    ThresholdEstimate = vec(ThresholdEstimate)
    StaircaseStopPoint = vec(StaircaseStopPoint)
    # Then remove NaN values
    nan_idx = findall(isnan.(ThresholdEstimate) .== 0)
    ThresholdEstimate = ThresholdEstimate[nan_idx]
    StaircaseStopPoint = Int.(StaircaseStopPoint[nan_idx])

    # Find the minimum and maximum number of reversion
    min_repeats = nanminimum(StaircaseStopPoint)
    max_repeats = nanmaximum(StaircaseStopPoint)

    # Initalize outputs
    NumRepeats = min_repeats:max_repeats
    SortedMean = zeros(length(NumRepeats))
    SortedSTD = zeros(length(NumRepeats))
    SortedError = zeros(length(NumRepeats))

    # Find all values matching the number of repeats and perform stats on them
    for (ri, r) in enumerate(NumRepeats)
        r_idx = findall(StaircaseStopPoint .== r)
        SortedMean[ri] = mean(ThresholdEstimate[r_idx])
        SortedSTD[ri] = std(ThresholdEstimate[r_idx])
        SortedError[ri] = mean(abs.(ThresholdEstimate[r_idx].- ThresholdGroundTruth))
    end

    return NumRepeats, SortedMean, SortedSTD, SortedError
end